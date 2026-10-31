"""Tests for the market data pipeline, on synthetic quotes so no network is needed.

Pricing a synthetic chain with known inputs means the right answer is known
exactly, so each step of the pipeline can be checked against it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bs_lab import market_data as md
from bs_lab.market_data import (
    at_the_money_vol,
    build_smile,
    fetch_option_chain,
    flat_vol_gap,
    implied_forward,
    parity_misses,
    load_bloomberg_csv,
    load_chain,
    obtain_snapshot,
    return_statistics,
    save_chain,
    term_summary,
    year_fraction,
)
from bs_lab.pricing import bs_price

FORWARD, RATE = 5000.0, 0.04
EXPIRY = pd.Timestamp("2027-06-18")
QUOTE_TIME = pd.Timestamp("2027-03-18 16:00", tz="America/New_York")
T = year_fraction(EXPIRY, QUOTE_TIME)
DISCOUNT = float(np.exp(-RATE * T))
STRIKES = np.arange(3500.0, 6505.0, 25.0)


def skewed_vol(strikes: np.ndarray) -> np.ndarray:
    """A downward-sloping smile: lower strikes carry higher vol."""
    return 0.18 - 0.30 * np.log(strikes / FORWARD)


def make_chain(vol_of_strike=lambda k: np.full_like(k, 0.2), half_spread: float = 0.1) -> pd.DataFrame:
    """Price calls and puts on one expiry and quote them around fair value."""
    rows = []
    vols = vol_of_strike(STRIKES)
    for option_type in ("call", "put"):
        fair = bs_price(FORWARD, STRIKES, T, RATE, vols, option_type, RATE)
        rows.append(
            pd.DataFrame(
                {
                    "strike": STRIKES,
                    "option_type": option_type,
                    "bid": np.maximum(fair - half_spread, 0.0),
                    "ask": fair + half_spread,
                }
            )
        )
    chain = pd.concat(rows, ignore_index=True)
    chain = chain.assign(underlying="TEST", quote_time=QUOTE_TIME, spot=np.nan, expiry=EXPIRY)
    for column in md.CHAIN_COLUMNS:
        if column not in chain:
            chain[column] = np.nan
    return chain[md.CHAIN_COLUMNS]


def set_quote(chain: pd.DataFrame, strike: float, option_type: str, bid: float, ask: float) -> None:
    row = (chain["strike"] == strike) & (chain["option_type"] == option_type)
    chain.loc[row, ["bid", "ask"]] = [bid, ask]


def test_year_fraction_counts_calendar_time_to_the_close():
    expiry = pd.Timestamp("2027-01-29")
    one_year_earlier = pd.Timestamp("2026-01-29 16:00", tz="America/New_York")
    assert year_fraction(expiry, one_year_earlier) == pytest.approx(1.0)


def test_implied_forward_recovers_forward_and_discount():
    fit = implied_forward(make_chain())
    assert fit.forward == pytest.approx(FORWARD, abs=0.05)
    assert fit.discount == pytest.approx(DISCOUNT, abs=1e-5)
    assert fit.rate == pytest.approx(RATE, abs=1e-3)
    assert fit.bad_strikes == ()


def test_single_bad_quote_is_flagged_and_does_not_move_the_forward():
    chain = make_chain()
    fair = chain[(chain["strike"] == 5100.0) & (chain["option_type"] == "call")]["bid"].iloc[0]
    set_quote(chain, 5100.0, "call", fair + 90.0, fair + 90.2)
    fit = implied_forward(chain)
    assert 5100.0 in fit.bad_strikes
    assert fit.forward == pytest.approx(FORWARD, abs=0.1)


def test_stale_pair_from_before_a_market_move_is_flagged():
    """Both sides of a strike left at an old forward still break parity."""
    chain = make_chain()
    for option_type in ("call", "put"):
        stale = float(bs_price(FORWARD + 120.0, 4900.0, T, RATE, 0.2, option_type, RATE))
        set_quote(chain, 4900.0, option_type, stale - 0.1, stale + 0.1)
    assert 4900.0 in implied_forward(chain).bad_strikes


def test_parity_misses_show_the_size_of_a_stale_quote():
    chain = make_chain()
    for option_type in ("call", "put"):
        stale = float(bs_price(FORWARD + 120.0, 4900.0, T, RATE, 0.2, option_type, RATE))
        set_quote(chain, 4900.0, option_type, stale - 0.1, stale + 0.1)
    misses = parity_misses(chain, implied_forward(chain))
    assert misses.loc[4900.0] == pytest.approx(120.0 * DISCOUNT, abs=0.5)
    assert misses.drop(4900.0).abs().max() < 0.5


def test_empty_quote_with_near_zero_gap_cannot_hijack_the_forward():
    """Regression: a far-OTM strike with C - P near 0 was once taken as the forward."""
    chain = make_chain()
    set_quote(chain, 6500.0, "put", 0.0, 0.15)
    set_quote(chain, 6500.0, "call", 0.05, 0.10)
    assert implied_forward(chain).forward == pytest.approx(FORWARD, abs=0.1)


def test_too_few_quotes_raises():
    chain = make_chain()
    with pytest.raises(ValueError):
        implied_forward(chain[chain["strike"] < 3600.0])


def test_smile_of_a_flat_vol_market_is_flat():
    smile = build_smile(make_chain())
    np.testing.assert_allclose(smile["iv"], 0.2, atol=1e-3)


def test_smile_recovers_a_known_skew():
    smile = build_smile(make_chain(skewed_vol))
    np.testing.assert_allclose(smile["iv"], skewed_vol(smile["strike"].to_numpy()), atol=2e-3)


def test_smile_keeps_only_out_of_the_money_options():
    smile = build_smile(make_chain())
    assert (smile.loc[smile["option_type"] == "put", "strike"] < FORWARD).all()
    assert (smile.loc[smile["option_type"] == "call", "strike"] >= FORWARD).all()


def test_smile_excludes_strikes_that_fail_parity():
    chain = make_chain()
    set_quote(chain, 5100.0, "call", 400.0, 400.2)
    assert 5100.0 not in build_smile(chain)["strike"].to_numpy()


def test_smile_excludes_strikes_quoted_on_one_side_only():
    chain = make_chain()
    chain = chain[~((chain["strike"] == 5200.0) & (chain["option_type"] == "put"))]
    assert 5200.0 not in build_smile(chain)["strike"].to_numpy()


def test_iv_bid_and_ask_bracket_the_mid_vol():
    smile = build_smile(make_chain(skewed_vol))
    assert (smile["iv_bid"] <= smile["iv"]).all() and (smile["iv"] <= smile["iv_ask"]).all()


def test_at_the_money_vol_reads_the_smile_at_the_forward():
    smile = build_smile(make_chain(skewed_vol))
    assert at_the_money_vol(smile).iloc[0] == pytest.approx(0.18, abs=2e-3)


def test_term_summary_reports_a_known_skew():
    row = term_summary(build_smile(make_chain(skewed_vol)), spot=FORWARD).iloc[0]
    assert row["atm_vol"] == pytest.approx(0.18, abs=2e-3)
    assert row["skew_90_100"] == pytest.approx(-0.30 * np.log(0.9), abs=2e-3)


def test_term_summary_gives_nan_beyond_the_quoted_strikes():
    chain = make_chain(skewed_vol)
    smile = build_smile(chain[chain["strike"] <= 5250.0])
    assert np.isnan(term_summary(smile, spot=FORWARD).iloc[0]["vol_110"])


def test_flat_vol_gap_vanishes_in_a_flat_market():
    gaps = flat_vol_gap(build_smile(make_chain()))
    np.testing.assert_allclose(gaps["ratio"], 1.0, atol=0.02)


def test_flat_vol_underprices_puts_and_overprices_calls_under_skew():
    gaps = flat_vol_gap(build_smile(make_chain(skewed_vol)))
    assert (gaps.loc[gaps["moneyness"] < 0.97, "ratio"] > 1.0).all()
    assert (gaps.loc[gaps["moneyness"] > 1.03, "ratio"] < 1.0).all()


def test_chain_survives_a_csv_round_trip(tmp_path):
    chain = make_chain()
    chain["quote_time"] = chain["quote_time"].dt.tz_convert("UTC")
    path = tmp_path / "chain.csv"
    save_chain(chain, path)
    loaded = load_chain(path)
    assert loaded["quote_time"].iloc[0] == chain["quote_time"].iloc[0]
    assert build_smile(loaded)["iv"].mean() == pytest.approx(0.2, abs=1e-3)


def write_bloomberg_file(path, chain: pd.DataFrame) -> None:
    """Write a chain the way a Bloomberg export might label it."""
    pd.DataFrame(
        {
            "Expiry": chain["expiry"].dt.strftime("%m/%d/%Y"),
            "Strike": chain["strike"].map("{:,.2f}".format),
            "Type": chain["option_type"].str[0].str.upper(),
            "Bid": chain["bid"],
            "Ask": chain["ask"],
            "Volm": 10,
            "IVM": 20.0,
        }
    ).to_csv(path, index=False)


def test_bloomberg_csv_feeds_the_same_pipeline(tmp_path):
    path = tmp_path / "bbg.csv"
    write_bloomberg_file(path, make_chain())
    quotes = load_bloomberg_csv(path, quote_time="2027-03-18 16:00", spot=5000.0, underlying="SPX Index")
    assert list(quotes.columns) == md.CHAIN_COLUMNS
    assert set(quotes["option_type"]) == {"call", "put"}
    assert quotes["vendor_iv"].iloc[0] == pytest.approx(0.20)
    assert quotes["strike"].max() == 6500.0  # Thousands separators are handled.
    assert build_smile(quotes)["iv"].mean() == pytest.approx(0.2, abs=1e-3)


def test_bloomberg_csv_accepts_a_custom_column_map(tmp_path):
    path = tmp_path / "bbg.csv"
    write_bloomberg_file(path, make_chain())
    raw = pd.read_csv(path).rename(columns={"Strike": "Exercise Px"})
    raw.to_csv(path, index=False)
    quotes = load_bloomberg_csv(path, "2027-03-18 16:00", column_map={"Exercise Px": "strike"})
    assert quotes["strike"].notna().all()


def test_bloomberg_csv_takes_the_expiry_as_an_argument(tmp_path):
    path = tmp_path / "bbg.csv"
    write_bloomberg_file(path, make_chain())
    pd.read_csv(path).drop(columns="Expiry").to_csv(path, index=False)
    quotes = load_bloomberg_csv(path, "2027-03-18 16:00", expiry="2027-06-18")
    assert quotes["expiry"].iloc[0] == EXPIRY


def test_bloomberg_csv_reports_missing_columns(tmp_path):
    path = tmp_path / "bbg.csv"
    pd.DataFrame({"Strike": [100.0], "Bid": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_bloomberg_csv(path, "2027-03-18")


def test_bloomberg_csv_rejects_unknown_option_types(tmp_path):
    path = tmp_path / "bbg.csv"
    pd.DataFrame(
        {"Expiry": ["2027-06-18"], "Strike": [100.0], "Type": ["X"], "Bid": [1.0], "Ask": [2.0]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="option_type"):
        load_bloomberg_csv(path, "2027-03-18")


FIXTURE = Path(__file__).parent / "fixtures" / "bloomberg_example.csv"


def test_documented_example_file_loads_and_gives_a_smile():
    quotes = load_bloomberg_csv(FIXTURE, quote_time="2027-03-18 15:00", spot=4975.0, underlying="TEST Index")
    assert quotes["expiry"].nunique() == 2
    assert set(quotes["option_type"]) == {"call", "put"}
    assert quotes["vendor_iv"].between(0.05, 0.60).all()  # Points converted to fractions.
    assert quotes["contract"].iloc[0].startswith("TEST")
    smile = build_smile(quotes)
    assert len(smile) >= 10
    assert smile["forward"].iloc[0] == pytest.approx(5000.0, abs=1.0)
    expected = 0.20 - 0.30 * np.log(smile["strike"] / 5000.0)
    np.testing.assert_allclose(smile["iv"], expected, atol=0.01)


def write_rows(path, header: str, *rows: str) -> None:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


HEADER = "Expiry,Strike,Type,Bid,Ask"
QUOTE_TIME_TEXT = "2027-03-18 15:00"


@pytest.mark.parametrize(
    "header, rows, kwargs, message",
    [
        pytest.param("Expiry;Strike;Type;Bid;Ask", ["06/18/2027;100;C;1;2"], {}, "separator", id="wrong-separator"),
        pytest.param("Bloomberg report\n" + HEADER, ["06/18/2027,100,C,1,2"], {}, "title lines", id="title-line"),
        pytest.param(HEADER, [], {}, "no data rows", id="no-rows"),
        pytest.param("Expiry,Strike,Type,Bid", ["06/18/2027,100,C,1"], {}, r"Missing columns \['ask'\]", id="missing-column"),
        pytest.param(HEADER, ['06/18/2027,100,C,"12,50",13'], {}, "Column 'bid'.*'12,50'", id="decimal-comma"),
        pytest.param(HEADER, ["06/18/2027,abc,C,1,2"], {}, "Column 'strike'", id="text-strike"),
        pytest.param(HEADER, ["06/18/2027,,C,1,2"], {}, "positive strike", id="blank-strike"),
        pytest.param(HEADER, ["06/18/2027,100,Cash,1,2"], {}, "cash", id="unknown-type"),
        pytest.param(HEADER, ["18/06/2027,100,C,1,2"], {}, "day-first", id="day-first-date"),
        pytest.param(HEADER, ["June 2027,100,C,1,2"], {}, "expiry dates", id="unreadable-date"),
        pytest.param(HEADER, ["06/18/2026,100,C,1,2"], {}, "not after quote_time", id="expired"),
        pytest.param(HEADER, ["06/18/2027,100,C,1,2", "06/18/2027,100,C,1,2"], {}, "appear twice", id="duplicates"),
        pytest.param(HEADER, ["06/18/2027,100,C,1,2"], {"column_map": {"Nope": "strike"}}, "column_map names", id="map-header"),
        pytest.param(HEADER, ["06/18/2027,100,C,1,2"], {"column_map": {"Bid": "price"}}, "column_map targets", id="map-target"),
    ],
)
def test_bloomberg_format_problems_are_explained(tmp_path, header, rows, kwargs, message):
    path = tmp_path / "bbg.csv"
    write_rows(path, header, *rows)
    with pytest.raises(md.BloombergFormatError, match=message):
        load_bloomberg_csv(path, QUOTE_TIME_TEXT, **kwargs)


def test_bloomberg_empty_file_is_explained(tmp_path):
    path = tmp_path / "bbg.csv"
    path.write_text("")
    with pytest.raises(md.BloombergFormatError, match="is empty"):
        load_bloomberg_csv(path, QUOTE_TIME_TEXT)


def test_bloomberg_wrong_encoding_is_explained(tmp_path):
    path = tmp_path / "bbg.csv"
    path.write_bytes(f"{HEADER}\n".encode("utf-16"))
    with pytest.raises(md.BloombergFormatError, match="UTF-8"):
        load_bloomberg_csv(path, QUOTE_TIME_TEXT)


def test_bloomberg_unreadable_quote_time_is_explained(tmp_path):
    path = tmp_path / "bbg.csv"
    write_rows(path, HEADER, "06/18/2027,100,C,1,2")
    with pytest.raises(md.BloombergFormatError, match="quote_time"):
        load_bloomberg_csv(path, "not a time")


def test_bloomberg_error_is_a_value_error_so_callers_can_catch_either(tmp_path):
    assert issubclass(md.BloombergFormatError, ValueError)


def test_bloomberg_accepts_a_semicolon_separator(tmp_path):
    path = tmp_path / "bbg.csv"
    write_rows(path, "Expiry;Strike;Type;Bid;Ask", "06/18/2027;100;C;1;2")
    assert load_bloomberg_csv(path, QUOTE_TIME_TEXT, sep=";")["strike"].iloc[0] == 100.0


def test_bloomberg_accepts_a_day_first_format_when_told(tmp_path):
    path = tmp_path / "bbg.csv"
    write_rows(path, HEADER, "18/06/2027,100,C,1,2")
    quotes = load_bloomberg_csv(path, QUOTE_TIME_TEXT, date_format="%d/%m/%Y")
    assert quotes["expiry"].iloc[0] == EXPIRY


@pytest.mark.parametrize("text", ["06/18/2027", "06/18/27", "2027-06-18"])
def test_bloomberg_accepts_the_documented_date_formats(tmp_path, text):
    path = tmp_path / "bbg.csv"
    write_rows(path, HEADER, f"{text},100,C,1,2")
    assert load_bloomberg_csv(path, QUOTE_TIME_TEXT)["expiry"].iloc[0] == EXPIRY


def test_bloomberg_excel_byte_order_mark_is_ignored(tmp_path):
    path = tmp_path / "bbg.csv"
    path.write_bytes(f"{HEADER}\n06/18/2027,100,C,1,2\n".encode("utf-8-sig"))
    assert load_bloomberg_csv(path, QUOTE_TIME_TEXT)["expiry"].notna().all()


def test_bloomberg_blank_quotes_mean_no_quote_rather_than_an_error(tmp_path):
    path = tmp_path / "bbg.csv"
    write_rows(path, HEADER, "06/18/2027,100,C,,#N/A N/A", "06/18/2027,110,C,-,N/A")
    quotes = load_bloomberg_csv(path, QUOTE_TIME_TEXT)
    assert quotes["bid"].isna().all() and quotes["ask"].isna().all()


def test_bloomberg_field_mnemonics_are_recognised(tmp_path):
    path = tmp_path / "bbg.csv"
    write_rows(
        path,
        "OPT_EXPIRE_DT,OPT_STRIKE_PX,OPT_PUT_CALL,PX_BID,PX_ASK,PX_LAST,PX_VOLUME,OPEN_INT,IVOL_MID",
        "2027-06-18,100,Put,1,2,1.5,3,4,20",
    )
    row = load_bloomberg_csv(path, QUOTE_TIME_TEXT).iloc[0]
    assert (row["strike"], row["option_type"], row["last"], row["volume"]) == (100.0, "put", 1.5, 3.0)
    assert row["open_interest"] == 4.0 and row["vendor_iv"] == pytest.approx(0.20)


def test_nearest_expiries_never_repeat():
    listed = ["2027-01-15", "2027-02-19", "2027-03-19"]
    chosen = md._nearest_expiries(listed, (30, 31, 60), pd.Timestamp("2026-12-20"))
    assert chosen == ["2027-01-15", "2027-02-19"]


def test_fetch_option_chain_normalises_yahoo_columns(monkeypatch):
    """Run the download code against a stand-in for the Yahoo client."""

    class FakeChain:
        calls = pd.DataFrame(
            {
                "contractSymbol": ["C1"],
                "lastTradeDate": [QUOTE_TIME],
                "strike": [100.0],
                "lastPrice": [2.0],
                "bid": [1.9],
                "ask": [2.1],
                "volume": [5.0],
                "openInterest": [7.0],
                "impliedVolatility": [0.2],
            }
        )
        puts = calls.assign(contractSymbol="P1")

    class FakeTicker:
        options = ("2030-01-18",)
        fast_info = {"lastPrice": 101.0}

        def __init__(self, ticker):
            self.ticker = ticker

        def option_chain(self, expiry):
            return FakeChain()

    monkeypatch.setattr(md.yf, "Ticker", FakeTicker)
    quotes = fetch_option_chain("TEST", target_days=(30,))
    assert list(quotes.columns) == md.CHAIN_COLUMNS
    assert sorted(quotes["option_type"]) == ["call", "put"]
    assert quotes["spot"].iloc[0] == 101.0


def write_snapshot(folder, date: str) -> None:
    chain = make_chain()
    chain["quote_time"] = chain["quote_time"].dt.tz_convert("UTC")
    save_chain(chain, folder / f"spx_options_{date}.csv")
    pd.Series([1.0, 2.0], index=pd.to_datetime(["2027-03-17", "2027-03-18"]), name="close").to_csv(
        folder / f"spx_daily_close_{date}.csv"
    )
    pd.DataFrame({"ticker": ["^IRX"], "tenor": ["13-week T-bill"], "yield_pct": [4.0], "date": [date]}).to_csv(
        folder / f"treasury_yields_{date}.csv", index=False
    )


def fail_if_called(*args, **kwargs):
    raise AssertionError("The network must not be used")


def stub_download(monkeypatch, chain: pd.DataFrame) -> None:
    close = pd.Series([1.0, 2.0], index=pd.to_datetime(["2027-03-17", "2027-03-18"]), name="close")
    treasuries = pd.DataFrame({"ticker": ["^IRX"], "tenor": ["t"], "yield_pct": [4.0], "date": ["d"]})
    monkeypatch.setattr(md, "fetch_option_chain", lambda ticker: chain)
    monkeypatch.setattr(md, "fetch_price_history", lambda ticker: close)
    monkeypatch.setattr(md, "fetch_treasury_yields", lambda date: treasuries)


def utc_chain() -> pd.DataFrame:
    chain = make_chain()
    chain["quote_time"] = chain["quote_time"].dt.tz_convert("UTC")
    return chain


def test_saved_reference_snapshot_is_used_without_the_network(tmp_path, monkeypatch):
    write_snapshot(tmp_path, "2026-10-30")
    monkeypatch.setattr(md, "fetch_option_chain", fail_if_called)
    snapshot = obtain_snapshot(tmp_path, "2026-10-30")
    assert snapshot.date == "2026-10-30" and not snapshot.downloaded
    assert len(snapshot.quotes) == len(make_chain())


def test_partial_snapshot_triggers_a_download(tmp_path, monkeypatch):
    write_snapshot(tmp_path, "2026-10-30")
    (tmp_path / "treasury_yields_2026-10-30.csv").unlink()
    stub_download(monkeypatch, utc_chain())
    assert obtain_snapshot(tmp_path, "2026-10-30").downloaded


def test_missing_snapshot_is_downloaded_and_saved_locally(tmp_path, monkeypatch):
    stub_download(monkeypatch, utc_chain())
    data_dir = tmp_path / "new_folder"
    snapshot = obtain_snapshot(data_dir, "2026-10-30")
    assert snapshot.downloaded and snapshot.date != "2026-10-30"
    assert len(list(data_dir.glob("*.csv"))) == 3
    assert build_smile(snapshot.quotes)["iv"].mean() == pytest.approx(0.2, abs=1e-3)


def test_failed_download_raises_a_clear_error_and_saves_nothing(tmp_path, monkeypatch):
    def no_network(ticker):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(md, "fetch_option_chain", no_network)
    with pytest.raises(md.MarketDataUnavailable, match="ConnectionError: no route to host"):
        obtain_snapshot(tmp_path, "2026-10-30")
    assert list(tmp_path.glob("*.csv")) == []


def test_unusable_quotes_count_as_unavailable(tmp_path, monkeypatch):
    """With the market closed, Yahoo can return a chain whose bids are all zero."""
    stub_download(monkeypatch, utc_chain().assign(bid=0.0, ask=0.0))
    with pytest.raises(md.MarketDataUnavailable, match="ValueError"):
        obtain_snapshot(tmp_path, "2026-10-30")
    assert list(tmp_path.glob("*.csv")) == []


def test_return_statistics_see_nothing_unusual_in_gaussian_returns():
    returns = np.random.default_rng(0).normal(0.0, 0.01, 200_000)
    stats = return_statistics(pd.Series(100.0 * np.exp(np.cumsum(returns))))
    assert stats["annual_vol"] == pytest.approx(0.01 * np.sqrt(252), rel=0.01)
    assert stats["excess_kurtosis"] == pytest.approx(0.0, abs=0.1)
    assert stats["days_beyond_4sd"] == pytest.approx(stats["expected_days_beyond_4sd"], rel=0.5)


def test_return_statistics_detect_a_crash():
    returns = np.random.default_rng(1).normal(0.0, 0.01, 5_000)
    returns[2_500] = -0.20
    stats = return_statistics(pd.Series(100.0 * np.exp(np.cumsum(returns))))
    assert stats["worst_day"] == pytest.approx(-0.20)
    assert stats["excess_kurtosis"] > 10.0
    assert stats["worst_day_in_sd"] < -10.0
