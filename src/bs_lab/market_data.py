"""Load option quotes and turn them into an implied volatility smile.

Both sources, Yahoo Finance and a Bloomberg CSV export, produce the same table
(CHAIN_COLUMNS), so the analysis code does not care where quotes came from.
Snapshots are saved to CSV because a live option chain cannot be downloaded
again later. They stay on the local machine: Yahoo's terms cover personal use,
so raw quotes are not published with this repository.

From quotes to a smile
----------------------
1. Use bid and ask only. The last trade can be days old.
2. Infer the forward and discount factor from put-call parity, C - P = D (F - K).
   This needs no spot, rate or dividend input, and it avoids mixing a spot and
   option quotes taken at different moments, which would distort every vol.
3. Keep only strikes that pass parity, within what noise and spreads can explain.
4. Keep out-of-the-money options, which are the liquid ones. In-the-money
   prices are mostly intrinsic value and carry little information about vol.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, theilslopes

from bs_lab.implied_vol import solve_implied_vol
from bs_lab.pricing import bs_price

CHAIN_COLUMNS = [
    "underlying",
    "quote_time",
    "spot",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "last",
    "last_trade",
    "volume",
    "open_interest",
    "contract",
    "vendor_iv",
]

MARKET_TIMEZONE = "America/New_York"
SNAPSHOT_TICKER = "^SPX"
TREASURY_TICKERS = {"^IRX": "13-week T-bill", "^FVX": "5-year Treasury"}

# Spelling variants accepted in a Bloomberg export, lower-cased.
BLOOMBERG_ALIASES = {
    "expiry": ["expiry", "exp", "expiration", "expiration date", "expiry date", "opt_expire_dt"],
    "strike": ["strike", "strike px", "strike price", "opt_strike_px"],
    "option_type": ["option_type", "type", "call/put", "put/call", "c/p", "opt_put_call"],
    "bid": ["bid", "px_bid", "bid price"],
    "ask": ["ask", "px_ask", "ask price", "offer"],
    "last": ["last", "px_last", "last price"],
    "volume": ["volume", "volm", "px_volume"],
    "open_interest": ["open_interest", "openint", "open int", "oi", "open_int"],
    "vendor_iv": ["vendor_iv", "ivm", "iv mid", "imp vol", "ivol_mid"],
    "contract": ["contract", "ticker", "security"],
}


def _nearest_expiries(listed: Sequence[str], target_days: Sequence[int], today: pd.Timestamp) -> list[str]:
    """Pick the listed expiry closest to each target horizon, without repeats."""
    dates = pd.to_datetime(list(listed))
    days_out = (dates - today).days.to_numpy()
    chosen = {listed[int(np.argmin(np.abs(days_out - target)))] for target in target_days}
    return sorted(chosen)


def fetch_option_chain(
    ticker: str,
    target_days: Sequence[int] = (30, 91, 182, 365),
) -> pd.DataFrame:
    """Download the quotes for the expiries nearest to the target horizons.

    Yahoo quotes are delayed and carry no guarantee of being simultaneous, so
    treat them as an approximation of the live market.
    """
    handle = yf.Ticker(ticker)
    quote_time = pd.Timestamp(datetime.now(timezone.utc))
    spot = float(handle.fast_info["lastPrice"])
    expiries = _nearest_expiries(handle.options, target_days, quote_time.tz_localize(None))

    frames = []
    for expiry in expiries:
        chain = handle.option_chain(expiry)
        for option_type, table in (("call", chain.calls), ("put", chain.puts)):
            frames.append(
                table.rename(
                    columns={
                        "lastPrice": "last",
                        "lastTradeDate": "last_trade",
                        "openInterest": "open_interest",
                        "contractSymbol": "contract",
                        "impliedVolatility": "vendor_iv",
                    }
                ).assign(option_type=option_type, expiry=pd.Timestamp(expiry))
            )
    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.assign(underlying=ticker, quote_time=quote_time, spot=spot)
    return quotes[CHAIN_COLUMNS]


def fetch_price_history(ticker: str, period: str = "10y") -> pd.Series:
    """Download daily closing prices."""
    close = yf.Ticker(ticker).history(period=period)["Close"]
    close.index = close.index.tz_localize(None).normalize()
    close.name = "close"
    return close


def fetch_treasury_yields(date: str) -> pd.DataFrame:
    """Download the latest Treasury yields, used to sanity-check implied rates."""
    rows = [
        {
            "ticker": ticker,
            "tenor": tenor,
            "yield_pct": round(float(yf.Ticker(ticker).history(period="5d")["Close"].iloc[-1]), 3),
            "date": date,
        }
        for ticker, tenor in TREASURY_TICKERS.items()
    ]
    return pd.DataFrame(rows)


def save_chain(quotes: pd.DataFrame, path: str | Path) -> None:
    """Write a chain snapshot to CSV."""
    quotes.to_csv(path, index=False)


def load_chain(path: str | Path) -> pd.DataFrame:
    """Read back a chain snapshot written by `save_chain`."""
    quotes = pd.read_csv(path)
    quotes["quote_time"] = pd.to_datetime(quotes["quote_time"], utc=True)
    quotes["last_trade"] = pd.to_datetime(quotes["last_trade"], utc=True)
    quotes["expiry"] = pd.to_datetime(quotes["expiry"])
    return quotes[CHAIN_COLUMNS]


def load_bloomberg_csv(
    path: str | Path,
    quote_time: str | pd.Timestamp,
    spot: float = np.nan,
    underlying: str = "",
    expiry: str | None = None,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load a Bloomberg export with one row per contract into the chain format.

    Required: strike, bid, ask, the option type (C/P or call/put) and the expiry
    (a column, or the `expiry` argument for a single-expiry export). Common
    Bloomberg names are recognised through BLOOMBERG_ALIASES. Layouts vary with
    the screen and template, so pass `column_map={"Your Header": "strike"}` for
    anything else. This loader was tested on synthetic files in this layout,
    not on a real terminal export. A side-by-side OMON layout must first be
    reshaped to one row per contract.

    `quote_time` should be when the quotes were taken; a time with no zone is
    read as New York time.
    """
    raw = pd.read_csv(path)
    renames = dict(column_map or {})
    for canonical, aliases in BLOOMBERG_ALIASES.items():
        for column in raw.columns:
            if column.strip().lower() in aliases and canonical not in renames.values():
                renames[column] = canonical
    quotes = raw.rename(columns=renames)

    if expiry is not None:
        quotes["expiry"] = expiry
    missing = {"expiry", "strike", "option_type", "bid", "ask"} - set(quotes.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(raw.columns)}")

    for column in ("strike", "bid", "ask", "last", "volume", "open_interest", "vendor_iv"):
        if column in quotes:
            cleaned = quotes[column].astype(str).str.replace(",", "", regex=False)
            quotes[column] = pd.to_numeric(cleaned, errors="coerce")

    first_letter = quotes["option_type"].astype(str).str.strip().str.upper().str[0]
    quotes["option_type"] = first_letter.map({"C": "call", "P": "put"})
    if quotes["option_type"].isna().any():
        raise ValueError("option_type must contain C/P or call/put")

    timestamp = pd.Timestamp(quote_time)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(MARKET_TIMEZONE)
    quotes["quote_time"] = timestamp.tz_convert("UTC")
    quotes["expiry"] = pd.to_datetime(quotes["expiry"])
    quotes["underlying"] = underlying
    quotes["spot"] = spot
    if "vendor_iv" in quotes:
        quotes["vendor_iv"] = quotes["vendor_iv"] / 100  # Bloomberg quotes vol in points.
    for column in CHAIN_COLUMNS:
        if column not in quotes:
            quotes[column] = pd.NaT if column == "last_trade" else np.nan
    return quotes[CHAIN_COLUMNS]


class MarketDataUnavailable(RuntimeError):
    """No saved snapshot exists and a usable one could not be downloaded."""


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the market analysis needs, taken on one date."""

    date: str
    quotes: pd.DataFrame
    close: pd.Series
    treasuries: pd.DataFrame
    downloaded: bool


def _snapshot_paths(data_dir: Path, date: str) -> dict[str, Path]:
    return {
        "quotes": data_dir / f"spx_options_{date}.csv",
        "close": data_dir / f"spx_daily_close_{date}.csv",
        "treasuries": data_dir / f"treasury_yields_{date}.csv",
    }


def _download_snapshot(data_dir: Path) -> MarketSnapshot:
    """Download a snapshot, check it can produce a smile, then save it locally."""
    date = datetime.now(timezone.utc).date().isoformat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # yfinance emits pandas deprecation notices.
        quotes = fetch_option_chain(SNAPSHOT_TICKER)
        close = fetch_price_history(SNAPSHOT_TICKER)
        treasuries = fetch_treasury_yields(date)
    build_smile(quotes)  # With the market closed, quotes can be empty or all zero.

    paths = _snapshot_paths(data_dir, date)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_chain(quotes, paths["quotes"])
    close.to_csv(paths["close"])
    treasuries.to_csv(paths["treasuries"], index=False)
    return MarketSnapshot(date, load_chain(paths["quotes"]), close, treasuries, downloaded=True)


def obtain_snapshot(data_dir: str | Path, reference_date: str) -> MarketSnapshot:
    """Read the reference snapshot if it is here, otherwise download a new one.

    A downloaded snapshot holds different quotes from the reference, so its
    results will not match the README. Raises MarketDataUnavailable if the
    download fails or returns quotes that cannot produce a smile.
    """
    data_dir = Path(data_dir)
    paths = _snapshot_paths(data_dir, reference_date)
    if all(path.exists() for path in paths.values()):
        close = pd.read_csv(paths["close"], index_col=0, parse_dates=True)["close"]
        treasuries = pd.read_csv(paths["treasuries"])
        return MarketSnapshot(reference_date, load_chain(paths["quotes"]), close, treasuries, downloaded=False)
    try:
        return _download_snapshot(data_dir)
    except Exception as error:  # yfinance raises many types: network, parsing, empty data.
        raise MarketDataUnavailable(
            "Could not download usable quotes. This needs network access and US market "
            f"hours, because Yahoo's bids are empty when the market is closed. "
            f"Cause: {type(error).__name__}: {error}"
        ) from error


def year_fraction(expiry: pd.Timestamp, quote_time: pd.Timestamp) -> float:
    """Calendar time to the close on expiry day, in years (ACT/365).

    Standard monthly SPX options settle at the open, not the close. That
    difference of 6.5 hours is ignored; it only matters for very short maturities.
    """
    close = expiry.tz_localize(MARKET_TIMEZONE) + pd.Timedelta(hours=16)
    return (close - quote_time).total_seconds() / (365.0 * 86_400.0)


@dataclass(frozen=True)
class ForwardFit:
    """Forward and discount factor implied by one expiry's quotes."""

    forward: float
    discount: float
    T: float
    n_pairs: int
    good_strikes: tuple[float, ...]
    bad_strikes: tuple[float, ...]
    noise: float

    @property
    def rate(self) -> float:
        """Continuously compounded rate implied by the discount factor."""
        return -np.log(self.discount) / self.T


def _quote_pairs(quotes: pd.DataFrame) -> pd.DataFrame:
    """Align call and put quotes by strike, keeping strikes quoted on both sides.

    A zero bid means nobody is making a market, so that quote says nothing.
    """
    live = quotes[(quotes["bid"] > 0) & (quotes["ask"] > quotes["bid"])]
    calls = live[live["option_type"] == "call"].groupby("strike")[["bid", "ask"]].mean()
    puts = live[live["option_type"] == "put"].groupby("strike")[["bid", "ask"]].mean()
    strikes = calls.index.intersection(puts.index).sort_values()
    calls, puts = calls.loc[strikes], puts.loc[strikes]
    return pd.DataFrame(
        {
            "gap": 0.5 * (calls["bid"] + calls["ask"]) - 0.5 * (puts["bid"] + puts["ask"]),
            "half_spreads": 0.5 * ((calls["ask"] - calls["bid"]) + (puts["ask"] - puts["bid"])),
        }
    )


def implied_forward(quotes: pd.DataFrame, window: float = 0.10) -> ForwardFit:
    """Fit put-call parity on one expiry to recover the forward and discount factor.

    C - P is a straight line in the strike: its slope is -D and it crosses zero
    at the forward. The line is fitted over strikes within `window` of the
    forward, because a narrow range leaves the slope poorly identified. The
    Theil-Sen estimator (a median of slopes) is used since a few bad quotes
    would drag an ordinary regression. A strike is then flagged if it misses
    the line by more than the fit noise and its bid-ask spreads can explain.
    A quote left over from before a market move fails this test: its call and
    put still reflect the old forward.
    """
    pairs = _quote_pairs(quotes)
    if len(pairs) < 5:
        raise ValueError(f"Need at least 5 strikes with call and put quotes, got {len(pairs)}")

    # Each pair gives its own rough forward; the median ignores the bad ones.
    centre = float((pairs.index + pairs["gap"]).median())
    near = pairs[np.abs(pairs.index / centre - 1.0) <= window]
    slope, intercept, _, _ = theilslopes(near["gap"], near.index)

    residual = pairs["gap"] - (intercept + slope * pairs.index)
    noise = 1.4826 * np.median(np.abs(residual.loc[near.index]))
    bad = residual.abs() > 5.0 * noise + pairs["half_spreads"]

    good = near[~bad.loc[near.index]]
    slope, intercept = np.polyfit(good.index, good["gap"], 1)
    discount = -slope
    expiry, quote_time = quotes["expiry"].iloc[0], quotes["quote_time"].iloc[0]
    return ForwardFit(
        forward=float(intercept / discount),
        discount=float(discount),
        T=year_fraction(expiry, quote_time),
        n_pairs=len(good),
        good_strikes=tuple(float(k) for k in pairs.index[~bad]),
        bad_strikes=tuple(float(k) for k in pairs.index[bad]),
        noise=float(noise),
    )


def parity_misses(quotes: pd.DataFrame, fit: ForwardFit) -> pd.Series:
    """Return how far each strike's C - P sits from the fitted parity line."""
    pairs = _quote_pairs(quotes)
    return pairs["gap"] - fit.discount * (fit.forward - pairs.index)


def build_smile(
    quotes: pd.DataFrame,
    max_rel_spread: float = 0.25,
    moneyness_range: tuple[float, float] = (0.6, 1.3),
) -> pd.DataFrame:
    """Compute the implied vol of out-of-the-money options, for each expiry.

    Passing the forward as the spot with q = r reproduces Black's formula for
    options on a forward: every effect of rates and dividends is already in F.
    `iv_bid` and `iv_ask` show how much the quoted spread blurs each vol.
    """
    tables = []
    for expiry, group in quotes.groupby("expiry"):
        fit = implied_forward(group)
        usable = group[
            (group["bid"] > 0) & (group["ask"] > group["bid"]) & group["strike"].isin(fit.good_strikes)
        ].copy()
        out_of_the_money = np.where(
            usable["option_type"] == "put", usable["strike"] < fit.forward, usable["strike"] >= fit.forward
        )
        usable = usable[out_of_the_money]
        usable["mid"] = 0.5 * (usable["bid"] + usable["ask"])
        usable["moneyness"] = usable["strike"] / fit.forward
        usable = usable[
            ((usable["ask"] - usable["bid"]) / usable["mid"] <= max_rel_spread)
            & usable["moneyness"].between(*moneyness_range)
        ]

        rows = []
        for option_type, side in usable.groupby("option_type"):
            K = side["strike"].to_numpy()
            solve = lambda price: solve_implied_vol(
                price, fit.forward, K, fit.T, fit.rate, option_type, q=fit.rate
            ).vol
            rows.append(
                side.assign(
                    iv=solve(side["mid"].to_numpy()),
                    iv_bid=solve(side["bid"].to_numpy()),
                    iv_ask=solve(side["ask"].to_numpy()),
                )
            )
        table = pd.concat(rows).sort_values("strike")
        tables.append(table.assign(T=fit.T, forward=fit.forward, discount=fit.discount, rate=fit.rate))
    smile = pd.concat(tables, ignore_index=True)
    columns = ["expiry", "T", "forward", "discount", "rate", "option_type", "strike",
               "moneyness", "bid", "ask", "mid", "iv", "iv_bid", "iv_ask", "vendor_iv"]
    return smile.dropna(subset=["iv"])[columns]


def at_the_money_vol(smile: pd.DataFrame) -> pd.Series:
    """Interpolate the implied vol at K = F, for each expiry."""
    return smile.groupby("expiry").apply(
        lambda group: float(np.interp(1.0, group["moneyness"], group["iv"])), include_groups=False
    )


def flat_vol_gap(smile: pd.DataFrame) -> pd.DataFrame:
    """Compare market prices with Black-Scholes at one flat vol per expiry.

    This prices every strike at the at-the-money vol, which is what the
    constant-volatility assumption implies. A `ratio` of 3 means the market
    pays three times the price of that model.
    """
    flat = smile["expiry"].map(at_the_money_vol(smile))
    model = np.empty(len(smile))
    for option_type in ("call", "put"):
        side = (smile["option_type"] == option_type).to_numpy()
        model[side] = bs_price(
            smile["forward"].to_numpy()[side],
            smile["strike"].to_numpy()[side],
            smile["T"].to_numpy()[side],
            smile["rate"].to_numpy()[side],
            flat.to_numpy()[side],
            option_type,
            smile["rate"].to_numpy()[side],
        )
    return smile.assign(
        flat_vol=flat,
        flat_price=model,
        gap=smile["mid"] - model,
        ratio=smile["mid"] / model,
    )


def term_summary(smile: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Summarise each expiry: implied carry, at-the-money vol and skew."""
    rows = []
    atm = at_the_money_vol(smile)
    for expiry, group in smile.groupby("expiry"):
        forward, T = group["forward"].iloc[0], group["T"].iloc[0]
        # NaN beyond the last quoted strike: an edge value would pass for a real reading.
        at = lambda level: float(np.interp(level, group["moneyness"], group["iv"], left=np.nan, right=np.nan))
        rows.append(
            {
                "expiry": expiry.date(),
                "T": T,
                "forward": forward,
                "implied_rate": group["rate"].iloc[0],
                "implied_dividend": group["rate"].iloc[0] - np.log(forward / spot) / T,
                "atm_vol": atm[expiry],
                "vol_90": at(0.90),
                "vol_110": at(1.10),
                "skew_90_100": at(0.90) - atm[expiry],
                "n_options": len(group),
            }
        )
    return pd.DataFrame(rows)


def return_statistics(close: pd.Series, window: int = 21) -> dict[str, float]:
    """Measure how far daily returns are from the lognormal assumption."""
    returns = np.log(close).diff().dropna()
    daily_vol = returns.std()
    rolling = returns.rolling(window).std().dropna() * np.sqrt(252)
    threshold = 4.0
    return {
        "annual_vol": float(daily_vol * np.sqrt(252)),
        "excess_kurtosis": float(returns.kurt()),
        "worst_day": float(returns.min()),
        "worst_day_in_sd": float(returns.min() / daily_vol),
        "days_beyond_4sd": int((returns.abs() > threshold * daily_vol).sum()),
        "expected_days_beyond_4sd": float(2.0 * norm.sf(threshold) * len(returns)),
        "rolling_vol_min": float(rolling.min()),
        "rolling_vol_median": float(rolling.median()),
        "rolling_vol_max": float(rolling.max()),
        "n_days": len(returns),
    }
