"""
India segment registry: who belongs to which NSE index bucket.

Two India-only sources, joined into one symbol table:

  1. NSE index archives (public CSVs, no auth) give OFFICIAL constituent lists
     with company name, industry and ISIN. NSE's own structure makes the
     segment split exact rather than guessed:

         Nifty 50 + Nifty Next 50              == Nifty 100    (largecap)
         Nifty 100 + Midcap 150 + Smallcap 250 == Nifty 500    (disjoint)
         Nifty 500 + Microcap 250              == Total Market (disjoint)

     All three identities are asserted at load time, so a silent NSE reshuffle
     surfaces as a warning instead of a wrong "Midcap" label. Microcap 250 is
     what carries the genuinely explosive daily moves -- a 20% day usually
     happens below the Nifty 500 line, so the scan universe is Total Market.

  2. Groww's instrument master (public CSV, no auth) is the tradeability
     authority: it confirms the symbol exists on NSE CASH series EQ, carries
     the lot size, and its FNO rows tell us which underlyings have futures and
     options. Groww's paid live-data and historical endpoints are NOT used --
     the configured key has no entitlement for them (they answer 403).

Prices are NOT sourced here. India quotes still come through data_source
(Yahoo), which is also the only source for US / crypto / commodities / global
indices. This module is strictly the India membership + metadata layer.

Everything caches to disk so a blocked upstream degrades to the last good
snapshot rather than an empty screen.
"""
import csv
import io
import json
import os
import threading
import time
from datetime import datetime, timezone

import data_source

# ------------------------------------------------------------------ config ---

_NSE_INDEX_URL = "https://nsearchives.nseindia.com/content/indices/{f}.csv"
_NSE_HEADERS = {
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    # nsearchives serves the archive files but wants to look like a site visit.
    "Referer": "https://www.nseindia.com/",
}

_GROWW_INSTRUMENTS_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"

# id -> (NSE archive filename, human label)
INDEX_FILES = {
    "nifty50": ("ind_nifty50list", "NIFTY 50"),
    "next50": ("ind_niftynext50list", "NIFTY NEXT 50"),
    "midcap150": ("ind_niftymidcap150list", "NIFTY MIDCAP 150"),
    "smallcap250": ("ind_niftysmallcap250list", "NIFTY SMALLCAP 250"),
    "microcap250": ("ind_niftymicrocap250_list", "NIFTY MICROCAP 250"),
}

# Which index each cap segment is built from. Nifty 100 and Nifty 500 are
# deliberately DERIVED from their parts rather than fetched, so a bucket can
# never drift out of sync with the indices that compose it.
_SEGMENT_OF_INDEX = {
    "nifty50": "largecap",
    "next50": "largecap",
    "midcap150": "midcap",
    "smallcap250": "smallcap",
    "microcap250": "microcap",
}

SEGMENT_LABELS = {
    "largecap": "Largecap",
    "midcap": "Midcap",
    "smallcap": "Smallcap",
    "microcap": "Microcap",
    # Everything listed and tradeable on NSE that sits BELOW the Total Market
    # 750 line. NSE publishes no size index down here, so we do not invent a
    # cap label for it -- these names are simply unranked by the exchange.
    "unranked": "Unranked",
}

# Cap segments in size order, for stable ordering in UI pickers.
SEGMENT_ORDER = ["largecap", "midcap", "smallcap", "microcap", "unranked"]

# ISINs beginning INE identify operating companies. INF is the mutual-fund /
# ETF range: NSE lists 339 of those under CASH series EQ (BANKBEES, KOTAKNV20,
# gold and liquid funds). They are not companies and their "daily move" is
# frequently a NAV artefact -- KOTAKNV20 printed +84,260% on 2026-08-12 -- so
# they are excluded from the equity universe entirely.
_COMPANY_ISIN_PREFIX = "INE"

# Snapshot of the joined table. Survives restarts AND a 7-day upstream outage
# (data_source's stale window), so the screen keeps working offline.
_SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), ".nse_segments.json")
_INSTRUMENTS_FILE = os.path.join(os.path.dirname(__file__), ".groww_instruments.json")

_INDEX_TTL = 24 * 3600       # NSE rebalances semi-annually; a day is generous
_INSTRUMENTS_TTL = 24 * 3600  # Groww regenerates the master daily

_lock = threading.Lock()
_table: dict | None = None


class SegmentsUnavailableError(RuntimeError):
    """No membership data at all: upstream blocked and no snapshot on disk."""


# --------------------------------------------------------- NSE index lists ---

def _fetch_index(index_id: str) -> list[dict] | None:
    """One official constituent list. Cached; None when never fetched and
    upstream is unreachable."""
    filename, _label = INDEX_FILES[index_id]
    key = f"nse:index:{index_id}"
    cached, fresh = data_source._cache_get(key, _INDEX_TTL)
    if cached is not None and fresh:
        return cached

    try:
        resp = data_source.http_get(
            _NSE_INDEX_URL.format(f=filename), timeout=25, headers=_NSE_HEADERS)
        resp.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(resp.text)))
        parsed = [
            {
                "symbol": (r.get("Symbol") or "").strip(),
                "name": (r.get("Company Name") or "").strip().rstrip("."),
                "industry": (r.get("Industry") or "").strip(),
                "isin": (r.get("ISIN Code") or "").strip(),
                "series": (r.get("Series") or "").strip(),
            }
            for r in rows
        ]
        # NSE keeps DUMMY* placeholder tickers in the constituent files to hold
        # a slot through a corporate action. They are not tradeable and have no
        # price feed, so they would only ever show up as permanent blanks.
        parsed = [p for p in parsed
                  if p["symbol"] and not p["symbol"].startswith("DUMMY")]
        if not parsed:
            raise ValueError(f"{filename}: no rows parsed")
        data_source._cache_put(key, parsed)
        return parsed
    except Exception as e:
        print(f"[nse_segments] index '{index_id}' fetch failed: {e}")
        return cached  # stale-if-error


# ---------------------------------------------------- Groww instrument master ---

def _fetch_instruments() -> dict | None:
    """Compact view of Groww's 20MB master: NSE CASH series-EQ equities plus
    the set of NSE FNO underlyings.

    The raw CSV is ~139k rows, so the parsed result is cached to its own JSON
    file rather than re-parsed per process.
    """
    try:
        st = os.stat(_INSTRUMENTS_FILE)
        if time.time() - st.st_mtime <= _INSTRUMENTS_TTL:
            with open(_INSTRUMENTS_FILE) as f:
                return json.load(f)
    except Exception:
        pass

    try:
        # No Authorization header: this asset is public, and sending the paid
        # bearer token to the asset host gets a 401.
        resp = data_source.http_get(
            _GROWW_INSTRUMENTS_URL, timeout=90, headers={"Accept": "text/csv,*/*"})
        resp.raise_for_status()

        equities: dict = {}
        fno_underlyings: set = set()
        for r in csv.DictReader(io.StringIO(resp.text)):
            if r.get("exchange") != "NSE":
                continue
            seg = r.get("segment")
            if seg == "CASH" and r.get("series") == "EQ":
                sym = (r.get("trading_symbol") or "").strip()
                if sym:
                    equities[sym] = {
                        "name": (r.get("name") or "").strip(),
                        "isin": (r.get("isin") or "").strip(),
                        "lot_size": _as_int(r.get("lot_size")),
                        "is_intraday": r.get("is_intraday") == "1",
                    }
            elif seg == "FNO":
                und = (r.get("underlying_symbol") or "").strip()
                if und:
                    fno_underlyings.add(und)

        if not equities:
            raise ValueError("instrument master parsed 0 NSE CASH EQ rows")

        out = {"equities": equities, "fno": sorted(fno_underlyings)}
        try:
            with open(_INSTRUMENTS_FILE, "w") as f:
                json.dump(out, f)
        except Exception:
            pass
        return out
    except Exception as e:
        print(f"[nse_segments] Groww instrument master fetch failed: {e}")
        try:  # any age beats nothing
            with open(_INSTRUMENTS_FILE) as f:
                return json.load(f)
        except Exception:
            return None


def _as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------- table ---

def _build() -> dict | None:
    """Join the NSE index lists with the Groww master into one symbol table."""
    members: dict = {}
    index_sets: dict = {}
    missing_indices = []

    for index_id in INDEX_FILES:
        rows = _fetch_index(index_id)
        if not rows:
            missing_indices.append(index_id)
            continue
        index_sets[index_id] = {r["symbol"] for r in rows}
        for r in rows:
            entry = members.setdefault(r["symbol"], {
                "symbol": r["symbol"],
                "name": r["name"],
                "industry": r["industry"] or "Unclassified",
                "isin": r["isin"],
                "indices": [],
                "segment": None,
            })
            if index_id not in entry["indices"]:
                entry["indices"].append(index_id)
            # Cap segment comes from the size index the symbol sits in.
            entry["segment"] = _SEGMENT_OF_INDEX[index_id]

    if not members:
        return None

    # Structural assertions. NSE guarantees these; if a rebalance breaks one we
    # want a loud warning in the payload, not a quietly mislabelled segment.
    warnings = []
    if missing_indices:
        warnings.append(
            f"index lists unavailable: {', '.join(missing_indices)}")

    n50 = index_sets.get("nifty50", set())
    nn50 = index_sets.get("next50", set())
    mid = index_sets.get("midcap150", set())
    small = index_sets.get("smallcap250", set())
    micro = index_sets.get("microcap250", set())
    n100 = n50 | nn50
    n500 = n100 | mid | small

    for a_name, a, b_name, b in (
        ("nifty50", n50, "next50", nn50),
        ("nifty100", n100, "midcap150", mid),
        ("nifty100", n100, "smallcap250", small),
        ("midcap150", mid, "smallcap250", small),
        ("nifty500", n500, "microcap250", micro),
    ):
        overlap = a & b
        if overlap:
            warnings.append(
                f"{a_name} and {b_name} overlap on {len(overlap)} symbols "
                f"(e.g. {', '.join(sorted(overlap)[:3])})")

    # Nifty 100 and Nifty 500 are derived from their parts, never fetched.
    for sym in n100:
        if sym in members and "nifty100" not in members[sym]["indices"]:
            members[sym]["indices"].append("nifty100")
    for sym in n500:
        if sym in members and "nifty500" not in members[sym]["indices"]:
            members[sym]["indices"].append("nifty500")

    # Tradeability + F&O from Groww.
    instruments = _fetch_instruments()
    if instruments:
        equities = instruments["equities"]
        fno = set(instruments["fno"])

        # ------------------------------------------------------------------
        # Widen the universe past the index lists.
        #
        # The five NSE size indices only cover 752 names. NSE lists ~2,120
        # operating companies in the cash EQ series, and the ones that actually
        # hit a 20% circuit are overwhelmingly the ~1,370 BELOW the Total
        # Market 750 line -- on 2026-08-12 both +20% names (ABMINTLLTD,
        # SINGERIND) sat outside every index, so an index-only scan could not
        # see them however hard it looked.
        #
        # These names are added with segment "unranked" and an empty `indices`
        # list: they are genuinely tradeable NSE equities, but the exchange
        # publishes no size ranking for them and we do not fabricate one.
        # ------------------------------------------------------------------
        added_unranked = 0
        for sym, eq in equities.items():
            if sym in members:
                continue
            if not (eq.get("isin") or "").startswith(_COMPANY_ISIN_PREFIX):
                continue  # ETF / mutual fund, not a company
            members[sym] = {
                "symbol": sym,
                "name": eq.get("name") or sym,
                # Groww's master carries no sector, and guessing one from the
                # ticker would be worse than admitting we do not have it.
                "industry": "Unclassified",
                "isin": eq.get("isin", ""),
                "indices": [],
                "segment": "unranked",
            }
            added_unranked += 1
        if added_unranked:
            warnings.append(
                f"{added_unranked} NSE cash-EQ companies sit outside every NSE "
                f"size index; they are scanned and labelled 'Unranked'")

        unlisted = []
        for sym, entry in members.items():
            eq = equities.get(sym)
            entry["tradeable"] = eq is not None
            entry["fno"] = sym in fno
            entry["lot_size"] = (eq or {}).get("lot_size")
            entry["intraday"] = (eq or {}).get("is_intraday", False)
            # Groww's company name is shorter and cleaner for dense tables;
            # NSE's is the legal name. Prefer Groww when present.
            if eq and eq.get("name"):
                entry["name"] = eq["name"]
            if eq is None:
                unlisted.append(sym)
        if unlisted:
            warnings.append(
                f"{len(unlisted)} index members absent from the Groww NSE "
                f"cash master (e.g. {', '.join(sorted(unlisted)[:3])})")
    else:
        warnings.append("Groww instrument master unavailable: no F&O or lot-size data")
        for entry in members.values():
            entry.setdefault("tradeable", None)
            entry.setdefault("fno", None)
            entry.setdefault("lot_size", None)
            entry.setdefault("intraday", None)

    # Yahoo is still the price source for India, so carry its symbol form.
    for entry in members.values():
        entry["yahoo"] = f"{entry['symbol']}.NS"

    table = {
        "symbols": members,
        "counts": {
            seg: sum(1 for e in members.values() if e["segment"] == seg)
            for seg in SEGMENT_ORDER
        },
        "index_counts": {k: len(v) for k, v in index_sets.items()},
        "warnings": warnings,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    try:
        with open(_SNAPSHOT_FILE, "w") as f:
            json.dump(table, f)
    except Exception:
        pass
    return table


def get_table(force: bool = False) -> dict:
    """The joined symbol table. Falls back to the on-disk snapshot, and only
    raises when there is genuinely nothing to serve."""
    global _table
    with _lock:
        if _table is not None and not force:
            return _table
        built = _build()
        if built is None:
            try:
                with open(_SNAPSHOT_FILE) as f:
                    built = json.load(f)
                    built.setdefault("warnings", []).append(
                        "serving an on-disk snapshot: NSE index lists unreachable")
            except Exception:
                raise SegmentsUnavailableError(
                    "NSE constituent lists are unreachable and no snapshot is "
                    "cached yet.")
        _table = built
        return _table


# ------------------------------------------------------------- public views ---

# Selectable universes on the movers screen. Each resolves to a symbol subset
# of the single 500-name scan, so switching tabs costs no extra upstream calls.
SCOPES = {
    # The widest scope: every NSE cash-EQ operating company. This is the only
    # scope that can see a 20% circuit, because those names are almost never
    # index constituents.
    "allnse": {"label": "All NSE equity", "kind": "all", "group": "index"},
    # "Total Market" stays the exchange's own 750-name index, NOT "everything
    # we scan" -- relabelling it would misreport what the number means.
    "totalmarket": {"label": "NIFTY Total Market", "kind": "ranked",
                    "group": "index"},
    "nifty50": {"label": "NIFTY 50", "kind": "index", "group": "index"},
    "next50": {"label": "NIFTY NEXT 50", "kind": "index", "group": "index"},
    "nifty100": {"label": "NIFTY 100", "kind": "index", "group": "index"},
    "nifty500": {"label": "NIFTY 500", "kind": "index", "group": "index"},
    "midcap150": {"label": "NIFTY MIDCAP 150", "kind": "index", "group": "index"},
    "smallcap250": {"label": "NIFTY SMALLCAP 250", "kind": "index", "group": "index"},
    "microcap250": {"label": "NIFTY MICROCAP 250", "kind": "index", "group": "index"},
    "largecap": {"label": "Largecap", "kind": "segment", "group": "segment"},
    "midcap": {"label": "Midcap", "kind": "segment", "group": "segment"},
    "smallcap": {"label": "Smallcap", "kind": "segment", "group": "segment"},
    "microcap": {"label": "Microcap", "kind": "segment", "group": "segment"},
    "unranked": {"label": "Unranked", "kind": "segment", "group": "segment"},
    "fno": {"label": "F&O universe", "kind": "flag", "group": "segment"},
    "intraday": {"label": "Intraday (MIS)", "kind": "flag", "group": "segment"},
}


class UnknownScopeError(ValueError):
    """Scope id not in SCOPES — the route maps this to HTTP 400."""


def resolve_scope(scope_id: str) -> list[dict]:
    """Symbol entries belonging to a scope."""
    spec = SCOPES.get(scope_id)
    if spec is None:
        raise UnknownScopeError(
            f"Unknown scope '{scope_id}'. Valid: {', '.join(SCOPES)}")

    entries = list(get_table()["symbols"].values())
    if spec["kind"] == "all":
        return entries
    if spec["kind"] == "ranked":
        # Members of at least one NSE size index -- the official Total Market.
        return [e for e in entries if e["segment"] != "unranked"]
    if spec["kind"] == "segment":
        return [e for e in entries if e["segment"] == scope_id]
    if spec["kind"] == "flag":
        if scope_id == "intraday":
            return [e for e in entries if e.get("intraday")]
        return [e for e in entries if e.get("fno")]
    return [e for e in entries if scope_id in e["indices"]]


def all_symbols() -> list[str]:
    """Every Yahoo symbol in the scan universe -- all NSE cash-EQ operating
    companies, index-ranked or not -- sorted so the spark chunk boundaries, and
    therefore the cache keys, stay stable.

    Sorting matters: chunk membership must not change between calls, or every
    reshuffle would miss the cache and re-hit the upstream for data we hold.
    """
    return sorted(e["yahoo"] for e in get_table()["symbols"].values())


def scope_index() -> list[dict]:
    """Scope picker payload: id, label, group and how many names each covers."""
    out = []
    for scope_id, spec in SCOPES.items():
        try:
            count = len(resolve_scope(scope_id))
        except SegmentsUnavailableError:
            count = 0
        out.append({
            "id": scope_id, "label": spec["label"], "kind": spec["kind"],
            "group": spec["group"], "count": count,
        })
    return out
