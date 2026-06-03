#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ON-CHAIN PATTERN RETRIEVAL SCANNER  v1.0.0                         ║
║          Author  : Senior Web3 Data Engineer                                ║
║          License : MIT (Open Source)                                        ║
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Scan any NFT smart contract address and surface recurring behavioural     ║
║    patterns — bots, wash-traders, whale concentration, and loyal holders.   ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN
──────────
1.  Install dependencies (Python ≥ 3.9 recommended):

        pip install requests pandas python-dotenv tabulate colorama

2.  Add your free Etherscan API key.  Two options:

    Option A  — .env file (recommended):
        Create a file named  .env  in the same directory as this script:
            ETHERSCAN_API_KEY=YOUR_KEY_HERE

    Option B  — environment variable:
        export ETHERSCAN_API_KEY="YOUR_KEY_HERE"

    Get a free key at: https://etherscan.io/myapikey

3.  Run:

        python onchain_scanner.py --contract 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D

    Optional flags:
        --max-txns  500      Cap the number of transactions fetched  (default: 500)
        --output    report   Also write a Markdown report to  report.md
        --no-color           Disable terminal colour (useful for piping)

ARCHITECTURE
────────────
    EtherscanClient   →  raw HTTP calls to Etherscan token-transfer endpoint
    TransactionParser →  clean raw JSON → typed pandas DataFrame
    PatternEngine     →  pure-analytics layer (retention, flip, concentration)
    Reporter          →  renders ASCII tables + optional Markdown file
    CLI               →  argparse entry point
"""

import os
import sys
import time
import argparse
import textwrap
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate
from colorama import init as colorama_init, Fore, Style

# ── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()                       # reads .env if present
colorama_init(autoreset=True)       # Windows-safe colour codes


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  ETHERSCAN CLIENT                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class EtherscanClient:
    """
    Thin wrapper around the Etherscan v2 public API.

    Only the  tokennfttx  (ERC-721 transfer) endpoint is used here, but the
    class is easy to extend with  txlist,  tokentx, etc.

    Rate limit: free tier allows 5 calls/second → we use a tiny sleep between
    paginated requests to stay well within quota.
    """

    BASE_URL = "https://api.etherscan.io/api"
    MAX_RESULTS_PER_PAGE = 10_000   # Etherscan hard cap per request

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "ETHERSCAN_API_KEY is not set.\n"
                "Run:  export ETHERSCAN_API_KEY='your_key'  or add it to .env"
            )
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OnChainPatternScanner/1.0"})

    # ── public helpers ────────────────────────────────────────────────────

    def fetch_nft_transfers(
        self,
        contract_address: str,
        max_records: int = 500,
    ) -> list[dict]:
        """
        Retrieve ERC-721 token transfers for *contract_address*.

        Automatically paginates until  max_records  is reached or the
        API returns an empty page.

        Returns a list of raw Etherscan result dicts.
        """
        all_results: list[dict] = []
        page = 1

        print(f"\n{Fore.CYAN}⬡  Fetching NFT transfer history…{Style.RESET_ALL}")

        while len(all_results) < max_records:
            batch_size = min(self.MAX_RESULTS_PER_PAGE, max_records - len(all_results))

            params = {
                "module":          "account",
                "action":          "tokennfttx",
                "contractaddress": contract_address,
                "page":            page,
                "offset":          batch_size,
                "sort":            "asc",
                "apikey":          self.api_key,
            }

            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                print(f"{Fore.RED}✗  Network error: {exc}{Style.RESET_ALL}")
                break

            # Etherscan signals errors via status == "0"
            if data.get("status") == "0":
                msg = data.get("message", "Unknown error")
                result = data.get("result", "")
                # "No transactions found" is not a real error — it just means empty
                if "No transactions found" in str(result):
                    break
                print(f"{Fore.YELLOW}⚠  Etherscan: {msg} — {result}{Style.RESET_ALL}")
                break

            batch: list[dict] = data.get("result", [])
            if not batch:
                break   # exhausted all pages

            all_results.extend(batch)
            print(
                f"   {Fore.GREEN}✓{Style.RESET_ALL}  Page {page:>3}  "
                f"│  +{len(batch):>5} txns  │  total: {len(all_results)}"
            )

            if len(batch) < batch_size:
                break   # last page was partial — no more data

            page += 1
            time.sleep(0.25)   # polite rate-limiting

        return all_results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  TRANSACTION PARSER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TransactionParser:
    """
    Converts raw Etherscan JSON records into a clean, typed pandas DataFrame.

    Columns produced
    ────────────────
    timestamp       datetime (UTC)
    block_number    int
    tx_hash         str
    from_address    str  (lower-cased)
    to_address      str  (lower-cased)
    token_id        str
    token_name      str
    event_type      str  ('Mint' | 'Transfer' | 'Burn')
    """

    # The zero address conventionally marks mint / burn events
    ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

    @staticmethod
    def parse(raw_records: list[dict]) -> pd.DataFrame:
        """
        Accepts the raw list returned by  EtherscanClient.fetch_nft_transfers
        and returns a clean DataFrame sorted by timestamp ascending.
        """
        if not raw_records:
            return pd.DataFrame()

        rows = []
        for rec in raw_records:
            from_addr = rec.get("from", "").lower()
            to_addr   = rec.get("to",   "").lower()

            # ── classify the event ────────────────────────────────────────
            if from_addr == TransactionParser.ZERO_ADDRESS:
                event_type = "Mint"
            elif to_addr == TransactionParser.ZERO_ADDRESS:
                event_type = "Burn"
            else:
                event_type = "Transfer"

            rows.append({
                "timestamp":    datetime.fromtimestamp(
                                    int(rec.get("timeStamp", 0)),
                                    tz=timezone.utc,
                                ),
                "block_number": int(rec.get("blockNumber", 0)),
                "tx_hash":      rec.get("hash", ""),
                "from_address": from_addr,
                "to_address":   to_addr,
                "token_id":     rec.get("tokenID", ""),
                "token_name":   rec.get("tokenName", ""),
                "event_type":   event_type,
            })

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  PATTERN ENGINE                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PatternEngine:
    """
    Core analytics layer.

    All three metrics are pure functions that accept a DataFrame and return
    structured result dictionaries — they have no side-effects and are easy
    to unit-test independently.

    Metrics
    ───────
    A)  Wallet Retention Rate   — how long wallets hold tokens on average
    B)  Flip Velocity           — high-frequency buy→sell patterns
    C)  Concentration Metrics   — whale dominance analysis
    """

    # ── A)  WALLET RETENTION RATE ─────────────────────────────────────────

    @staticmethod
    def wallet_retention(df: pd.DataFrame) -> dict:
        """
        For each (wallet, token_id) pair, compute the hold duration:

            hold_duration = timestamp_of_transfer_OUT − timestamp_of_transfer_IN

        Wallets that never transferred out are considered current holders.

        Returns
        ───────
        {
          "per_token_holds": DataFrame  — one row per (wallet, token_id) hold event
          "avg_hold_days":   float      — fleet-wide average hold in days
          "median_hold_days":float
          "current_holders": set        — wallets that currently hold ≥1 token
          "retention_bands": dict       — distribution bucketed by hold length
        }
        """
        # Build a directed per-token timeline:
        # "IN"  events  = to_address receives the token
        # "OUT" events  = from_address sends the token
        transfers = df[df["event_type"] == "Transfer"].copy()
        mints     = df[df["event_type"] == "Mint"].copy()

        # Arrivals: a wallet receives a token
        arrivals = pd.concat([
            mints[["timestamp",  "to_address",   "token_id"]].rename(
                columns={"to_address": "wallet"}),
            transfers[["timestamp", "to_address", "token_id"]].rename(
                columns={"to_address": "wallet"}),
        ]).sort_values("timestamp").reset_index(drop=True)

        # Departures: a wallet sends a token
        departures = transfers[["timestamp", "from_address", "token_id"]].rename(
            columns={"from_address": "wallet", "timestamp": "depart_ts"}
        ).sort_values("depart_ts").reset_index(drop=True)

        # Merge on wallet + token_id; find the first departure that comes
        # AFTER each arrival (approximate — good enough for pattern detection).
        hold_records = []
        for _, row in arrivals.iterrows():
            wallet   = row["wallet"]
            token_id = row["token_id"]
            arrive   = row["timestamp"]

            # find earliest departure of this specific (wallet, token) after arrival
            matching = departures[
                (departures["wallet"]   == wallet) &
                (departures["token_id"] == token_id) &
                (departures["depart_ts"] > arrive)
            ]

            if matching.empty:
                # wallet still holds → hold_days = None (current holder)
                hold_records.append({
                    "wallet":     wallet,
                    "token_id":   token_id,
                    "arrive_ts":  arrive,
                    "depart_ts":  None,
                    "hold_days":  None,
                    "still_holding": True,
                })
            else:
                depart = matching.iloc[0]["depart_ts"]
                hold_days = (depart - arrive).total_seconds() / 86_400
                hold_records.append({
                    "wallet":     wallet,
                    "token_id":   token_id,
                    "arrive_ts":  arrive,
                    "depart_ts":  depart,
                    "hold_days":  hold_days,
                    "still_holding": False,
                })

        holds_df = pd.DataFrame(hold_records)

        # wallets still holding at least one token
        current_holders = set(
            holds_df[holds_df["still_holding"]]["wallet"].unique()
        )

        # stats on CLOSED holds only (where hold_days is not None)
        closed = holds_df["hold_days"].dropna()
        avg_hold    = float(closed.mean())    if not closed.empty else 0.0
        median_hold = float(closed.median())  if not closed.empty else 0.0

        # bucket holds into human-readable bands
        def band(days):
            if days is None:       return "Current Holder"
            if days < 1:           return "< 1 day   (Flipper)"
            if days < 7:           return "1–7 days"
            if days < 30:          return "7–30 days"
            if days < 90:          return "1–3 months"
            if days < 365:         return "3–12 months"
            return                        "> 1 year  (Diamond Hands)"

        holds_df["band"] = holds_df["hold_days"].apply(band)
        retention_bands  = holds_df["band"].value_counts().to_dict()

        return {
            "per_token_holds":  holds_df,
            "avg_hold_days":    round(avg_hold, 2),
            "median_hold_days": round(median_hold, 2),
            "current_holders":  current_holders,
            "retention_bands":  retention_bands,
        }

    # ── B)  FLIP VELOCITY ─────────────────────────────────────────────────

    @staticmethod
    def flip_velocity(df: pd.DataFrame, hold_threshold_hours: float = 48.0) -> dict:
        """
        A "flip" is defined as: a wallet receives a token and then sends it
        away within  hold_threshold_hours  (default 48 h).

        High flip velocity is a bot / wash-trading signal.

        Returns
        ───────
        {
          "flippers":        DataFrame  — wallets with flip_count ≥ 1, sorted desc
          "flip_rate":       float      — % of all transfers that are flips
          "bot_candidates":  list[str]  — wallets with ≥5 flips (strong bot signal)
          "flip_threshold_hours": float
        }
        """
        transfers = df[df["event_type"].isin(["Transfer", "Mint"])].copy()

        # reuse the hold dataframe logic but filter by threshold
        arrivals = pd.concat([
            df[df["event_type"] == "Mint"][["timestamp", "to_address", "token_id"]].rename(
                columns={"to_address": "wallet"}),
            df[df["event_type"] == "Transfer"][["timestamp", "to_address", "token_id"]].rename(
                columns={"to_address": "wallet"}),
        ])

        departures = df[df["event_type"] == "Transfer"][
            ["timestamp", "from_address", "token_id"]
        ].rename(columns={"from_address": "wallet", "timestamp": "depart_ts"})

        flip_records = []
        for _, row in arrivals.iterrows():
            wallet, token_id, arrive = row["wallet"], row["token_id"], row["timestamp"]
            matching = departures[
                (departures["wallet"]   == wallet) &
                (departures["token_id"] == token_id) &
                (departures["depart_ts"] > arrive)
            ]
            if not matching.empty:
                depart    = matching.iloc[0]["depart_ts"]
                hold_hrs  = (depart - arrive).total_seconds() / 3_600
                if hold_hrs <= hold_threshold_hours:
                    flip_records.append({
                        "wallet":    wallet,
                        "token_id":  token_id,
                        "hold_hrs":  round(hold_hrs, 2),
                    })

        flips_df = pd.DataFrame(flip_records)

        if flips_df.empty:
            return {
                "flippers":             pd.DataFrame(),
                "flip_rate":            0.0,
                "bot_candidates":       [],
                "flip_threshold_hours": hold_threshold_hours,
            }

        # aggregate per wallet
        flipper_summary = (
            flips_df.groupby("wallet")
            .agg(
                flip_count=("token_id", "count"),
                avg_hold_hrs=("hold_hrs", "mean"),
                min_hold_hrs=("hold_hrs", "min"),
            )
            .reset_index()
            .sort_values("flip_count", ascending=False)
        )
        flipper_summary["avg_hold_hrs"] = flipper_summary["avg_hold_hrs"].round(2)
        flipper_summary["min_hold_hrs"] = flipper_summary["min_hold_hrs"].round(2)

        total_transfers = len(df[df["event_type"] == "Transfer"])
        flip_rate = round(len(flips_df) / max(total_transfers, 1) * 100, 2)

        # wallets with ≥5 flips within the threshold → strong bot candidates
        bot_candidates = flipper_summary[
            flipper_summary["flip_count"] >= 5
        ]["wallet"].tolist()

        return {
            "flippers":             flipper_summary,
            "flip_rate":            flip_rate,
            "bot_candidates":       bot_candidates,
            "flip_threshold_hours": hold_threshold_hours,
        }

    # ── C)  CONCENTRATION METRICS ─────────────────────────────────────────

    @staticmethod
    def concentration_metrics(df: pd.DataFrame) -> dict:
        """
        Determines how many unique wallets hold tokens and whether a small
        group ("whales") disproportionately dominate activity.

        Signals computed
        ────────────────
        • Current token distribution (from arrival/departure analysis)
        • Herfindahl-Hirschman Index (HHI) — industry-standard market
          concentration metric.  HHI > 2500 = highly concentrated.
        • Top-10 wallets' share of total supply
        • Transaction activity concentration (not just holdings)

        Returns a dict with summary stats and a per-wallet DataFrame.
        """
        # ── current holdings per wallet ───────────────────────────────────
        # We simulate this with a running balance.
        balance: dict[str, dict[str, int]] = {}   # wallet → {token_id: count}

        for _, row in df.iterrows():
            frm = row["from_address"]
            to  = row["to_address"]
            tid = row["token_id"]

            # credit the receiver
            if to not in balance:
                balance[to] = {}
            balance[to][tid] = balance[to].get(tid, 0) + 1

            # debit the sender (skip zero address = mint source)
            zero = TransactionParser.ZERO_ADDRESS
            if frm != zero:
                if frm not in balance:
                    balance[frm] = {}
                balance[frm][tid] = max(balance[frm].get(tid, 0) - 1, 0)

        # flatten to (wallet, held_count)
        holdings = {
            w: sum(v for v in tokens.values() if v > 0)
            for w, tokens in balance.items()
        }
        # remove zero-balance wallets and the zero address
        zero = TransactionParser.ZERO_ADDRESS
        holdings = {
            w: c for w, c in holdings.items()
            if c > 0 and w != zero
        }

        if not holdings:
            return {"error": "Could not compute holdings from available data."}

        total_supply = sum(holdings.values())
        holdings_df  = (
            pd.DataFrame(
                list(holdings.items()), columns=["wallet", "held_count"]
            )
            .sort_values("held_count", ascending=False)
            .reset_index(drop=True)
        )
        holdings_df["pct_supply"] = (
            holdings_df["held_count"] / total_supply * 100
        ).round(2)

        # Herfindahl-Hirschman Index
        # HHI = Σ (market_share_i)²  where share is expressed as 0-100
        hhi = float(((holdings_df["pct_supply"] ** 2)).sum())

        # Top-10 wallets share
        top10_share = holdings_df.head(10)["pct_supply"].sum()

        # Transaction activity distribution
        tx_counts = (
            df.groupby("from_address")
            .size()
            .reset_index(name="tx_count")
            .sort_values("tx_count", ascending=False)
        )
        tx_counts = tx_counts[tx_counts["from_address"] != zero]

        return {
            "holdings_df":    holdings_df,
            "total_supply":   total_supply,
            "unique_holders": len(holdings_df),
            "hhi":            round(hhi, 2),
            "hhi_label":      PatternEngine._hhi_label(hhi),
            "top10_pct":      round(top10_share, 2),
            "tx_activity":    tx_counts,
        }

    @staticmethod
    def _hhi_label(hhi: float) -> str:
        """Human-readable interpretation of HHI."""
        if hhi < 1500:   return "Competitive / Decentralised"
        if hhi < 2500:   return "Moderately Concentrated"
        return                  "Highly Concentrated (Whale Alert 🐋)"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  REPORTER                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Reporter:
    """
    Renders analysis results to the terminal as formatted ASCII tables
    and optionally writes a Markdown report to disk.
    """

    SEPARATOR = "─" * 78

    # ── Console ───────────────────────────────────────────────────────────

    @staticmethod
    def _header(title: str) -> str:
        top  = "┌" + "─" * 76 + "┐"
        mid  = f"│  {title:<74}│"
        bot  = "└" + "─" * 76 + "┘"
        return f"\n{Fore.CYAN}{top}\n{mid}\n{bot}{Style.RESET_ALL}"

    @staticmethod
    def print_overview(contract: str, df: pd.DataFrame) -> None:
        print(Reporter._header("📋  COLLECTION OVERVIEW"))
        total     = len(df)
        mints     = (df["event_type"] == "Mint").sum()
        transfers = (df["event_type"] == "Transfer").sum()
        burns     = (df["event_type"] == "Burn").sum()
        name      = df["token_name"].iloc[0] if not df.empty else "N/A"
        span_days = (
            (df["timestamp"].max() - df["timestamp"].min()).days
            if len(df) > 1 else 0
        )

        rows = [
            ["Contract Address", contract],
            ["Token Name",       name],
            ["Total Records",    total],
            ["Mints",            mints],
            ["Transfers",        transfers],
            ["Burns",            burns],
            ["Data Span (days)", span_days],
            ["Earliest Event",   str(df["timestamp"].min())[:19] if not df.empty else "N/A"],
            ["Latest Event",     str(df["timestamp"].max())[:19] if not df.empty else "N/A"],
        ]
        print(tabulate(rows, tablefmt="fancy_grid"))

    @staticmethod
    def print_retention(result: dict) -> None:
        print(Reporter._header("⏳  WALLET RETENTION RATE"))

        summary = [
            ["Avg Hold Duration (days)",    result["avg_hold_days"]],
            ["Median Hold Duration (days)", result["median_hold_days"]],
            ["Current Holders (wallets)",   len(result["current_holders"])],
        ]
        print(tabulate(summary, tablefmt="fancy_grid"))

        print(f"\n  {Fore.YELLOW}Hold Duration Distribution:{Style.RESET_ALL}")
        bands = [
            [band, count] for band, count in
            sorted(result["retention_bands"].items(), key=lambda x: -x[1])
        ]
        print(tabulate(bands, headers=["Band", "Count"], tablefmt="simple"))

    @staticmethod
    def print_flip_velocity(result: dict) -> None:
        print(Reporter._header(
            f"⚡  FLIP VELOCITY  (threshold ≤ {result['flip_threshold_hours']}h)"
        ))

        print(f"  Overall Flip Rate: {Fore.YELLOW}{result['flip_rate']}%{Style.RESET_ALL}")

        if result["bot_candidates"]:
            print(
                f"\n  {Fore.RED}🤖  Bot Candidates (≥5 flips):{Style.RESET_ALL}"
            )
            for addr in result["bot_candidates"][:10]:
                print(f"      {addr}")

        flippers = result["flippers"]
        if not flippers.empty:
            top_flippers = flippers.head(15).copy()
            top_flippers["wallet"] = top_flippers["wallet"].str[:10] + "…"
            print(f"\n  {Fore.YELLOW}Top Flippers:{Style.RESET_ALL}")
            print(
                tabulate(
                    top_flippers,
                    headers=["Wallet", "Flip Count", "Avg Hold (hrs)", "Min Hold (hrs)"],
                    tablefmt="simple",
                    showindex=False,
                )
            )
        else:
            print(f"  {Fore.GREEN}✓  No significant flipping activity detected.{Style.RESET_ALL}")

    @staticmethod
    def print_concentration(result: dict) -> None:
        if "error" in result:
            print(f"  {Fore.RED}{result['error']}{Style.RESET_ALL}")
            return

        print(Reporter._header("🐋  CONCENTRATION METRICS"))

        summary = [
            ["Total Circulating Supply",  result["total_supply"]],
            ["Unique Holders",            result["unique_holders"]],
            ["HHI Score",                 f"{result['hhi']}  ({result['hhi_label']})"],
            ["Top-10 Wallets Share",       f"{result['top10_pct']}%"],
        ]
        print(tabulate(summary, tablefmt="fancy_grid"))

        print(f"\n  {Fore.YELLOW}Top 15 Holders:{Style.RESET_ALL}")
        top15 = result["holdings_df"].head(15).copy()
        top15["wallet"] = top15["wallet"].str[:12] + "…"
        print(
            tabulate(
                top15,
                headers=["Wallet", "Tokens Held", "% of Supply"],
                tablefmt="simple",
                showindex=False,
            )
        )

    # ── Markdown report ───────────────────────────────────────────────────

    @staticmethod
    def write_markdown(
        path: str,
        contract: str,
        df: pd.DataFrame,
        retention: dict,
        flip: dict,
        concentration: dict,
    ) -> None:
        """Write a structured Markdown report to *path*."""

        name  = df["token_name"].iloc[0] if not df.empty else "N/A"
        now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"# On-Chain Pattern Retrieval Scanner — Report",
            f"",
            f"> Generated: {now}",
            f"",
            f"## Collection Overview",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Contract | `{contract}` |",
            f"| Token Name | {name} |",
            f"| Total Records | {len(df)} |",
            f"| Mints | {(df['event_type']=='Mint').sum()} |",
            f"| Transfers | {(df['event_type']=='Transfer').sum()} |",
            f"| Burns | {(df['event_type']=='Burn').sum()} |",
            f"",
            f"## Wallet Retention Rate",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Average Hold Duration | **{retention['avg_hold_days']} days** |",
            f"| Median Hold Duration | {retention['median_hold_days']} days |",
            f"| Current Holders | {len(retention['current_holders'])} wallets |",
            f"",
            f"### Hold Duration Distribution",
            f"",
            f"| Band | Count |",
            f"|------|-------|",
        ]

        for band, count in sorted(
            retention["retention_bands"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {band} | {count} |")

        lines += [
            f"",
            f"## Flip Velocity",
            f"",
            f"- **Overall Flip Rate:** {flip['flip_rate']}%",
            f"- **Threshold:** ≤ {flip['flip_threshold_hours']} hours",
            f"- **Bot Candidates (≥5 flips):** {len(flip['bot_candidates'])}",
            f"",
        ]

        if flip["bot_candidates"]:
            lines.append("### Bot Candidate Addresses\n")
            for addr in flip["bot_candidates"]:
                lines.append(f"- `{addr}`")
            lines.append("")

        if not flip["flippers"].empty:
            lines += [
                f"### Top Flippers",
                f"",
                f"| Wallet | Flip Count | Avg Hold (hrs) |",
                f"|--------|-----------|----------------|",
            ]
            for _, row in flip["flippers"].head(10).iterrows():
                lines.append(
                    f"| `{row['wallet'][:14]}…` | {row['flip_count']} | {row['avg_hold_hrs']} |"
                )

        if "error" not in concentration:
            lines += [
                f"",
                f"## Concentration Metrics",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total Supply | {concentration['total_supply']} |",
                f"| Unique Holders | {concentration['unique_holders']} |",
                f"| HHI Score | {concentration['hhi']} ({concentration['hhi_label']}) |",
                f"| Top-10 Share | {concentration['top10_pct']}% |",
                f"",
                f"### Top 15 Holders",
                f"",
                f"| Wallet | Tokens | % Supply |",
                f"|--------|--------|---------|",
            ]
            for _, row in concentration["holdings_df"].head(15).iterrows():
                lines.append(
                    f"| `{row['wallet'][:14]}…` | {row['held_count']} | {row['pct_supply']}% |"
                )

        lines += ["", "---", "_Report generated by On-Chain Pattern Retrieval Scanner_"]

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        print(f"\n{Fore.GREEN}✓  Markdown report saved →  {path}{Style.RESET_ALL}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  CLI ENTRY POINT                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onchain_scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            On-Chain Pattern Retrieval Scanner
            ───────────────────────────────────
            Scan an NFT smart contract and surface behavioural patterns:
              • Wallet Retention Rate
              • Flip Velocity (bot / wash-trade detection)
              • Concentration Metrics (whale analysis)
        """),
    )
    parser.add_argument(
        "--contract", "-c",
        required=True,
        help="NFT contract address (ERC-721).  Example: 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
    )
    parser.add_argument(
        "--max-txns", "-n",
        type=int,
        default=500,
        metavar="N",
        help="Maximum number of transfer records to fetch (default: 500).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Write a Markdown report to FILE.md  (e.g. --output report)",
    )
    parser.add_argument(
        "--flip-threshold",
        type=float,
        default=48.0,
        metavar="HOURS",
        help="Hold duration threshold in hours to classify a flip (default: 48).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output (useful when piping).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    if args.no_color:
        # Strip all colour codes by overriding colorama
        import colorama
        colorama.deinit()

    # ── banner ────────────────────────────────────────────────────────────
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║    ON-CHAIN PATTERN RETRIEVAL SCANNER  v1.0.0                        ║
║    Target : {args.contract[:42]:<42}   ║
╚══════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}""")

    # ── 1. fetch ──────────────────────────────────────────────────────────
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    client  = EtherscanClient(api_key)

    try:
        raw = client.fetch_nft_transfers(args.contract, max_records=args.max_txns)
    except ValueError as exc:
        print(f"{Fore.RED}✗  {exc}{Style.RESET_ALL}")
        sys.exit(1)

    if not raw:
        print(f"{Fore.YELLOW}⚠  No transfer records found for this contract.{Style.RESET_ALL}")
        sys.exit(0)

    # ── 2. parse ──────────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}⬡  Parsing {len(raw)} records…{Style.RESET_ALL}")
    df = TransactionParser.parse(raw)
    print(f"   {Fore.GREEN}✓{Style.RESET_ALL}  {len(df)} records parsed successfully.\n")

    # ── 3. analyse ────────────────────────────────────────────────────────
    print(f"{Fore.CYAN}⬡  Running pattern analysis…{Style.RESET_ALL}")
    retention     = PatternEngine.wallet_retention(df)
    flip          = PatternEngine.flip_velocity(df, args.flip_threshold)
    concentration = PatternEngine.concentration_metrics(df)
    print(f"   {Fore.GREEN}✓{Style.RESET_ALL}  Analysis complete.\n")

    # ── 4. render ─────────────────────────────────────────────────────────
    Reporter.print_overview(args.contract, df)
    Reporter.print_retention(retention)
    Reporter.print_flip_velocity(flip)
    Reporter.print_concentration(concentration)

    # ── 5. optional markdown output ───────────────────────────────────────
    if args.output:
        md_path = args.output if args.output.endswith(".md") else args.output + ".md"
        Reporter.write_markdown(
            md_path, args.contract, df, retention, flip, concentration
        )

    print(f"\n{Fore.CYAN}{'─'*70}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}  ✓  Scan complete.{Style.RESET_ALL}\n")


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
