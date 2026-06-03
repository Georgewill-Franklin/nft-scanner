import streamlit as st
import pandas as pd
from onchain_scanner import EtherscanClient, TransactionParser, PatternEngine

st.set_page_config(page_title="On-Chain Pattern Scanner", page_icon="⬡", layout="wide")

st.title("⬡ On-Chain Pattern Scanner")
st.caption("Scan any NFT contract for bots, whale concentration, and holder behaviour.")

contract = st.text_input("Paste NFT Contract Address", placeholder="0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D")
max_txns = st.slider("Max transactions to fetch", 100, 2000, 500, step=100)

if st.button("🔍 Scan Contract", use_container_width=True):
    if not contract.strip():
        st.warning("Please enter a contract address.")
    else:
        try:
            api_key = st.secrets["ETHERSCAN_API_KEY"]
            client = EtherscanClient(api_key)

            with st.spinner("Fetching transactions from Etherscan..."):
                raw = client.fetch_nft_transfers(contract.strip(), max_records=max_txns)

            if not raw:
                st.warning("No transfer records found for this contract.")
            else:
                df = TransactionParser.parse(raw)

                st.success(f"✓ {len(df)} records fetched and parsed.")

                # Overview
                st.subheader("📊 Collection Overview")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Records", len(df))
                col2.metric("Mints", int((df["event_type"] == "Mint").sum()))
                col3.metric("Transfers", int((df["event_type"] == "Transfer").sum()))
                col4.metric("Burns", int((df["event_type"] == "Burn").sum()))

                # Retention
                st.subheader("🕒 Wallet Retention")
                retention = PatternEngine.wallet_retention(df)
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Hold Duration", f"{retention['avg_hold_days']} days")
                col2.metric("Median Hold Duration", f"{retention['median_hold_days']} days")
                col3.metric("Current Holders", len(retention["current_holders"]))

                bands_df = pd.DataFrame(
                    list(retention["retention_bands"].items()),
                    columns=["Hold Duration", "Count"]
                ).sort_values("Count", ascending=False)
                st.bar_chart(bands_df.set_index("Hold Duration"))

                # Flip velocity
                st.subheader("⚡ Flip Velocity (Bot Detection)")
                flip = PatternEngine.flip_velocity(df)
                col1, col2 = st.columns(2)
                col1.metric("Overall Flip Rate", f"{flip['flip_rate']}%")
                col2.metric("Bot Candidates (≥5 flips)", len(flip["bot_candidates"]))

                if not flip["flippers"].empty:
                    st.dataframe(flip["flippers"].head(10), use_container_width=True)

                # Concentration
                st.subheader("🐋 Whale Concentration")
                concentration = PatternEngine.concentration_metrics(df)
                if "error" not in concentration:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Supply", concentration["total_supply"])
                    col2.metric("Unique Holders", concentration["unique_holders"])
                    col3.metric("HHI Score", f"{concentration['hhi']} ({concentration['hhi_label']})")
                    col4.metric("Top-10 Share", f"{concentration['top10_pct']}%")
                    st.dataframe(concentration["holdings_df"].head(15), use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
