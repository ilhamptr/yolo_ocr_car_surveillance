"""Streamlit UI for the CCTV parking-zone tracker.

Users type their plate to find which sector their car is parked in.
Detection runs automatically in the background (every N minutes), so the
UI only reads from the API. A demo button can trigger a cycle on demand.
"""

import os

import requests
import streamlit as st

# Configurable so the UI can reach the API by service name inside Docker.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Find My Car")
st.title("Find My Car")
st.caption("Type your license plate to see which parking sector it's in.")


def api_get(path, **params):
    try:
        return requests.get(f"{API_URL}{path}", params=params, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is the backend running?")
        return None


tab_find, tab_all, tab_status = st.tabs(["Find my car", "All parked", "Status"])


# ------------------------------------------------------------------ Find tab
with tab_find:
    query = st.text_input("Your license plate", placeholder="e.g. B214FEX")

    if st.button("Find", type="primary"):
        if not query:
            st.error("Please enter a plate.")
        else:
            resp = api_get("/search", plate=query)
            if resp is not None and resp.ok:
                results = resp.json()["results"]
                if not results:
                    st.warning(
                        f"'{query}' is not currently parked "
                        "(not detected in the latest cycle)."
                    )
                else:
                    for r in results:
                        st.success(
                            f"**{r['plate']}** is in sector **{r['sector']}**"
                        )
                        st.caption(f"Last seen: {r['last_seen']}")


# ------------------------------------------------------------- All parked tab
with tab_all:
    st.subheader("Currently parked vehicles")
    if st.button("Refresh list"):
        st.rerun()

    resp = api_get("/vehicles")
    if resp is not None and resp.ok:
        results = resp.json()["results"]
        if results:
            st.dataframe(
                [
                    {
                        "Plate": r["plate"],
                        "Sector": r["sector"],
                        "Last seen": r["last_seen"],
                        "Confidence": r.get("confidence"),
                    }
                    for r in results
                ],
                use_container_width=True,
            )
        else:
            st.info("No vehicles detected yet. Run a cycle from the Status tab.")


# ---------------------------------------------------------------- Status tab
with tab_status:
    st.subheader("System status")
    if st.button("Refresh status"):
        st.rerun()

    resp = api_get("/status")
    if resp is not None and resp.ok:
        data = resp.json()
        st.write(f"**Sectors monitored:** {', '.join(data.get('sectors', [])) or 'none'}")
        st.write(f"**Workers started:** {data.get('started_at', '-')}")
        st.write(f"**Total plate reads:** {data.get('reads', 0)}")
        st.write("**Last expiry check:**")
        st.json(data.get("last_expiry") or {"info": "no expiry run yet"})

    st.caption(
        "Each sector is monitored continuously. YOLO runs only when motion "
        "is detected, so idle cameras cost almost nothing."
    )
