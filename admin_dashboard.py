"""
Streamlit admin dashboard for viewing bookings.

Features:
- Table view of all bookings
- Search/filter by customer name or email
- Read-only (no booking creation)
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from db.database import fetch_all_bookings, init_db


def _filter_bookings(bookings: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Return bookings filtered by name or email (case-insensitive substring match)."""
    q = query.strip().lower()
    if not q:
        return bookings
    return [
        b
        for b in bookings
        if q in str(b.get("customer_name", "")).lower()
        or q in str(b.get("customer_email", "")).lower()
    ]


def render_admin_dashboard() -> None:
    """Render the admin dashboard content."""
    st.title("Admin Dashboard")
    st.caption("Read-only view of all bookings. Search by customer name or email.")

    # Ensure database schema exists (no data writes here).
    init_db()

    search_query = st.text_input("Search by name or email", value="")

    with st.spinner("Loading bookings..."):
        bookings = fetch_all_bookings()

    filtered = _filter_bookings(bookings, search_query)

    st.write(f"Total bookings: {len(filtered)}")

    if not filtered:
        st.info("No bookings found matching the current filter.")
        return

    # Define a readable column order for display.
    columns = [
        "booking_id",
        "customer_name",
        "customer_email",
        "customer_phone",
        "service",
        "booking_date",
        "booking_time",
        "status",
        "notes",
        "booking_created_at",
    ]

    # Normalize rows to ensure consistent keys for st.dataframe
    normalized_rows = []
    for row in filtered:
        normalized_rows.append({col: row.get(col, "") for col in columns})

    st.dataframe(normalized_rows, use_container_width=True)


if __name__ == "__main__":
    render_admin_dashboard()



