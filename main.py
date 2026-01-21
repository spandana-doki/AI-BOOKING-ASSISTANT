"""
Streamlit UI entry point for the AI Booking Assistant.

This is the ONLY file responsible for UI concerns (Streamlit widgets, rendering).
All non-UI logic is delegated to other modules.
"""

from __future__ import annotations

from typing import Dict, List
import os
import sys

import streamlit as st
import google.generativeai as genai

# ✅ IMPORTS (NO app. because there is no app folder)
from chat_logic import get_message_history, handle_user_message
from rag_pipeline import ingest_pdfs
from tools import booking_persistence_tool, email_tool
from admin_dashboard import render_admin_dashboard
from config import APP_NAME


STATUS_KEY = "status_messages"


def _init_ui_state() -> None:
    if STATUS_KEY not in st.session_state:
        st.session_state[STATUS_KEY] = []


def _push_status(level: str, message: str) -> None:
    _init_ui_state()
    st.session_state[STATUS_KEY].append({"level": level, "message": message})
    st.session_state[STATUS_KEY] = st.session_state[STATUS_KEY][-10:]


def _render_status_messages() -> None:
    _init_ui_state()
    for item in st.session_state[STATUS_KEY]:
        level = item.get("level", "info")
        msg = item.get("message", "")
        if not msg:
            continue
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "error":
            st.error(msg)
        else:
            st.info(msg)


def _render_chat_history(messages: List[Dict[str, str]]) -> None:
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if not content:
            continue
        with st.chat_message(role):
            st.markdown(content)


def _chat_page() -> None:
    st.title(APP_NAME)
    st.caption("Upload PDFs to enable RAG. Use chat below to ask questions or make a booking.")
    st.info("Using Google Gemini API for AI responses.")

    with st.sidebar:
        st.subheader("Knowledge Base")
        uploaded = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded:
            try:
                added = ingest_pdfs(uploaded)
                _push_status("success", f"Ingested {added} chunks from PDFs.")
            except Exception as exc:
                _push_status("error", f"PDF ingestion failed: {exc}")

    _render_status_messages()

    history = get_message_history()
    if not history:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I can answer questions using your PDFs or help you place a booking. "
                "How can I help today?"
            )
    else:
        _render_chat_history(history)

    user_text = st.chat_input("Type your message...")
    if not user_text:
        return

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.spinner("Thinking..."):
        assistant_reply, booking_payload = handle_user_message(user_text)

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    if booking_payload is not None:
        persist = booking_persistence_tool(booking_payload)
        if persist.get("success"):
            booking_id = persist.get("booking_id")
            _push_status("success", f"Booking saved (ID={booking_id}).")

            email_res = email_tool(
                to_email=booking_payload.email,
                subject=f"Booking Confirmation (ID: {booking_id})",
                body=(
                    "Your booking is confirmed.\n\n"
                    f"Name: {booking_payload.name}\n"
                    f"Email: {booking_payload.email}\n"
                    f"Phone: {booking_payload.phone}\n"
                    f"Type: {booking_payload.booking_type}\n"
                    f"Date: {booking_payload.date}\n"
                    f"Time: {booking_payload.time}\n\n"
                    "Thank you."
                ),
            )

            if email_res.get("success"):
                _push_status("success", "Confirmation email sent.")
            else:
                _push_status("warning", "Booking saved but email failed.")
        else:
            _push_status("error", "Booking confirmed but DB save failed.")

        st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    _init_ui_state()

    # ✅ Load Gemini API key from Streamlit Secrets
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        genai.configure(api_key=api_key)
    else:
        _push_status(
            "error",
            "Missing GEMINI_API_KEY. Add it in Streamlit Cloud → Settings → Secrets.",
        )

    with st.sidebar:
        st.header(APP_NAME)
        page = st.radio("Navigate", ["Chat", "Admin Dashboard"])

    if page == "Admin Dashboard":
        render_admin_dashboard()
    else:
        _chat_page()


if __name__ == "__main__":
    main()
