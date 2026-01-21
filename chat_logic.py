"""
Core chat orchestration logic for the AI Booking Assistant.

Responsibilities:
- Detect intent: booking vs general query
- Maintain the last 25 messages using `st.session_state`
- Route:
    - General questions → RAG (`rag_pipeline.answer_query`)
    - Booking intent → booking flow (`booking_flow.handle_booking_flow`)
- Prevent repeated questions using conversational memory

This module deliberately contains **no Streamlit UI components** such as
`st.write` or `st.chat_message`. It only manipulates `st.session_state`
and returns strings / payloads for the caller to render.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

import streamlit as st

from .booking_flow import BookingData, BookingState, handle_booking_flow
from .tools import rag_tool


SESSION_MESSAGES_KEY = "messages"
SESSION_BOOKING_STATE_KEY = "booking_state"
MAX_MESSAGES = 25


def _init_session_state() -> None:
    """Ensure required session_state keys exist."""
    if SESSION_MESSAGES_KEY not in st.session_state:
        st.session_state[SESSION_MESSAGES_KEY] = []  # type: ignore[assignment]
    if SESSION_BOOKING_STATE_KEY not in st.session_state:
        st.session_state[SESSION_BOOKING_STATE_KEY] = None


def get_message_history() -> List[Dict[str, str]]:
    """Return the current message history from session state."""
    _init_session_state()
    return list(st.session_state[SESSION_MESSAGES_KEY])  # type: ignore[return-value]


def _add_message(role: str, content: str) -> None:
    """
    Append a message to the session history and keep only the last MAX_MESSAGES.
    """
    _init_session_state()
    messages: List[Dict[str, str]] = st.session_state[SESSION_MESSAGES_KEY]  # type: ignore[assignment]
    messages.append({"role": role, "content": content})
    # Trim history to last MAX_MESSAGES
    if len(messages) > MAX_MESSAGES:
        overflow = len(messages) - MAX_MESSAGES
        del messages[0:overflow]


def _normalize_text(text: str) -> str:
    """Lowercase and strip extra whitespace for comparison."""
    return " ".join(text.strip().lower().split())


def _find_previous_answer_for_question(user_input: str) -> Optional[str]:
    """
    If the user has already asked this (or a very similar) question,
    return the previous assistant reply, otherwise None.
    """
    history = get_message_history()
    norm_input = _normalize_text(user_input)

    # Scan history for a matching user question and the next assistant reply
    for idx, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        if _normalize_text(msg.get("content", "")) == norm_input:
            # Look for the next assistant message in history
            for j in range(idx + 1, len(history)):
                if history[j].get("role") == "assistant":
                    return history[j].get("content", "")

    return None


def detect_intent(text: str) -> str:
    """
    Very lightweight intent detection between 'booking' and 'general'.

    Uses simple keyword heuristics rather than an LLM to keep this
    module lightweight and deterministic.
    """
    lowered = text.lower().strip()

    # Treat clearly conversational / meta questions as general,
    # even if the word "booking" appears (e.g. "AI Booking Assistant project").
    if any(word in lowered for word in ["project", "requirements", "objective", "overview", "how it works"]):
        return "general"

    # Look for verbs that strongly indicate intent to make or manage a booking.
    # Examples: "I want to book a table", "Can you schedule an appointment", "cancel my booking"
    booking_patterns = [
        r"\bbook\b",
        r"\bbook\s+(a|an|the)\b",
        r"\bmake\s+a\s+booking\b",
        r"\bcreate\s+a\s+booking\b",
        r"\breserve\b",
        r"\breservation\b",
        r"\bschedule\b",
        r"\bappointment\b",
        r"\bcancel\s+my\s+booking\b",
        r"\bchange\s+my\s+booking\b",
    ]

    if any(re.search(pat, lowered) for pat in booking_patterns):
        return "booking"

    return "general"


def _get_booking_state() -> Optional[BookingState]:
    _init_session_state()
    state = st.session_state[SESSION_BOOKING_STATE_KEY]
    return state


def _set_booking_state(state: Optional[BookingState]) -> None:
    _init_session_state()
    st.session_state[SESSION_BOOKING_STATE_KEY] = state


def handle_user_message(
    user_input: str,
) -> Tuple[str, Optional[BookingData]]:
    """
    Main entry point for chat handling.

    Steps:
    1. Initializes session state if needed.
    2. Checks if the question was already asked; if so, reuses the prior answer.
    3. Detects intent (booking vs general).
    4. Routes to the appropriate handler:
       - Booking flow for booking intent.
       - RAG pipeline for general questions.
    5. Updates session history and (optionally) booking state.

    Args:
        user_input: The latest user message content.

    Returns:
        assistant_reply: Text response that the UI should display.
        booking_payload: A BookingData object once a booking is confirmed, otherwise None.
    """
    user_input = user_input or ""
    user_input = user_input.strip()

    _init_session_state()

    if not user_input:
        reply = "Please type a message so I can assist you."
        _add_message("assistant", reply)
        return reply, None

    # Add user message to history first
    _add_message("user", user_input)

    # 1) Prevent repeated questions using memory
    previous_answer = _find_previous_answer_for_question(user_input)
    if previous_answer is not None:
        reply = (
            "You asked a similar question earlier. Here’s the answer I shared before:\n\n"
            f"{previous_answer}"
        )
        _add_message("assistant", reply)
        return reply, None

    # 2) If a booking flow is already in progress, continue it regardless of new intent keywords.
    current_state = _get_booking_state()
    if current_state and current_state.stage in {"collecting", "confirm"}:
        response_text, new_state, payload = handle_booking_flow(
            current_state,
            user_input,
        )
        _set_booking_state(new_state)
        _add_message("assistant", response_text)
        return response_text, payload

    # 3) Detect intent
    intent = detect_intent(user_input)

    # 4) Route based on intent
    if intent == "booking":
        response_text, new_state, payload = handle_booking_flow(
            current_state,
            user_input,
        )
        _set_booking_state(new_state)
        _add_message("assistant", response_text)
        return response_text, payload

    # General query → RAG
    history = get_message_history()
    rag_result = rag_tool(query=user_input, chat_history=history[-10:])
    answer = rag_result.get("answer") if rag_result.get("success") else None

    if not answer:
        error = rag_result.get("error") if isinstance(rag_result, dict) else None
        answer = (
            "I’m not fully sure based on the information I have. "
            "If you upload a PDF or provide more details, I can try again."
        )
        if error:
            answer += f"\n\n(Details: {error})"

    _add_message("assistant", answer)
    return answer, None


__all__ = [
    "detect_intent",
    "get_message_history",
    "handle_user_message",
]



