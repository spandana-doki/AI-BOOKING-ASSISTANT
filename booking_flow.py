"""
Conversation-oriented booking flow helpers.

Responsibilities:
- Detect missing booking fields:
    name, email, phone, booking_type, date, time
- Ask only for missing fields
- Validate email, date (YYYY-MM-DD), and time (HH:MM)
- Summarize booking details
- Ask for explicit confirmation (yes/no)
- Return a structured booking payload only after confirmation

No database access is performed in this module; it is purely
concerned with conversational logic and validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple


REQUIRED_FIELDS = ["name", "email", "phone", "booking_type", "date", "time"]


@dataclass
class BookingData:
    """Structured representation of a booking."""

    name: str
    email: str
    phone: str
    booking_type: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM


@dataclass
class BookingState:
    """
    Tracks the state of an in-progress booking conversation.

    Attributes:
        booking: Partially filled booking fields.
        stage: 'collecting', 'confirm', 'completed', or 'cancelled'.
        awaiting_field: The specific field we are currently asking the user for.
        confirmed: Whether the user has explicitly confirmed the booking.
    """

    booking: Dict[str, str]
    stage: str = "collecting"
    awaiting_field: Optional[str] = None
    confirmed: bool = False


def _is_valid_email(email: str) -> bool:
    """Basic email validation using a simple regex."""
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, email.strip()))


def _is_valid_date(date_str: str) -> bool:
    """Validate date in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _is_valid_time(time_str: str) -> bool:
    """Validate time in HH:MM (24-hour) format."""
    try:
        datetime.strptime(time_str.strip(), "%H:%M")
        return True
    except ValueError:
        return False


def detect_missing_fields(booking: Dict[str, str]) -> Dict[str, bool]:
    """
    Return a dict indicating which required fields are missing or empty.
    """
    return {field: not bool(booking.get(field)) for field in REQUIRED_FIELDS}


def _validate_field(field: str, value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single field value.

    Returns:
        (is_valid, error_message_if_any)
    """
    value = value.strip()

    if field in ("name", "booking_type"):
        if not value:
            return False, f"Please provide a valid {field.replace('_', ' ')}."
        return True, None

    if field == "email":
        if not _is_valid_email(value):
            return False, "That email address doesn't look valid. Please enter a valid email (e.g. name@example.com)."
        return True, None

    if field == "phone":
        # Keep this loose; just require a minimum length.
        digits = re.sub(r"\D", "", value)
        if len(digits) < 7:
            return False, "Please provide a valid phone number (at least 7 digits)."
        return True, None

    if field == "date":
        if not _is_valid_date(value):
            return False, "Please enter a valid date in the format YYYY-MM-DD."
        return True, None

    if field == "time":
        if not _is_valid_time(value):
            return False, "Please enter a valid time in 24-hour format HH:MM (e.g. 14:30)."
        return True, None

    # Unknown fields are treated as valid by default
    return True, None


def summarize_booking(booking: Dict[str, str]) -> str:
    """
    Return a human-readable summary of the booking details.
    """
    name = booking.get("name", "").strip()
    email = booking.get("email", "").strip()
    phone = booking.get("phone", "").strip()
    booking_type = booking.get("booking_type", "").strip()
    date = booking.get("date", "").strip()
    time = booking.get("time", "").strip()

    lines = [
        "Here are your booking details:",
        f"- Name: {name or 'N/A'}",
        f"- Email: {email or 'N/A'}",
        f"- Phone: {phone or 'N/A'}",
        f"- Booking type: {booking_type or 'N/A'}",
        f"- Date: {date or 'N/A'}",
        f"- Time: {time or 'N/A'}",
    ]
    return "\n".join(lines)


def _next_missing_field(booking: Dict[str, str]) -> Optional[str]:
    """
    Return the next missing required field, or None if all are present.
    """
    for field in REQUIRED_FIELDS:
        if not booking.get(field):
            return field
    return None


def _field_prompt(field: str) -> str:
    """Return a user-friendly prompt for a given field."""
    if field == "name":
        return "To get started, what's your full name?"
    if field == "email":
        return "Please provide your email address."
    if field == "phone":
        return "What is the best phone number to reach you?"
    if field == "booking_type":
        return "What type of booking would you like to make (e.g. consultation, demo, reservation)?"
    if field == "date":
        return "On which date would you like the booking? (format: YYYY-MM-DD)"
    if field == "time":
        return "At what time? (24-hour format HH:MM, e.g. 14:30)"
    return f"Please provide a value for {field}."


def handle_booking_flow(
    state: Optional[BookingState],
    user_input: Optional[str],
) -> Tuple[str, BookingState, Optional[BookingData]]:
    """
    Core booking flow state machine.

    This function is designed to be called repeatedly by a chat layer.
    It:
    - Tracks which fields are missing
    - Asks only for missing fields
    - Validates user input for each field
    - Produces a natural-language summary and asks for explicit yes/no confirmation
    - Returns a structured BookingData payload only once confirmed

    Args:
        state: The current BookingState (or None to start a new booking).
        user_input: The latest user message (may be None at the very start).

    Returns:
        response_text: What the assistant should say next.
        new_state: Updated BookingState.
        booking_payload: A BookingData instance once the booking is confirmed, otherwise None.
    """
    if state is None:
        state = BookingState(booking={})

    user_text = (user_input or "").strip()

    # If we are awaiting confirmation
    if state.stage == "confirm":
        if not user_text:
            return (
                "Please respond with 'yes' to confirm the booking or 'no' to cancel.",
                state,
                None,
            )

        normalized = user_text.lower()
        if normalized in {"yes", "y", "confirm", "sure"}:
            state.confirmed = True
            state.stage = "completed"

            payload = BookingData(
                name=state.booking["name"].strip(),
                email=state.booking["email"].strip(),
                phone=state.booking["phone"].strip(),
                booking_type=state.booking["booking_type"].strip(),
                date=state.booking["date"].strip(),
                time=state.booking["time"].strip(),
            )

            return (
                "Great, your booking is confirmed!",
                state,
                payload,
            )

        if normalized in {"no", "n", "cancel"}:
            state.stage = "cancelled"
            state.confirmed = False
            return (
                "Okay, I’ve cancelled this booking request. If you’d like to start over, just let me know.",
                state,
                None,
            )

        return (
            "Please answer with 'yes' to confirm the booking or 'no' to cancel.",
            state,
            None,
        )

    # If we are collecting information and currently waiting on a specific field
    if state.stage == "collecting" and state.awaiting_field:
        field = state.awaiting_field
        if not user_text:
            return (
                "I didn't catch that. " + _field_prompt(field),
                state,
                None,
            )

        is_valid, error = _validate_field(field, user_text)
        if not is_valid:
            return (
                error,
                state,
                None,
            )

        # Save the validated value
        state.booking[field] = user_text.strip()
        state.awaiting_field = None

    # At this point, either we have just started or we've just filled a field.
    missing_field = _next_missing_field(state.booking)
    if missing_field:
        state.stage = "collecting"
        state.awaiting_field = missing_field
        return _field_prompt(missing_field), state, None

    # All required fields present: move to confirmation
    if state.stage == "collecting":
        state.stage = "confirm"
        summary = summarize_booking(state.booking)
        confirmation_prompt = (
            f"{summary}\n\n"
            "Please confirm: do you want me to place this booking? (yes/no)"
        )
        return confirmation_prompt, state, None

    # Fallback: if we somehow reach here, ask for confirmation again.
    if state.stage not in {"completed", "cancelled"}:
        state.stage = "confirm"
        summary = summarize_booking(state.booking)
        return (
            f"{summary}\n\nPlease confirm this booking with 'yes' or 'no'.",
            state,
            None,
        )

    # If already completed or cancelled, acknowledge and do nothing else
    if state.stage == "completed":
        # Booking is already confirmed; return a payload if we have all fields.
        if all(state.booking.get(k) for k in REQUIRED_FIELDS):
            payload = BookingData(
                name=state.booking["name"].strip(),
                email=state.booking["email"].strip(),
                phone=state.booking["phone"].strip(),
                booking_type=state.booking["booking_type"].strip(),
                date=state.booking["date"].strip(),
                time=state.booking["time"].strip(),
            )
            return "This booking has already been confirmed.", state, payload
        return "This booking has already been confirmed.", state, None

    return "This booking flow has been cancelled. Start a new booking if needed.", state, None


__all__ = [
    "BookingData",
    "BookingState",
    "detect_missing_fields",
    "summarize_booking",
    "handle_booking_flow",
]



