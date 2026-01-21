"""
Lightweight SQLite database utilities for the AI Booking Assistant.

This module:
- Connects to the SQLite database
- Initializes required tables
- Exposes helper functions to insert customers and bookings
- Provides a convenience function to fetch all bookings (for the admin dashboard)

No ORM is used; all queries are written in plain SQL.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from app.config import SQLITE_DB_PATH


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection and ensures it is closed.

    Foreign key support is enabled for each connection.
    """
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            conn.rollback()
        # In a real application, swap this for proper logging
        print(f"[DB ERROR] {exc}")
    finally:
        if conn is not None:
            conn.close()


def init_db() -> None:
    """
    Initialize the database schema if tables do not exist.

    Tables:
    - customers
    - bookings
    """
    with get_connection() as conn:
        if conn is None:
            return

        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bookings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    service TEXT NOT NULL,
                    booking_date TEXT NOT NULL,
                    booking_time TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers (id)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE
                );
                """
            )
        except sqlite3.Error as exc:
            print(f"[DB INIT ERROR] {exc}")


def insert_customer(name: str, email: str, phone: Optional[str] = None) -> Optional[int]:
    """
    Insert a new customer and return the new customer ID.

    Returns None if the insert fails.
    """
    with get_connection() as conn:
        if conn is None:
            return None

        try:
            cursor = conn.execute(
                """
                INSERT INTO customers (name, email, phone)
                VALUES (?, ?, ?);
                """,
                (name, email, phone),
            )
            return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            print(f"[DB INSERT CUSTOMER ERROR] {exc}")
            return None


def insert_booking(
    customer_id: int,
    service: str,
    booking_date: str,
    booking_time: str,
    status: str = "pending",
    notes: Optional[str] = None,
) -> Optional[int]:
    """
    Insert a new booking and return the new booking ID.

    Returns None if the insert fails.
    """
    with get_connection() as conn:
        if conn is None:
            return None

        try:
            cursor = conn.execute(
                """
                INSERT INTO bookings (
                    customer_id,
                    service,
                    booking_date,
                    booking_time,
                    status,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (customer_id, service, booking_date, booking_time, status, notes),
            )
            return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            print(f"[DB INSERT BOOKING ERROR] {exc}")
            return None


def fetch_all_bookings() -> List[Dict[str, Any]]:
    """
    Fetch all bookings with basic customer information.

    Returns:
        A list of dictionaries, each representing a booking row joined with
        the corresponding customer.
    """
    results: List[Dict[str, Any]] = []

    with get_connection() as conn:
        if conn is None:
            return results

        try:
            cursor = conn.execute(
                """
                SELECT
                    b.id AS booking_id,
                    b.customer_id,
                    c.name AS customer_name,
                    c.email AS customer_email,
                    c.phone AS customer_phone,
                    b.service,
                    b.booking_date,
                    b.booking_time,
                    b.status,
                    b.notes,
                    b.created_at AS booking_created_at
                FROM bookings b
                JOIN customers c ON b.customer_id = c.id
                ORDER BY b.created_at DESC;
                """
            )
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
        except sqlite3.Error as exc:
            print(f"[DB FETCH BOOKINGS ERROR] {exc}")

    return results



