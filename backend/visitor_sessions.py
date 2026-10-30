
"""
VideoMoment public visitor session utilities.

Creates and validates cryptographically secure visitor tokens.

Tokens are intended to be held in browser session memory,
not placed in URLs or stored in localStorage.

This module provides token utilities only.
Database persistence and API authorization are added
in subsequent steps.
"""

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone


TOKEN_BYTES = 32
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")

SESSION_LIFETIME_SECONDS = 7200


def utc_now():
    return datetime.now(timezone.utc)


def generate_visitor_token() -> str:
    """
    Generate a cryptographically secure visitor token.

    32 random bytes produce a 43-character URL-safe token.
    """
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_visitor_token(token: str) -> str:
    """
    Hash a visitor token for database storage.
    """
    if not isinstance(token, str):
        raise ValueError("Visitor token must be a string.")

    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Invalid visitor token format.")

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def verify_visitor_token(
    supplied_token: str,
    stored_token_hash: str,
) -> bool:
    """
    Compare a supplied visitor token against its stored hash.
    """
    try:
        supplied_hash = hash_visitor_token(supplied_token)
    except ValueError:
        return False

    if not isinstance(stored_token_hash, str):
        return False

    return hmac.compare_digest(
        supplied_hash,
        stored_token_hash
    )


def generate_visitor_id() -> str:
    """
    Generate a random visitor identifier.
    """
    return secrets.token_hex(16)


def session_expiry_time():
    """
    Return the expiry time for a newly created session.
    """
    return utc_now() + timedelta(
        seconds=SESSION_LIFETIME_SECONDS
    )


def is_session_expired(expires_at) -> bool:
    """
    Check whether a stored session has expired.
    """
    if expires_at is None:
        return True

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(
            tzinfo=timezone.utc
        )

    return utc_now() >= expires_at
