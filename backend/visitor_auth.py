
"""
VideoMoment visitor authentication and video authorization.

Features:
- Secure anonymous visitor sessions
- Hashed tokens stored in PostgreSQL
- Two-hour session expiry
- Per-visitor video ownership
- Public access to explicitly configured sample videos
- Optional access-code checks handled separately

Public requests use:
    Authorization: Bearer <visitor_token>

Important:
- The frontend must obtain a session before uploading.
- Upload and YouTube import must assign visitor_id.
- Every private video route must call require_video_access.
"""

import logging
from datetime import datetime, timezone

from fastapi import HTTPException, Request
from sqlalchemy import select

from database import SessionLocal
from models import Video, VisitorSession

from deployment_config import IS_PUBLIC, SAMPLE_VIDEO_IDS

from visitor_sessions import (
    generate_visitor_token,
    generate_visitor_id,
    hash_visitor_token,
    session_expiry_time,
    is_session_expired,
)

logger = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc)


def create_visitor_session() -> dict:
    """
    Create a new anonymous visitor session.

    Only the token hash is stored in PostgreSQL.
    The original token is returned once to the client.
    """
    token = generate_visitor_token()
    visitor_id = generate_visitor_id()

    with SessionLocal() as db:
        visitor = VisitorSession(
            id=visitor_id,
            token_hash=hash_visitor_token(token),
            created_at=utc_now(),
            expires_at=session_expiry_time(),
        )

        db.add(visitor)
        db.commit()
        db.refresh(visitor)

        return {
            "visitor_id":visitor.id,
            "visitor_token":token,
            "expires_at":visitor.expires_at.isoformat(),
        }


def extract_bearer_token(request: Request) -> str:
    """
    Read the bearer token from the Authorization header.
    """
    authorization = request.headers.get(
        "authorization", ""
    )

    scheme, separator, token = authorization.partition(" ")

    if (
        not separator
        or scheme.lower() != "bearer"
        or not token
    ):
        raise HTTPException(
            status_code=401,
            detail="Visitor session required.",
            headers={"WWW-Authenticate":"Bearer"},
        )

    return token.strip()


def get_current_visitor(request: Request) -> VisitorSession:
    """
    Validate a visitor token and return the active session.

    Returns a detached ORM object; use its scalar fields
    only after the database session closes.
    """
    token = extract_bearer_token(request)

    try:
        token_hash = hash_visitor_token(token)
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid visitor session.",
        )

    with SessionLocal() as db:
        visitor = db.scalar(
            select(VisitorSession).where(
                VisitorSession.token_hash == token_hash
            )
        )

        if visitor is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid visitor session.",
            )

        if is_session_expired(visitor.expires_at):
            raise HTTPException(
                status_code=401,
                detail="Visitor session expired.",
            )

        return visitor


def get_current_visitor_id(request: Request) -> str:
    """
    Return the authenticated visitor ID.
    """
    visitor = get_current_visitor(request)
    return visitor.id


def require_video_access(
    video_id: str,
    request: Request,
) -> Video:
    """
    Authorize access to a video.

    Public samples are accessible without a visitor token.

    In public mode, all other videos require a valid
    visitor session belonging to their owner.

    In local mode, existing behavior is preserved.
    """
    with SessionLocal() as db:
        video = db.get(Video, video_id)

        if video is None:
            raise HTTPException(
                status_code=404,
                detail="Video not found.",
            )

        if not IS_PUBLIC:
            return video

        if video_id in SAMPLE_VIDEO_IDS:
            return video

        visitor_id = get_current_visitor_id(request)

        if video.visitor_id != visitor_id:
            raise HTTPException(
                status_code=404,
                detail="Video not found.",
            )

        return video


def assign_video_owner(
    video_id: str,
    request: Request,
) -> str | None:
    """
    Assign an uploaded video to the authenticated visitor.

    Called immediately after creating a video record.

    In local mode, ownership remains optional.
    """
    if not IS_PUBLIC:
        return None

    visitor_id = get_current_visitor_id(request)

    with SessionLocal() as db:
        video = db.get(Video, video_id)

        if video is None:
            raise HTTPException(
                status_code=404,
                detail="Video not found.",
            )

        if video.visitor_id is not None:
            if video.visitor_id != visitor_id:
                raise HTTPException(
                    status_code=403,
                    detail="Video already belongs to another visitor.",
                )

            return visitor_id

        video.visitor_id = visitor_id
        db.commit()

    return visitor_id
