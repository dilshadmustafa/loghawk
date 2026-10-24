"""Password authentication and expiring bearer sessions for LogHawk clients."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import base64
import hashlib
import hmac
import secrets

from loghawk.database.database_layer import connect, initialize_database


_ITERATIONS = 310_000
_SESSION_HOURS = 12


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(value: datetime) -> str:
    return value.isoformat()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _ITERATIONS,
        base64.urlsafe_b64encode(salt).decode(),
        base64.urlsafe_b64encode(digest).decode(),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt, expected = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        salt_bytes = base64.urlsafe_b64decode(salt.encode())
        expected_bytes = base64.urlsafe_b64decode(expected.encode())
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt_bytes, int(iterations)
        )
        return hmac.compare_digest(actual, expected_bytes)
    except (ValueError, TypeError):
        return False


def initialize_auth() -> None:
    """Create the one-time demo administrator, without resetting its password."""
    initialize_database("server")
    with connect("server") as db:
        existing = db.execute(
            "SELECT userid FROM auth_users WHERE userid = ?", ("admin",)
        ).fetchone()
        if existing is None:
            now = _stamp(_now())
            db.execute(
                """INSERT INTO auth_users
                   (userid, password_hash, role, active, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("admin", hash_password("admin"), "Administrator", True, now, now),
            )


def authenticate(userid: str, password: str, client: str) -> dict | None:
    initialize_database("server")
    with connect("server") as db:
        row = db.execute(
            "SELECT userid, password_hash, role, active FROM auth_users WHERE userid = ?",
            (userid.strip(),),
        ).fetchone()
        if not row or not row["active"] or not verify_password(password, row["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        now = _now()
        expires = now + timedelta(hours=_SESSION_HOURS)
        db.execute(
            """INSERT INTO auth_sessions
               (token_hash, userid, client, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                hashlib.sha256(token.encode()).hexdigest(), row["userid"], client,
                _stamp(now), _stamp(expires),
            ),
        )
    return {"access_token": token, "token_type": "bearer", "expires_at": _stamp(expires),
            "user": {"userid": row["userid"], "role": row["role"]}}


def user_for_token(token: str, client: str | None = None) -> dict | None:
    initialize_database("server")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with connect("server") as db:
        row = db.execute(
            """SELECT u.userid, u.role, s.client, s.expires_at
               FROM auth_sessions s JOIN auth_users u ON u.userid = s.userid
               WHERE s.token_hash = ? AND s.revoked_at IS NULL AND u.active = ?""",
            (token_hash, True),
        ).fetchone()
    if row is None or (client and row["client"] != client):
        return None
    try:
        if datetime.fromisoformat(row["expires_at"]) <= _now():
            return None
    except (TypeError, ValueError):
        return None
    return {"userid": row["userid"], "role": row["role"], "client": row["client"]}


def revoke_token(token: str) -> None:
    with connect("server") as db:
        db.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (_stamp(_now()), hashlib.sha256(token.encode()).hexdigest()),
        )


def change_password(userid: str, current_password: str, new_password: str) -> bool:
    with connect("server") as db:
        row = db.execute(
            "SELECT password_hash FROM auth_users WHERE userid = ? AND active = ?",
            (userid, True),
        ).fetchone()
        if row is None or not verify_password(current_password, row["password_hash"]):
            return False
        db.execute(
            "UPDATE auth_users SET password_hash = ?, updated_at = ? WHERE userid = ?",
            (hash_password(new_password), _stamp(_now()), userid),
        )
    return True


def list_users() -> list[dict]:
    with connect("server") as db:
        rows = db.execute(
            "SELECT userid, role, active, created_at, updated_at FROM auth_users ORDER BY userid COLLATE NOCASE"
        ).fetchall()
    return [dict(row) for row in rows]


def create_user(userid: str, password: str, role: str) -> dict:
    userid = userid.strip()
    if not userid or len(userid) > 128:
        raise ValueError("User ID must contain between 1 and 128 characters")
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    if role not in {"User", "Power User", "Administrator"}:
        raise ValueError("Role must be User, Power User, or Administrator")
    now = _stamp(_now())
    with connect("server") as db:
        db.execute(
            """INSERT INTO auth_users
               (userid, password_hash, role, active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (userid, hash_password(password), role, True, now, now),
        )
    return {"userid": userid, "role": role, "active": True, "created_at": now, "updated_at": now}


def delete_user(userid: str, requesting_userid: str) -> bool:
    """Delete a user account while preserving at least one active administrator."""
    userid = userid.strip()
    if userid == requesting_userid:
        raise ValueError("You cannot delete your own account.")

    with connect("server") as db:
        row = db.execute(
            "SELECT role, active FROM auth_users WHERE userid = ?", (userid,)
        ).fetchone()
        if row is None:
            return False

        if row["role"] == "Administrator" and row["active"]:
            count = db.execute(
                "SELECT COUNT(*) AS count FROM auth_users "
                "WHERE role = ? AND active = ?",
                ("Administrator", True),
            ).fetchone()["count"]
            if count <= 1:
                raise ValueError("Cannot delete the last active Administrator.")

        db.execute("DELETE FROM auth_users WHERE userid = ?", (userid,))
    return True
