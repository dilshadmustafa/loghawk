"""Authentication endpoints shared by the Web UI and Hawk Console."""

from fastapi import APIRouter, Depends, Header, HTTPException

from loghawk.api.schemas import LoginRequest, PasswordChange, UserCreate
from loghawk.api.services import auth_service
from loghawk.database.database_layer import DatabaseIntegrityError


router = APIRouter()


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Please sign in.")
    return authorization[7:].strip()


def current_user(authorization: str | None = Header(default=None), client: str | None = None) -> dict:
    user = auth_service.user_for_token(_bearer(authorization), client)
    if user is None:
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    return user


def require_webui_user(authorization: str | None = Header(default=None)) -> dict:
    return current_user(authorization, "webui")


def require_console_user(authorization: str | None = Header(default=None)) -> dict:
    return current_user(authorization, "console")


def require_authenticated_user(authorization: str | None = Header(default=None)) -> dict:
    return current_user(authorization)


def require_admin(user: dict = Depends(require_webui_user)) -> dict:
    if user["role"] != "Administrator":
        raise HTTPException(status_code=403, detail="Administrator role required.")
    return user


@router.get("/status")
def auth_status() -> dict:
    return {"webui_login": True, "console_login": True}


@router.post("/login")
def login(body: LoginRequest) -> dict:
    result = auth_service.authenticate(body.userid, body.password, body.client)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid user ID or password.")
    return result


@router.post("/logout", status_code=204)
def logout(
    authorization: str | None = Header(default=None),
    _user: dict = Depends(current_user),
) -> None:
    auth_service.revoke_token(_bearer(authorization))


@router.get("/me")
def me(user: dict = Depends(require_webui_user)) -> dict:
    return user


@router.post("/change-password")
def change_password(body: PasswordChange, user: dict = Depends(require_webui_user)) -> dict:
    if not auth_service.change_password(
        user["userid"], body.current_password, body.new_password
    ):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    return {"status": "password_changed"}


@router.get("/users")
def list_users(_admin: dict = Depends(require_admin)) -> list[dict]:
    return auth_service.list_users()


@router.post("/users", status_code=201)
def create_user(body: UserCreate, _admin: dict = Depends(require_admin)) -> dict:
    try:
        return auth_service.create_user(body.userid, body.password, body.role)
    except DatabaseIntegrityError as exc:
        raise HTTPException(status_code=409, detail="That user ID already exists.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/users/{userid}", status_code=204)
def delete_user(userid: str, admin: dict = Depends(require_admin)) -> None:
    try:
        deleted = auth_service.delete_user(userid, admin["userid"])
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found.")
