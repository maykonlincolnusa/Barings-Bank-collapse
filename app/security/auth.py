from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.configs.settings import settings


USERS = {
    settings.admin_username: {"password": settings.admin_password, "role": "admin"},
    settings.analyst_username: {"password": settings.analyst_password, "role": "analyst"},
}

http_bearer = HTTPBearer(auto_error=False)


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    user = USERS.get(username)
    if not user or user["password"] != password:
        return None
    return {"username": username, "role": user["role"]}


def create_access_token(user: dict[str, str]) -> str:
    payload = {
        "sub": user["username"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer)) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
        return {"username": payload["sub"], "role": payload["role"]}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def require_roles(*roles: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def dependency(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return dependency

