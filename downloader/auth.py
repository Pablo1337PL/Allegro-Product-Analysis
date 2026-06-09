import os
import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

_token: str | None = None
_token_expires_at: float = 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)
def _fetch_token(client_id: str, client_secret: str) -> tuple[str, float]:
    auth_url = os.environ["ALLEGRO_AUTH_URL"]
    with httpx.Client() as client:
        resp = client.post(
            auth_url,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=15.0,
        )
        resp.raise_for_status()
    data = resp.json()
    token = data["access_token"]
    expires_in = float(data.get("expires_in", 3600))
    return token, time.monotonic() + expires_in - 60


def get_token() -> str:
    global _token, _token_expires_at
    if _token is None or time.monotonic() >= _token_expires_at:
        client_id = os.environ["ALLEGRO_CLIENT_ID"]
        client_secret = os.environ["ALLEGRO_CLIENT_SECRET"]
        _token, _token_expires_at = _fetch_token(client_id, client_secret)
    return _token


def invalidate_token() -> None:
    global _token, _token_expires_at
    _token = None
    _token_expires_at = 0.0
