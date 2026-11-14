"""Small read-only client for FakeStore's public product catalog."""

from __future__ import annotations

import json
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import Config


class FakeStoreError(RuntimeError):
    """A network, HTTP, JSON, or schema error from FakeStore."""


class FakeStoreNotFound(FakeStoreError):
    """The requested product does not exist."""


class FakeStoreClient:
    def __init__(self, base_url: str, timeout: float = 4.0, opener=None):
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("FakeStore base URL must be HTTP(S).")
        if timeout <= 0:
            raise ValueError("FakeStore timeout must be positive.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener or urlopen

    def _get_json(self, path: str, params: dict | None = None):
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urlencode(params)
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "ecom-langgraph/1.0"})
        try:
            with self._opener(request, timeout=self.timeout) as response:
                if response.status < 200 or response.status >= 300:
                    raise FakeStoreError("FakeStore returned an unsuccessful response.")
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise FakeStoreNotFound("FakeStore product was not found.") from exc
            raise FakeStoreError("FakeStore request failed.") from exc
        except (URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FakeStoreError("FakeStore is unavailable or returned invalid JSON.") from exc

    @staticmethod
    def _validate_product(value):
        required = {"id": int, "title": str, "price": (int, float), "description": str, "category": str, "image": str}
        if not isinstance(value, dict) or any(
            isinstance(value.get(key), bool) or not isinstance(value.get(key), expected)
            for key, expected in required.items()
        ):
            raise FakeStoreError("FakeStore returned an invalid product record.")
        return {key: value[key] for key in required}

    def list_products(self, limit: int = 20) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Product list limit must be between 1 and 100.")
        payload = self._get_json("/products", {"limit": limit})
        if not isinstance(payload, list):
            raise FakeStoreError("FakeStore returned an invalid product list.")
        return [self._validate_product(item) for item in payload]

    def get_product(self, product_id: int) -> dict:
        if isinstance(product_id, bool) or not isinstance(product_id, int) or product_id < 1:
            raise ValueError("Product ID must be a positive integer.")
        return self._validate_product(self._get_json(f"/products/{product_id}"))


@lru_cache(maxsize=1)
def get_fakestore_client() -> FakeStoreClient:
    """Instantiate on first use, so normal application imports need no network."""
    return FakeStoreClient(Config.FAKESTORE_API_BASE_URL, Config.FAKESTORE_TIMEOUT_SECONDS)
