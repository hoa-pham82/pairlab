"""Read secrets from Vault KV v2 and inject them into os.environ.

Only activates when VAULT_ADDR and VAULT_TOKEN are set.
Silently no-ops when unconfigured; logs a warning when configured but a fetch fails.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


def load_secrets(paths: list[str]) -> None:
    """Fetch each KV path from Vault and set missing env vars.

    Args:
        paths: Vault KV paths, e.g. ["secret/data/minio", "secret/data/postgres"].
              Each key in the secret becomes an env var if not already set.
    """
    addr = os.environ.get("VAULT_ADDR")
    token = os.environ.get("VAULT_TOKEN")
    if not (addr and token):
        return  # Vault not configured — skip silently

    for path in paths:
        url = f"{addr}/v1/{path}"
        req = urllib.request.Request(url, headers={"X-Vault-Token": token})
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())["data"]["data"]
        except urllib.error.HTTPError as exc:
            log.warning("vault: failed to read %s — HTTP %s", path, exc.code)
            continue
        except urllib.error.URLError as exc:
            log.warning("vault: unreachable at %s — %s", addr, exc.reason)
            continue
        except (KeyError, json.JSONDecodeError) as exc:
            log.warning("vault: unexpected response for %s — %s", path, exc)
            continue

        for key, value in data.items():
            env_key = key.upper()
            if env_key not in os.environ:
                os.environ[env_key] = str(value)
