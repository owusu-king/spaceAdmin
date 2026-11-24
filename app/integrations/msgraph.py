"""Microsoft 365 connection (Microsoft Graph), read-only: outage email and record files.

Sign-in
  device  (default) James signs in himself: the app shows a short code, he enters it at microsoft.com/devicelogin
          and approves. The app never sees his password; only the token is stored (data/graph_token.json,
          git-ignored) and refreshed automatically.
  app     unattended service: an app registration with a client secret and *application* permission Mail.Read,
          which Rice IT restricts to one mailbox with an application access policy.

Read-only by construction: GraphClient has a GET method only. Delegated scopes are Mail.Read and Mail.Read.Shared
(and Files.Read.All for the space and people record files); nothing can send, move, flag or delete.

Configuration (.env)
  FAI_GRAPH_TENANT_ID     Rice's Azure AD tenant (or "organizations")
  FAI_GRAPH_CLIENT_ID     the app registration's Application (client) ID
  FAI_GRAPH_AUTH          device | app
  FAI_GRAPH_CLIENT_SECRET only for FAI_GRAPH_AUTH=app
  FAI_GRAPH_MAILBOX       "me" (the signed-in mailbox) or a shared mailbox address, e.g. facilities-outages@rice.edu
  FAI_GRAPH_FOLDER        folder to read: Inbox (default) or a folder name such as "Outages"
  FAI_GRAPH_SINCE_DAYS    how far back the first sync goes (default 14)
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from app.config import DATA_DIR, ENV_FILE  # noqa: F401  importing config loads .env
from app.logging_setup import get_logger, kv

log = get_logger("integrations.msgraph")
GRAPH = "https://graph.microsoft.com/v1.0"
LOGIN = "https://login.microsoftonline.com"
TOKEN_FILE = DATA_DIR / "graph_token.json"
DELEGATED_SCOPES = "offline_access User.Read Mail.Read Mail.Read.Shared Files.Read.All"


class GraphError(RuntimeError):
    pass


def graph_env() -> dict:
    e = os.environ
    return {"tenant": (e.get("FAI_GRAPH_TENANT_ID") or "").strip(), "client_id": (e.get("FAI_GRAPH_CLIENT_ID") or "").strip(),
            "auth": (e.get("FAI_GRAPH_AUTH") or "device").strip().lower(), "secret": e.get("FAI_GRAPH_CLIENT_SECRET") or "",
            "mailbox": (e.get("FAI_GRAPH_MAILBOX") or "me").strip(), "folder": (e.get("FAI_GRAPH_FOLDER") or "Inbox").strip(),
            "since_days": int(e.get("FAI_GRAPH_SINCE_DAYS") or 14)}


# ------------------------------------------------------------------------------ sign-in
class GraphAuth:
    """Tokens for Microsoft Graph. http: an httpx.Client (tests pass one with a mock transport)."""

    def __init__(self, env: dict | None = None, token_file: Path = TOKEN_FILE, http: httpx.Client | None = None) -> None:
        self.env = env or graph_env()
        self.token_file, self.http = token_file, http or httpx.Client(timeout=30)
        self._lock = threading.Lock()
        self._tok: dict = json.loads(token_file.read_text(encoding="utf-8")) if token_file.exists() else {}
        self.pending: dict | None = None            # device-code sign-in in progress
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        e = self.env
        return bool(e["tenant"] and e["client_id"] and (e["auth"] != "app" or e["secret"]))

    @property
    def connected(self) -> bool:
        return bool(self._tok.get("refresh_token") or (self._tok.get("access_token") and self.env["auth"] == "app"))

    def status(self) -> dict:
        if not self.configured:
            return {"status": "not_configured", "detail": "Set FAI_GRAPH_TENANT_ID and FAI_GRAPH_CLIENT_ID in .env"}
        if self.pending and time.time() < self.pending["expires_at"]:
            return {"status": "signing_in", "user_code": self.pending["user_code"],
                    "verification_uri": self.pending["verification_uri"], "expires_in": int(self.pending["expires_at"] - time.time())}
        if self.connected:
            return {"status": "connected", "account": self._tok.get("account"), "mode": self.env["auth"],
                    "mailbox": self.env["mailbox"], "folder": self.env["folder"]}
        return {"status": "not_connected", "mode": self.env["auth"], "error": self.last_error}

    def _token_url(self) -> str:
        return f"{LOGIN}/{self.env['tenant']}/oauth2/v2.0/token"

    def _save(self, tok: dict) -> None:
        tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 120
        keep = {k: tok.get(k) or self._tok.get(k) for k in ("access_token", "refresh_token", "expires_at", "account")}
        self._tok = keep
        self.token_file.parent.mkdir(exist_ok=True)
        self.token_file.write_text(json.dumps(keep), encoding="utf-8")

    # ---- device code (James signs in himself)
    def start_device_flow(self) -> dict:
        if not self.configured:
            raise GraphError("Microsoft 365 is not configured (FAI_GRAPH_TENANT_ID, FAI_GRAPH_CLIENT_ID)")
        r = self.http.post(f"{LOGIN}/{self.env['tenant']}/oauth2/v2.0/devicecode",
                           data={"client_id": self.env["client_id"], "scope": DELEGATED_SCOPES})
        if r.status_code != 200:
            raise GraphError(f"Microsoft sign-in could not start: {r.json().get('error_description', r.text)[:200]}")
        d = r.json()
        self.pending = {"device_code": d["device_code"], "user_code": d["user_code"], "verification_uri": d["verification_uri"],
                        "interval": int(d.get("interval", 5)), "expires_at": time.time() + int(d.get("expires_in", 900))}
        threading.Thread(target=self._poll_device, daemon=True, name="graph-signin").start()
        log.info("microsoft sign-in started (device code)")
        return self.status()

    def _poll_device(self) -> None:
        p = self.pending
        while p and time.time() < p["expires_at"] and self.pending is p:
            time.sleep(p["interval"])
            r = self.http.post(self._token_url(), data={"grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                                                        "client_id": self.env["client_id"], "device_code": p["device_code"]})
            d = r.json()
            if r.status_code == 200:
                with self._lock:
                    self._save(d)
                    self.pending, self.last_error = None, None
                try:
                    self._tok["account"] = self.get("/me", params={"$select": "userPrincipalName"}).get("userPrincipalName")
                    self._save({**self._tok, "expires_in": int(self._tok["expires_at"] - time.time() + 120)})
                except Exception:
                    pass
                log.info("microsoft sign-in complete " + kv(account=self._tok.get("account")))
                return
            if d.get("error") == "slow_down":
                p["interval"] += 5
            elif d.get("error") != "authorization_pending":
                self.last_error = d.get("error_description", d.get("error", "sign-in failed"))[:200]
                self.pending = None
                log.warning(f"microsoft sign-in failed error={d.get('error')}")
                return
        if self.pending is p:
            self.pending, self.last_error = None, "The sign-in code expired. Start again."

    def sign_out(self) -> None:
        with self._lock:
            self._tok, self.pending = {}, None
            self.token_file.unlink(missing_ok=True)
        log.info("microsoft signed out")

    # ---- tokens
    def token(self) -> str:
        with self._lock:
            if self._tok.get("access_token") and time.time() < self._tok.get("expires_at", 0):
                return self._tok["access_token"]
            if self.env["auth"] == "app":
                data = {"grant_type": "client_credentials", "client_id": self.env["client_id"],
                        "client_secret": self.env["secret"], "scope": "https://graph.microsoft.com/.default"}
            elif self._tok.get("refresh_token"):
                data = {"grant_type": "refresh_token", "client_id": self.env["client_id"],
                        "refresh_token": self._tok["refresh_token"], "scope": DELEGATED_SCOPES}
            else:
                raise GraphError("Not connected to Microsoft 365. Connect in Settings.")
            r = self.http.post(self._token_url(), data=data)
            if r.status_code != 200:
                self.last_error = r.json().get("error_description", r.text)[:200]
                raise GraphError(f"Microsoft 365 token refresh failed: {self.last_error}")
            self._save(r.json())
            return self._tok["access_token"]

    # ---- the only kind of request this module makes
    def get(self, path: str, params: dict | None = None, headers: dict | None = None, raw: bool = False):
        url = path if path.startswith("http") else GRAPH + path
        r = self.http.get(url, params=params, headers={"Authorization": f"Bearer {self.token()}", **(headers or {})},
                          follow_redirects=True)
        if r.status_code >= 400:
            raise GraphError(f"Graph {r.status_code}: {r.text[:200]}")
        return r.content if raw else r.json()


# ------------------------------------------------------------------------------ mail
class GraphMailSource:
    """Reads one mailbox folder with Graph's delta query: each fetch returns only mail that is new since the last one.
    Read-only: GET requests only; messages are never marked read, moved or deleted."""
    mode = "graph"

    def __init__(self, auth: GraphAuth, state_file: Path = DATA_DIR / "graph_mail_state.json") -> None:
        self.auth, self.state_file = auth, state_file
        self.state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}

    def _base(self) -> str:
        mb = self.auth.env["mailbox"]
        return "/me" if mb.lower() == "me" else f"/users/{mb}"

    def _folder_id(self) -> str:
        name = self.auth.env["folder"]
        if name.lower() in ("inbox", "archive", "junkemail", "sentitems"):
            return name.lower()
        if self.state.get("folder_name") == name and self.state.get("folder_id"):
            return self.state["folder_id"]
        hits = self.auth.get(f"{self._base()}/mailFolders", params={"$filter": f"displayName eq '{name}'", "$select": "id"})["value"]
        if not hits:
            raise GraphError(f"Mail folder '{name}' not found")
        self.state.update(folder_name=name, folder_id=hits[0]["id"])
        return hits[0]["id"]

    def health_check(self) -> dict:
        st = self.auth.status()
        if st["status"] != "connected":
            return {"mode": self.mode, **st}
        return {"status": "ok", "mode": self.mode, "account": st.get("account"), "mailbox": self.auth.env["mailbox"],
                "folder": self.auth.env["folder"], "last_sync": self.state.get("last_sync")}

    def fetch(self, limit: int = 25) -> list:
        from app.integrations.email import InboundEmail
        if self.auth.status()["status"] != "connected":
            return []
        link = self.state.get("next_link") or self.state.get("delta_link")
        params = None
        if not link:                                   # first sync: the last N days of this folder
            since = (datetime.now(timezone.utc) - timedelta(days=self.auth.env["since_days"])).strftime("%Y-%m-%dT%H:%M:%SZ")
            link = f"{self._base()}/mailFolders/{self._folder_id()}/messages/delta"
            params = {"$select": "id,internetMessageId,subject,from,receivedDateTime,body",
                      "$filter": f"receivedDateTime ge {since}"}
        out = []
        while link and len(out) < limit:
            page = self.auth.get(link, params=params, headers={"Prefer": 'outlook.body-content-type="text", odata.maxpagesize=25'})
            params = None
            for m in page.get("value", []):
                if "@removed" in m:
                    continue
                sender = (m.get("from") or {}).get("emailAddress") or {}
                received = m.get("receivedDateTime")
                date = datetime.fromisoformat(received.replace("Z", "+00:00")).astimezone().replace(tzinfo=None) if received else None
                out.append(InboundEmail(m.get("internetMessageId") or m["id"], f"{sender.get('name', '')} <{sender.get('address', '')}>",
                                        m.get("subject") or "", date, ((m.get("body") or {}).get("content") or "").strip()))
            if page.get("@odata.nextLink"):
                link = page["@odata.nextLink"]
                self.state["next_link"] = link             # resume here next time if we stop early
            else:
                self.state.pop("next_link", None)
                self.state["delta_link"] = page.get("@odata.deltaLink", self.state.get("delta_link"))
                link = None
        self.state["last_sync"] = datetime.now().isoformat(timespec="seconds")
        self.state_file.write_text(json.dumps(self.state), encoding="utf-8")
        log.info("graph mail sync " + kv(messages=len(out), more=bool(self.state.get("next_link"))))
        return sorted(out, key=lambda e: e.date or datetime.min)

    def execute(self, action) -> None:
        from app.integrations.email import ExecutionNotPermitted
        raise ExecutionNotPermitted("The Microsoft 365 reader is read-only")


# ------------------------------------------------------------------------------ files (SharePoint / OneDrive)
def share_id(url: str) -> str:
    """Graph 'shares' id for a SharePoint or OneDrive link: u! + unpadded base64url of the URL."""
    import base64
    return "u!" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


class GraphFile:
    """One file in SharePoint or OneDrive, addressed by its sharing link or web URL. version() is cheap (metadata);
    download() fetches the content only when it changed."""

    def __init__(self, auth: GraphAuth, url: str) -> None:
        self.auth, self.url = auth, url

    def version(self) -> dict:
        d = self.auth.get(f"/shares/{share_id(self.url)}/driveItem", params={"$select": "name,eTag,lastModifiedDateTime,size"})
        return {"name": d.get("name"), "etag": d.get("eTag"), "modified": d.get("lastModifiedDateTime"), "size": d.get("size")}

    def download(self) -> bytes:
        return self.auth.get(f"/shares/{share_id(self.url)}/driveItem/content", raw=True)
