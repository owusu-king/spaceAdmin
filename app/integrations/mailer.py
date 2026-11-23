"""Outgoing email for drafts a person has approved (integration Level 4, behind a switch).

Only the API's send endpoint uses this module, after it has checked the 'send_email' switch,
the case approval and the revision. Agents have no access to it.

Configuration (.env):
  FAI_SMTP_HOST, FAI_SMTP_PORT (587), FAI_SMTP_SECURITY (starttls | ssl | none),
  FAI_SMTP_USER, FAI_SMTP_PASSWORD, FAI_SMTP_FROM
  FAI_EMAIL_REDIRECT_TO   test address that receives every email while
                          'Deliver to real recipients' is off

Note: Microsoft 365 usually disables password (SMTP AUTH) sign-in. For a Rice mailbox a
Microsoft Graph sender with the same `send` interface would replace SmtpMailer.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from email.utils import make_msgid

from app.config import ENV_FILE  # noqa: F401  importing config loads .env
from app.logging_setup import get_logger

log = get_logger("integrations.mailer")


@dataclass
class OutgoingEmail:
    subject: str
    body: str
    to: list[str]
    bcc: list[str] = field(default_factory=list)


class SmtpMailer:
    mode = "smtp"

    def __init__(self, env: dict | None = None, client_factory=None) -> None:
        e = env if env is not None else os.environ
        self.host = (e.get("FAI_SMTP_HOST") or "").strip()
        self.port = int(e.get("FAI_SMTP_PORT") or 587)
        self.security = (e.get("FAI_SMTP_SECURITY") or "starttls").strip().lower()
        self.user = (e.get("FAI_SMTP_USER") or "").strip()
        self._password = e.get("FAI_SMTP_PASSWORD") or ""
        self.sender = (e.get("FAI_SMTP_FROM") or self.user).strip()
        self.redirect_to = (e.get("FAI_EMAIL_REDIRECT_TO") or "").strip()
        self._factory = client_factory

    @property
    def configured(self) -> bool:
        return bool(self.host and self.sender)

    def readiness(self) -> tuple[bool, str]:
        if not self.configured:
            return False, "Outgoing mail is not configured (set FAI_SMTP_HOST and FAI_SMTP_FROM in .env)"
        return True, ""

    def describe(self) -> dict:
        """Settings shown in the UI. Never includes the password."""
        return {"mode": self.mode, "configured": self.configured, "host": self.host or None, "port": self.port,
                "security": self.security, "from": self.sender or None, "redirect_to": self.redirect_to or None}

    def _client(self):
        if self._factory:
            return self._factory(self.host, self.port)
        if self.security == "ssl":
            return smtplib.SMTP_SSL(self.host, self.port, context=ssl.create_default_context(), timeout=30)
        client = smtplib.SMTP(self.host, self.port, timeout=30)
        if self.security == "starttls":
            client.starttls(context=ssl.create_default_context())
        return client

    def send(self, mail: OutgoingEmail) -> str:
        """Send one message. Returns its Message-ID. Raises on any delivery error."""
        ok, why = self.readiness()
        if not ok:
            raise RuntimeError(why)
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, ", ".join(mail.to), mail.subject
        msg["Message-ID"] = make_msgid(domain=self.sender.split("@")[-1] or None)
        msg["Date"] = datetime.now().astimezone().strftime("%a, %d %b %Y %H:%M:%S %z")
        msg.set_content(mail.body)
        client = self._client()
        try:
            if self.user and self._password:
                client.login(self.user, self._password)
            client.send_message(msg, to_addrs=list(dict.fromkeys(mail.to + mail.bcc)))   # BCC never in headers
        finally:
            client.quit()
        log.info(f"email sent to={len(mail.to)} bcc={len(mail.bcc)} id={msg['Message-ID']}")
        return msg["Message-ID"]
