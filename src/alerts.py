"""Alert and notification system for event-driven monitoring.

Triggers alerts on specific events (anomaly detected, zone capacity exceeded,
loitering) and sends notifications via configurable channels.
"""

import json
import logging
import smtplib
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A triggered alert."""

    alert_type: str  # "anomaly", "zone_capacity", "loitering", "crowd", "custom"
    message: str
    severity: str = "info"  # "info", "warning", "critical"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AlertRule:
    """A configurable rule that triggers alerts based on conditions.

    Example:
        rule = AlertRule(
            name="crowd_alert",
            condition=lambda ctx: ctx.get("person_count", 0) > 20,
            alert_type="crowd",
            message="Crowd exceeded 20 persons",
            severity="warning",
            cooldown=60.0,
        )
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[dict], bool],
        alert_type: str,
        message: str,
        severity: str = "warning",
        cooldown: float = 60.0,
    ):
        """Initialize alert rule.

        Args:
            name: Unique rule name.
            condition: Function taking context dict, returns True to trigger.
            alert_type: Alert category string.
            message: Alert message text.
            severity: "info", "warning", or "critical".
            cooldown: Minimum seconds between repeated triggers.
        """
        self.name = name
        self.condition = condition
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.cooldown = cooldown
        self._last_triggered: float = 0.0

    def check(self, context: dict) -> Alert | None:
        """Evaluate the rule against current context.

        Args:
            context: Dict with current detection/analytics state.

        Returns:
            Alert if triggered, None otherwise.
        """
        now = time.time()
        if now - self._last_triggered < self.cooldown:
            return None

        if self.condition(context):
            self._last_triggered = now
            return Alert(
                alert_type=self.alert_type,
                message=self.message,
                severity=self.severity,
                metadata={"rule": self.name},
            )
        return None


class WebhookNotifier:
    """Sends alerts via HTTP webhook (Slack, Discord, custom endpoints)."""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self._url = url
        self._headers = headers or {"Content-Type": "application/json"}

    def send(self, alert: Alert) -> bool:
        try:
            payload = {
                "text": f"[{alert.severity.upper()}] {alert.message}",
                "alert": alert.to_dict(),
            }
            resp = requests.post(self._url, json=payload, headers=self._headers, timeout=10)
            return resp.status_code < 400
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class EmailNotifier:
    """Sends alerts via SMTP email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipients: list[str],
        username: str = "",
        password: str = "",
        use_tls: bool = True,
    ):
        self._host = smtp_host
        self._port = smtp_port
        self._sender = sender
        self._recipients = recipients
        self._username = username
        self._password = password
        self._use_tls = use_tls

    def send(self, alert: Alert) -> bool:
        try:
            msg = MIMEText(
                f"Alert Type: {alert.alert_type}\n"
                f"Severity: {alert.severity}\n"
                f"Message: {alert.message}\n"
                f"Timestamp: {time.ctime(alert.timestamp)}\n"
                f"Metadata: {json.dumps(alert.metadata, indent=2)}"
            )
            msg["Subject"] = f"[{alert.severity.upper()}] Detection Alert: {alert.alert_type}"
            msg["From"] = self._sender
            msg["To"] = ", ".join(self._recipients)

            with smtplib.SMTP(self._host, self._port) as server:
                if self._use_tls:
                    server.starttls()
                if self._username:
                    server.login(self._username, self._password)
                server.sendmail(self._sender, self._recipients, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False


class LogNotifier:
    """Logs alerts to a JSON file for local persistence."""

    def __init__(self, log_path: str = "alerts.json"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> bool:
        try:
            existing = []
            if self._path.exists():
                existing = json.loads(self._path.read_text())
            existing.append(alert.to_dict())
            self._path.write_text(json.dumps(existing, indent=2))
            return True
        except Exception as e:
            logger.error(f"Log notification failed: {e}")
            return False


class AlertManager:
    """Central manager for alert rules and notification dispatch.

    Example:
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="crowd",
            condition=lambda ctx: ctx.get("person_count", 0) > 20,
            alert_type="crowd",
            message="Crowd detected",
        ))
        manager.add_notifier(LogNotifier("alerts.json"))

        # In detection loop:
        context = {"person_count": 25, "anomaly_score": -0.3}
        alerts = manager.evaluate(context)
    """

    def __init__(self, max_history: int = 100):
        self._rules: list[AlertRule] = []
        self._notifiers: list = []
        self._history: deque[Alert] = deque(maxlen=max_history)

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def add_notifier(self, notifier) -> None:
        """Add a notifier (WebhookNotifier, EmailNotifier, LogNotifier, or any with send())."""
        self._notifiers.append(notifier)

    def evaluate(self, context: dict) -> list[Alert]:
        """Evaluate all rules against current context and dispatch notifications.

        Args:
            context: Dict with current state (e.g., person_count, anomaly_score).

        Returns:
            List of triggered alerts.
        """
        triggered = []
        for rule in self._rules:
            alert = rule.check(context)
            if alert is not None:
                triggered.append(alert)
                self._history.append(alert)

                for notifier in self._notifiers:
                    try:
                        notifier.send(alert)
                    except Exception as e:
                        logger.error(f"Notifier failed: {e}")

        return triggered

    @property
    def history(self) -> list[Alert]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
