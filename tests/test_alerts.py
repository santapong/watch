"""Tests for alert and notification system."""

import json
import time

import pytest

from src.alerts import Alert, AlertRule, AlertManager, LogNotifier


class TestAlert:
    def test_creation(self):
        alert = Alert(alert_type="test", message="Test alert")
        assert alert.alert_type == "test"
        assert alert.message == "Test alert"
        assert alert.severity == "info"
        assert isinstance(alert.timestamp, float)

    def test_to_dict(self):
        alert = Alert(alert_type="anomaly", message="Anomaly detected", severity="critical")
        d = alert.to_dict()
        assert d["alert_type"] == "anomaly"
        assert d["severity"] == "critical"
        assert "timestamp" in d


class TestAlertRule:
    def test_triggers_when_condition_met(self):
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: ctx.get("count", 0) > 5,
            alert_type="test",
            message="Count exceeded",
            cooldown=0,
        )
        alert = rule.check({"count": 10})
        assert alert is not None
        assert alert.alert_type == "test"

    def test_no_trigger_when_condition_not_met(self):
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: ctx.get("count", 0) > 5,
            alert_type="test",
            message="Count exceeded",
        )
        alert = rule.check({"count": 3})
        assert alert is None

    def test_cooldown_prevents_repeat(self):
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: True,
            alert_type="test",
            message="Always triggers",
            cooldown=10.0,
        )
        alert1 = rule.check({})
        assert alert1 is not None
        alert2 = rule.check({})
        assert alert2 is None  # Cooldown not elapsed

    def test_no_cooldown(self):
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: True,
            alert_type="test",
            message="Always triggers",
            cooldown=0,
        )
        alert1 = rule.check({})
        assert alert1 is not None
        alert2 = rule.check({})
        assert alert2 is not None


class TestAlertManager:
    def test_add_rule_and_evaluate(self):
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="crowd",
            condition=lambda ctx: ctx.get("person_count", 0) > 10,
            alert_type="crowd",
            message="Crowd detected",
            cooldown=0,
        ))
        alerts = manager.evaluate({"person_count": 15})
        assert len(alerts) == 1
        assert alerts[0].alert_type == "crowd"

    def test_no_alerts_when_conditions_not_met(self):
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="crowd",
            condition=lambda ctx: ctx.get("person_count", 0) > 10,
            alert_type="crowd",
            message="Crowd detected",
        ))
        alerts = manager.evaluate({"person_count": 5})
        assert len(alerts) == 0

    def test_history(self):
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="test",
            condition=lambda ctx: True,
            alert_type="test",
            message="Test",
            cooldown=0,
        ))
        manager.evaluate({})
        assert len(manager.history) == 1

    def test_clear_history(self):
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="test",
            condition=lambda ctx: True,
            alert_type="test",
            message="Test",
            cooldown=0,
        ))
        manager.evaluate({})
        manager.clear_history()
        assert len(manager.history) == 0

    def test_multiple_rules(self):
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="rule1",
            condition=lambda ctx: ctx.get("a") is True,
            alert_type="type1",
            message="A",
            cooldown=0,
        ))
        manager.add_rule(AlertRule(
            name="rule2",
            condition=lambda ctx: ctx.get("b") is True,
            alert_type="type2",
            message="B",
            cooldown=0,
        ))
        alerts = manager.evaluate({"a": True, "b": True})
        assert len(alerts) == 2


class TestLogNotifier:
    def test_log_notifier(self, tmp_path):
        path = str(tmp_path / "test_alerts.json")
        notifier = LogNotifier(log_path=path)
        alert = Alert(alert_type="test", message="Hello")
        assert notifier.send(alert) is True

        data = json.loads(open(path).read())
        assert len(data) == 1
        assert data[0]["message"] == "Hello"

    def test_log_appends(self, tmp_path):
        path = str(tmp_path / "test_alerts.json")
        notifier = LogNotifier(log_path=path)
        notifier.send(Alert(alert_type="a", message="First"))
        notifier.send(Alert(alert_type="b", message="Second"))

        data = json.loads(open(path).read())
        assert len(data) == 2
