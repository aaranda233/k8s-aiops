"""Tests del circuit breaker — prevención de bucles de remediación."""

import time

import pytest

from src.remediation.circuit_breaker import CircuitBreaker


@pytest.mark.unit
def test_fingerprint_is_deterministic():
    cb = CircuitBreaker()
    fp1 = cb.fingerprint({"prod", "staging"}, "OOMKilled en pod X")
    fp2 = cb.fingerprint({"staging", "prod"}, "OOMKilled en pod X")
    assert fp1 == fp2  # orden de namespaces no importa


@pytest.mark.unit
def test_different_anomalies_different_fingerprint():
    cb = CircuitBreaker()
    fp1 = cb.fingerprint({"prod"}, "OOMKilled")
    fp2 = cb.fingerprint({"prod"}, "ImagePullBackOff")
    assert fp1 != fp2


@pytest.mark.unit
def test_not_blocked_initially():
    cb = CircuitBreaker(max_attempts=3)
    fp = cb.fingerprint({"prod"}, "OOMKilled")
    blocked, attempts = cb.is_blocked(fp)
    assert blocked is False
    assert attempts == 0


@pytest.mark.unit
def test_blocks_after_max_attempts():
    cb = CircuitBreaker(max_attempts=3, window_seconds=600)
    fp = cb.fingerprint({"prod"}, "OOMKilled")
    for _ in range(3):
        cb.record(fp, "kubectl rollout restart deployment/x", success=False)
    blocked, attempts = cb.is_blocked(fp)
    assert blocked is True
    assert attempts == 3


@pytest.mark.unit
def test_not_blocked_below_threshold():
    cb = CircuitBreaker(max_attempts=3)
    fp = cb.fingerprint({"prod"}, "OOMKilled")
    cb.record(fp, "cmd", success=False)
    cb.record(fp, "cmd", success=False)
    blocked, attempts = cb.is_blocked(fp)
    assert blocked is False
    assert attempts == 2


@pytest.mark.unit
def test_reset_clears_history():
    cb = CircuitBreaker(max_attempts=3)
    fp = cb.fingerprint({"prod"}, "OOMKilled")
    for _ in range(3):
        cb.record(fp, "cmd", success=False)
    assert cb.is_blocked(fp)[0] is True
    cb.reset(fp)
    assert cb.is_blocked(fp) == (False, 0)


@pytest.mark.unit
def test_old_attempts_purged_outside_window():
    cb = CircuitBreaker(max_attempts=3, window_seconds=1)
    fp = cb.fingerprint({"prod"}, "OOMKilled")
    for _ in range(3):
        cb.record(fp, "cmd", success=False)
    assert cb.is_blocked(fp)[0] is True
    time.sleep(1.1)  # esperar a que expire la ventana
    blocked, attempts = cb.is_blocked(fp)
    assert blocked is False
    assert attempts == 0
