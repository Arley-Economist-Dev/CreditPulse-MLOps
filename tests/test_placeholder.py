"""Smoke test to verify test harness and package importability."""

from credit_risk_service import __version__


def test_package_version():
    assert __version__ == "0.1.0"
