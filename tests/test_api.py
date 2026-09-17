# Tests for the fraud detection Flask API.
#
# These tests import the actual Flask app and use its built-in test
# client, which simulates HTTP requests without needing a real running
# server - fast, and doesn't require starting app.py separately.

import sys
import os

# Add the api/ folder to Python's search path, so we can import app.py
# from here in tests/, even though they're in different folders.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from app import app


def test_predict_rejects_missing_fields():
    """
    Sending incomplete transaction data should return a 400 error
    with a clear message about which fields are missing - not crash,
    and not silently accept bad data.
    """
    client = app.test_client()
    response = client.post("/predict", json={"amount_usd": 100.0})

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "missing_fields" in data


def test_predict_returns_valid_response_for_good_input():
    """
    Sending a complete, valid transaction should return a 200 status
    and a response containing fraud_probability, is_fraud, and
    threshold_used - the three fields the demo interface depends on.
    """
    client = app.test_client()

    valid_transaction = {
        "amount_usd": 150.0, "merchant_category": "Electronics", "card_type": "Visa",
        "auth_method": "PIN", "channel": "Online", "device_type": "iPhone",
        "is_foreign_transaction": False, "hours_since_last_txn": 5.0,
        "txn_count_last_24h": 2, "distance_from_home_km": 10.0,
        "card_age_months": 24, "customer_age": 35, "account_balance_usd": 5000.0,
        "is_new_merchant": False, "used_vpn": False, "ip_country_mismatch": False,
        "billing_shipping_mismatch": False, "cvv_retry_count": 0,
        "velocity_score": 15.0, "time_of_day_hour": 14, "day_of_week": 3,
        "is_ai_generated_scam_attempt": False, "merchant_risk_score": 20.0,
        "prior_disputes": 0
    }

    response = client.post("/predict", json=valid_transaction)

    assert response.status_code == 200
    data = response.get_json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "threshold_used" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_flags_obvious_fraud_correctly():
    """
    A transaction loaded with every known fraud signal (high CVV
    retries, VPN, country mismatch, high risk score, high velocity,
    new merchant, foreign transaction) should be flagged as fraud.
    This confirms the model's real predictive behavior, not just that
    the API responds without crashing.
    """
    client = app.test_client()

    suspicious_transaction = {
        "amount_usd": 2500.0, "merchant_category": "Electronics", "card_type": "Visa",
        "auth_method": "No Authentication", "channel": "Online", "device_type": "Windows PC",
        "is_foreign_transaction": True, "hours_since_last_txn": 0.1,
        "txn_count_last_24h": 15, "distance_from_home_km": 500.0,
        "card_age_months": 2, "customer_age": 35, "account_balance_usd": 200.0,
        "is_new_merchant": True, "used_vpn": True, "ip_country_mismatch": True,
        "billing_shipping_mismatch": True, "cvv_retry_count": 3,
        "velocity_score": 65.0, "time_of_day_hour": 3, "day_of_week": 3,
        "is_ai_generated_scam_attempt": True, "merchant_risk_score": 90.0,
        "prior_disputes": 2
    }

    response = client.post("/predict", json=suspicious_transaction)

    assert response.status_code == 200
    data = response.get_json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.85