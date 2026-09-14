# Flask API for the fraud detection model.
#
# This file loads the trained model, scaler, feature column order, and
# decision threshold (all saved during the notebook work), then exposes
# a single endpoint: POST /predict, which accepts transaction data and
# returns a fraud probability and decision.

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import os

# Create the Flask application instance.
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing), so a webpage loaded
# from a different origin (e.g. a local HTML file, or later, the
# deployed demo site) is allowed to call this API. Without this,
# browsers block the request as a security precaution by default.
CORS(app)

# Set up rate limiting, to protect the public demo endpoint from being
# overwhelmed (e.g. by accidental loops or abuse) once it's deployed.
# get_remote_address identifies each caller by their IP address, so
# limits apply per-visitor, not globally across everyone.
limiter = Limiter(get_remote_address, app=app, default_limits=["50 per hour"])

# Load the model, scaler, feature columns, and threshold ONCE at
# startup - not on every request, which would be slow and wasteful.
# We build paths relative to this file's location, so the app works
# correctly no matter where it's run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "..", "models", "threshold.txt")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read().strip())

print(f"Model, scaler, and feature columns loaded. Threshold: {THRESHOLD}")

# The exact list of fields every valid request must include. Used to
# validate input BEFORE attempting any processing, so we can return a
# clear, specific error message rather than a confusing internal one.
REQUIRED_FIELDS = [
    "amount_usd", "merchant_category", "card_type", "auth_method", "channel",
    "device_type", "is_foreign_transaction", "hours_since_last_txn",
    "txn_count_last_24h", "distance_from_home_km", "card_age_months",
    "customer_age", "account_balance_usd", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week",
    "is_ai_generated_scam_attempt", "merchant_risk_score", "prior_disputes"
]


# Define the /predict endpoint.
# This only accepts POST requests, since the client is sending data
# (transaction details) for the server to process - not just requesting
# a page.
#
# @limiter.limit sets a stricter limit specifically for this endpoint
# (10 per minute), separate from the default app-wide limit set above.
@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict():
    try:
        # Get the JSON data sent by the client.
        input_data = request.get_json()

        if input_data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        # Check that every required field is present BEFORE attempting
        # any processing. This gives a clear, specific error message
        # instead of a confusing internal one from deeper in the code.
        missing_fields = [field for field in REQUIRED_FIELDS if field not in input_data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Convert the single transaction (a dictionary) into a
        # one-row DataFrame, matching the structure the model expects.
        input_df = pd.DataFrame([input_data])

        # One-hot encode the same way we did during training.
        categorical_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Ensure the input has EXACTLY the same columns, in the same
        # order, as the model was trained on. Any missing column
        # (e.g. a category not present in this single transaction)
        # gets filled with 0 - reindex handles both adding missing
        # columns and enforcing correct order in one step.
        input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Scale the input using the SAME fitted scaler from training.
        input_scaled = scaler.transform(input_final)

        # Get the fraud probability, and apply our chosen threshold.
        probability = model.predict_proba(input_scaled)[0][1]
        is_fraud = bool(probability >= THRESHOLD)

        return jsonify({
            "fraud_probability": round(float(probability), 4),
            "is_fraud": is_fraud,
            "threshold_used": THRESHOLD
        })

    except Exception as e:
        # Catch-all for any unexpected error, so the API returns a
        # clean JSON error response instead of crashing.
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 400


# Start the Flask development server when this file is run directly.
# debug=True gives helpful error pages during development - this will
# be turned off later for the production deployment (Part 14).
if __name__ == "__main__":
    app.run(debug=True, port=5000)