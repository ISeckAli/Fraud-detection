# Model Card: Fraud Detection Classifier

## Model Details

- **Model type:** Logistic Regression (`scikit-learn`), `class_weight="balanced"`
- **Version:** 1.0
- **Training date:** September 2026
- **License:** MIT (this repository); dataset licensed CC0

## Intended Use

**Primary use case:** demonstrating a complete, honestly-evaluated fraud detection pipeline, from raw data through a deployed API, as a portfolio and learning artifact. The interactive demo lets a non-technical visitor submit a sample transaction and see a real probability score with the decision threshold applied.

**Out of scope:** this model is not intended for production financial fraud detection. See Limitations below.

## Training Data

- **Source:** "Credit Card Fraud Detection 2026" (Udit Jain, Kaggle), CC0 license
- **Size:** 20,000 synthetic transactions, 26 features
- **Class balance:** 1.69% fraud (339 fraud / 19,661 non-fraud)
- **Split:** 80/20 train/test, stratified to preserve class ratio in both sets
- **Features:** 24 input features after dropping the transaction identifier, covering transaction context (amount, merchant category, channel), behavioral signals (velocity score, transaction count in last 24h, CVV retry count), and risk indicators (IP country mismatch, VPN use, merchant risk score)

## Preprocessing

1. One-hot encoding of 5 categorical columns (`merchant_category`, `card_type`, `auth_method`, `channel`, `device_type`), expanding 24 features to 49
2. `transaction_id` dropped (identifier, not a feature)
3. `StandardScaler` fit on the training set only, applied to both splits
4. Class imbalance handled via `class_weight="balanced"` at training time, **not** synthetic oversampling (see Notes below)

## Evaluation

Evaluated on the untouched, real-imbalance test set (20% held out, never used in training or resampling).

| Metric | Value |
|---|---|
| Precision (fraud class) | 0.25 |
| Recall (fraud class) | 0.68 |
| F1 Score (fraud class) | 0.37 |
| Decision threshold | 0.85 |

**Comparison models evaluated:** Random Forest (F1 0.14) and XGBoost (F1 0.27) were also trained and evaluated on identical data splits. Logistic Regression outperformed both on this dataset.

## Threshold Rationale

The default 0.5 threshold was not used. **0.85** was selected to prioritize recall over precision, based on the business reasoning that a missed fraud case (false negative) causes direct financial loss, while a false alarm (false positive) causes a legitimate customer a recoverable inconvenience (e.g., additional verification). At this threshold, the model catches 68% of fraud cases in the test set, versus 43% at the default threshold, at the cost of a higher false-positive rate.

## Interpretability

SHAP (SHapley Additive exPlanations) values were computed on a 200-transaction sample of the test set. The dominant features are `merchant_risk_score`, `velocity_score`, and `cvv_retry_count`, consistent with the dataset's intended interpretable design and with independent correlation analysis performed during exploratory data analysis.

## Known Issue Investigated and Resolved

An earlier version of this model was trained using standard SMOTE oversampling applied after categorical encoding. This produced a data corruption bug: SMOTE's numerical interpolation generated fractional values for columns that should only ever be strictly 0 or 1, which were then silently cast to boolean `True`, systematically distorting the synthetic training data. This was caught via a SHAP result that contradicted independent correlation evidence, investigated, confirmed directly, and corrected by switching to class-weighted training (no synthetic data). Full details are in the project README.

## Limitations

- **Scale:** 20,000 transactions is small relative to real-world enterprise fraud volumes.
- **Synthetic data:** this dataset does not reflect evolving, adversarial real-world fraud patterns, and was generated for educational/demonstration purposes.
- **Precision at the chosen threshold is low (0.25):** roughly 3 in 4 flagged transactions are false alarms. This is a deliberate tradeoff favoring recall, documented above, but represents a real customer-friction cost in any actual deployment.
- **No temporal validation:** the train/test split is random (stratified), not time-based. A production fraud model would typically be validated on a strict future time window to detect concept drift.
- **Not adversarially tested:** this model has not been evaluated against adversarial examples or evolving fraud tactics designed specifically to evade it.

## Ethical Considerations

A false positive from this model (a legitimate transaction flagged as fraud) creates friction for a real customer. A false negative (missed fraud) creates direct financial harm. Any real deployment should treat the threshold as a business decision requiring input beyond a single engineering choice, and should monitor for disparate false-positive rates across customer segments, which this project's synthetic dataset does not contain the demographic detail to evaluate.