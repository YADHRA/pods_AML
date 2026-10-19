@'
# AML Model Comparison: Linear SVM vs XGBoost

## 1. Objective

This experiment compares a linear SVM baseline with two XGBoost models for anti-money laundering (AML) detection. The goal is to evaluate how the model choice and additional account-level and graph-derived features affect the identification of suspicious transactions.

## 2. Dataset and Evaluation Setup

The experiments use the IBM AML HI-Small transaction dataset.

- Total transactions: approximately 5.08 million
- Total laundering-positive transactions: 5,177
- Warm-up period: September 1, 2022
- Training period: September 2–6, 2022
- Validation period: September 7–8, 2022
- Test period: September 9–10, 2022
- Test transactions: 862,792
- Test laundering-positive transactions: 956
- Test positive rate: approximately 0.111%

The temporal split evaluates models on transactions occurring after the training and validation periods. The test set is highly imbalanced, making Precision-Recall AUC (PR-AUC) the primary evaluation metric.

## 3. Models Compared

### Model 1: Linear SVM M1

The SVM baseline uses eight transaction-level features. It was implemented using `SGDClassifier` with hinge loss, which provides a scalable linear SVM-style classifier.

The selected configuration used alpha = 0.0001 and balanced class weights. The configuration was selected using validation PR-AUC.

### Model 2: XGBoost M1

XGBoost M1 uses the same eight transaction-level features as the SVM. This comparison helps evaluate the difference between a linear classifier and a nonlinear tree-based model using the same feature set.

### Model 3: XGBoost M2

XGBoost M2 extends the transaction-level features with account-level and graph-derived features. It evaluates whether information about accounts and transaction relationships improves AML detection.

## 4. Test Results

The following values are the mean results across three random seeds: 42, 43, and 44.

| Model | Feature Set | Mean Test PR-AUC | Mean Test ROC-AUC |
|---|---|---:|---:|
| Linear SVM M1 | 8 transaction features | 0.0141 | 0.9103 |
| XGBoost M1 | 8 transaction features | 0.0731 | 0.9242 |
| XGBoost M2 | Transaction, account, and graph features | 0.5168 | 0.9818 |

PR-AUC is the primary metric because laundering-positive transactions represent only about 0.111% of the test set. ROC-AUC is included as a complementary measure.

## 5. Interpretation

### Linear SVM vs XGBoost M1

Both models use the same eight transaction-level features. XGBoost M1 achieves approximately 5.2 times the PR-AUC of the linear SVM.

This suggests that nonlinear decision boundaries and feature interactions captured by gradient-boosted trees are useful for this dataset.

### XGBoost M1 vs XGBoost M2

XGBoost M2 achieves approximately 7.1 times the PR-AUC of XGBoost M1.

The improvement indicates that account-level and graph-derived information provides substantial additional predictive value. This comparison changes both the feature set and the information available to the model, so the improvement cannot be attributed solely to the model algorithm.

### Alert Quality and Class Imbalance

At their separately selected validation thresholds, the SVM has mean recall of approximately 44.5% and mean precision of approximately 2.1%. It produces roughly 20,222 false positives per seed.

XGBoost M2 has mean recall of approximately 45.4% and mean precision of approximately 70.0%, with roughly 186 false positives per seed.

These threshold-based results illustrate the difference in alert burden at the selected operating points. Since the thresholds were selected separately, this is not a comparison at an identical alert budget.

## 6. Limitations

- The test set contains very few positive transactions relative to negative transactions.
- The SVM is a linear classifier and may not capture complex nonlinear relationships.
- XGBoost M2 uses additional account and graph-derived features, unlike M1.
- Results are reported across three seeds, but they remain specific to the dataset, feature engineering, and temporal split used.
- Threshold-based precision and recall depend on the chosen threshold.
- The later September 11–18 period has a substantially different positive rate and is treated as a separate distribution-shift diagnostic rather than part of the headline test comparison.

## 7. Conclusion

Among the evaluated models, XGBoost M2 achieves the strongest test performance, with a mean PR-AUC of 0.5168 and mean ROC-AUC of 0.9818.

XGBoost M1 outperforms the linear SVM when both use the same transaction-level features. Adding account-level and graph-derived features produces a much larger improvement in the XGBoost M2 experiment.

The results demonstrate the value of nonlinear modelling and richer transaction-network information for AML detection. The linear SVM remains a useful baseline for comparison.

## 8. Reproducibility

The evaluation results were generated using a temporal train-validation-test split. Model selection and threshold selection were performed using validation data. The final test results are treated as frozen evaluation outputs and were not reused for hyperparameter tuning.

Refer to the project's scripts, configuration files, and saved result JSON files for the implementation and detailed per-seed metrics.
'@ | Set-Content -Encoding utf8 docs\model_comparison.md