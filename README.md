# AML Graph Project

## Graph-Enhanced Anti-Money Laundering Detection

A research-oriented Anti-Money Laundering (AML) detection system that investigates whether **historical account behavior and transaction-network information can improve suspicious transaction detection beyond transaction-level information alone**.

The project combines:

* Transaction-level machine learning
* Historical account behavior
* Graph-based network features
* XGBoost classification
* Explainable AI
* Leakage-safe temporal evaluation
* A FastAPI backend
* A React dashboard

The system is designed as a **read-only analytical dashboard** rather than a live banking system.

---

## 1. Project Objective

Traditional transaction-level AML models mainly look at the properties of an individual transaction.

For example:

```text
Amount
Currency
Payment format
Time
Sender / receiver relationship
```

However, suspicious activity often depends on **historical context**.

A transaction may look normal by itself but become suspicious when we consider:

* Previous activity of the sender
* Previous activity of the receiver
* Whether the sender and receiver have interacted before
* How frequently they interact
* The number of counterparties connected to an account
* The account's position in the transaction network
* The size of its connected component

### Research Question

> **Does historical account and transaction-network information improve AML detection compared with using transaction-level information alone?**

---

# 2. Model Design

The project deliberately uses **two main models**.

## Model 1 — Transaction Only

M1 uses information available from the current transaction.

```text
Current Transaction
       │
       ├── Amount
       ├── Currency
       ├── Payment format
       ├── Time
       ├── Same-bank relationship
       └── Self-loop / FX indicators
       │
       ▼
    XGBoost
       │
       ▼
   AML Score
```

---

## Model 2 — Transaction + Historical Context

M2 extends M1 with historical account and network information.

```text
                    ┌── Transaction Features
                    │
                    ├── Account History
                    │
                    └── Graph / Network History
                              │
                              ▼
                         Feature Matrix
                              │
                              ▼
                           XGBoost
                              │
                              ▼
                          AML Score
```

Therefore:

```text
M1 = Transaction features

M2 = Transaction
     + Account history
     + Graph history
```

The primary comparison is:

```text
M2 PR-AUC − M1 PR-AUC
```

This directly measures whether historical context provides additional predictive value.

---

# 3. Dataset

The project uses a transaction dataset containing approximately **5.08 million transactions**.

The processed dataset contains:

* 5,078,345 transactions
* 11 original/derived core columns
* Very low positive-class prevalence
* 9 exact duplicate rows identified during auditing

The raw dataset is **not committed to GitHub** because of its size.

Local data is stored under:

```text
data/
```

and is excluded through `.gitignore`.

---

# 4. Temporal Data Split

A random train/test split is intentionally **not used**.

Financial transactions are time-dependent, so using future information to predict past transactions would create unrealistic results.

The dataset is divided chronologically:

| Period    | Purpose                            |
| --------- | ---------------------------------- |
| Sep 1     | Warm-up                            |
| Sep 2–6   | Training                           |
| Sep 7–8   | Validation                         |
| Sep 9–10  | Final Test                         |
| Sep 11–18 | Tail / Distribution Shift Analysis |

The main reported test results come only from:

```text
Sep 9–10
```

The tail period is analyzed separately and is not included in headline metrics.

---

# 5. Leakage Prevention

Leakage prevention is one of the most important design principles of this project.

For a transaction occurring on date `D`, historical features can only use information from:

```text
date < D
```

Never:

```text
date <= D
```

This prevents information from the same day from leaking into the prediction.

### Historical information includes:

* Previous account activity
* Previous sender/receiver relationships
* Previous pair interactions
* Previous network structure

Labels are never used to construct features.

Train-only statistics such as:

* Amount standardization parameters
* Class imbalance weights

are calculated using training data only.

The project also contains explicit leakage tests.

---

# 6. Feature Engineering

## 6.1 Transaction Features

M1 contains transaction-level features such as:

```text
t_log_amount_paid
t_amount_z_in_currency
t_payment_format
t_cur_paid
t_is_fx
t_is_self_loop
t_same_bank
t_hour
```

The raw transaction amount is log-transformed because transaction amounts are highly skewed.

Amount standardization is performed within currency using statistics learned from training data.

---

## 6.2 Historical Account Features

M2 adds historical behavior for both sender and receiver accounts.

Examples include:

```text
n_out_rate
n_in_rate
mean_z_out
mean_z_in
max_z_out
days_active_frac
days_since_last
fmt_diversity
```

These describe how an account behaved **before the current transaction**.

For example:

```text
Current transaction
       │
       ▼
Sender account
       │
       ├── Previous outgoing activity
       ├── Previous incoming activity
       ├── Previous counterparties
       ├── Previous activity frequency
       └── Previous transaction characteristics
```

---

# 7. Graph Representation

The transaction network is represented as a directed graph.

## Nodes

Each unique:

```text
(Bank, Account)
```

combination becomes a graph node.

Each node receives an integer `node_id`.

---

## Edges

A transaction:

```text
Sender → Receiver
```

becomes a directed graph edge.

Self-loops are excluded from graph topology:

```text
A → A   ❌
```

while:

```text
A → B   ✓
```

remains in the graph.

Raw transactions remain the prediction units even though graph edges are aggregated.

---

# 8. Historical Graph Snapshots

The graph is constructed historically.

For a feature date `D`, only edges whose first occurrence happened before `D` are included.

```text
Historical transactions
        │
        ▼
Edges observed before D
        │
        ▼
Graph snapshot for D
        │
        ▼
Network features
```

This ensures that the graph does not contain future information.

---

# 9. Graph Features

The project uses lightweight and interpretable graph features.

### Pair history

```text
g_pair_seen_before
g_pair_prior_count_log
g_reverse_pair_seen
```

These describe the previous relationship between two accounts.

### Network structure

```text
g_pagerank_log
g_wcc_size_log
g_n_unique_out_log
g_n_unique_in_log
g_has_sent_before
g_has_received_before
```

These capture the historical position and connectivity of an account.

---

## Why not use every graph algorithm?

Algorithms such as:

* Betweenness centrality
* Closeness centrality
* Eigenvector centrality

were intentionally excluded from the core pipeline.

The goal is to use **useful, interpretable, computationally practical features** rather than adding graph algorithms simply for complexity.

---

# 10. Model Training

Both models use:

```text
XGBoost
```

with histogram-based tree training.

Because laundering transactions are extremely rare, class imbalance is handled using:

```text
scale_pos_weight
```

calculated from the training data.

SMOTE is not used.

Accuracy is not treated as the primary metric because a highly imbalanced dataset can produce misleadingly high accuracy.

---

# 11. Experiment Design

Each model is evaluated in two feature settings:

| Model | Full Features | No-Shortcut Features |
| ----- | ------------: | -------------------: |
| M1    |             ✓ |                    ✓ |
| M2    |             ✓ |                    ✓ |

This produces four configurations:

```text
M1 Full
M1 No-Shortcut
M2 Full
M2 No-Shortcut
```

Final experiments use multiple random seeds for robustness.

---

# 12. Shortcut Analysis

Some dataset characteristics may provide unusually strong predictive signals that do not necessarily represent meaningful AML behavior.

Examples include:

```text
Payment format
Currency
FX indicator
Self-loop indicator
Format diversity
```

Therefore, a **no-shortcut feature variant** is evaluated.

The project also performs a hard-subset analysis on transactions satisfying:

```text
ACH
AND non-self
AND non-FX
```

This checks whether the observed model improvement survives after removing several strong dataset artifacts.

---

# 13. Evaluation

## Primary Metric

### PR-AUC

Precision-Recall AUC is the primary metric because the positive AML class is highly imbalanced.

The main research result is:

```text
M2 PR-AUC
      -
M1 PR-AUC
```

---

## Secondary Metrics

The project also reports:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Precision-Recall curve
* Recall at a fixed alert budget

Accuracy is reported only as a supplementary metric.

---

# 14. Threshold Selection

The classification threshold is selected using the validation set.

The chosen rule is:

```text
Threshold = value that maximizes F1 on validation
```

The threshold is then frozen before evaluating the test set.

Therefore:

```text
Training
   ↓
Validation
   ↓
Choose threshold
   ↓
FREEZE threshold
   ↓
Final Test
```

The test set is not used to select the threshold.

---

# 15. Statistical Analysis

To determine whether the difference between M1 and M2 is reliable, the project uses a **paired bootstrap**.

The same test samples are used for both models.

The analysis reports:

```text
M2 PR-AUC − M1 PR-AUC
```

along with a 95% confidence interval.

---

# 16. Explainability

The project uses multiple explainability techniques.

## SHAP

SHAP is used to understand which features contributed to individual model predictions and overall model behavior.

The Transaction Inspector can show the top contributing features for a selected transaction.

---

## Grouped Permutation Importance

M2 combines account and graph information.

To understand where the improvement comes from, features are grouped into:

### A — Account History

Historical account behavior.

### P — Pair History

Previous sender/receiver relationships.

### C — Network Structure

Features such as:

```text
PageRank
Component size
Unique counterparties
Historical connectivity
```

Each group is permuted and the change in PR-AUC is measured.

This provides a decomposition of the sources of predictive value without introducing a third model.

---

# 17. Dashboard

The project includes a read-only analytical dashboard.

### Overview

Shows:

* Research question
* Dataset summary
* M1 vs M2 headline result
* Confidence interval
* Key limitations

### Data & Split

Shows:

* Transaction volume
* Positive counts
* Temporal split
* Data quality
* Shortcut analysis

### Network

Shows:

* Network statistics
* Degree information
* New-pair analysis
* Suspicious subgraphs

### Model Results

Shows:

* M1 vs M2
* Full vs no-shortcut results
* PR curves
* Confusion matrices
* SHAP analysis
* Grouped permutation importance
* Hard-subset results

### Transaction Inspector

Allows users to inspect precomputed test transactions.

For a selected transaction, the dashboard can show:

```text
Transaction information
        ↓
M1 score
        ↓
M2 score
        ↓
Predicted outcome
        ↓
Actual outcome
        ↓
Top SHAP features
```

The Inspector uses precomputed results rather than requiring live model inference.

Live "what-if" scoring is considered a stretch feature and is only enabled if it can be validated against batch predictions.

### Limitations

Documents:

* Dataset limitations
* Shortcut effects
* Distribution shift
* Synthetic-data concerns
* Why a model prediction is not proof of criminal activity
* Why this is not a production banking system

---

# 18. System Architecture

```text
                         RAW TRANSACTIONS
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Data Validation     │
                    │ + Temporal Split    │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
          Transaction +             Historical Graph
          Account Features              Features
                  │                         │
                  │                         │
                  └────────────┬────────────┘
                               ▼
                       Feature Assembly
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
                   M1                     M2
             Transaction Only     Transaction + History
                    │                     │
                    └──────────┬──────────┘
                               ▼
                            XGBoost
                               │
                               ▼
                     Validation / Test
                               │
                               ▼
                 ┌────────────────────────┐
                 │ Evaluation + Explainability │
                 └────────────┬───────────┘
                              │
                              ▼
                          results/
                              │
                              ▼
                           FastAPI
                              │
                              ▼
                            React
                              │
                              ▼
                       AML Dashboard
```

---

# 19. Repository Structure

```text
pods_AML/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── Makefile
├── CODEOWNERS
│
├── configs/
│   ├── config.yaml
│   └── feature_sets.yaml
│
├── docs/
│   ├── data_contract.md
│   ├── api_contract.md
│   ├── methodology.md
│   ├── data_quality.md
│   └── limitations.md
│
├── src/
│   └── aml/
│       ├── common/
│       ├── data/
│       ├── features/
│       ├── graph/
│       ├── models/
│       ├── evaluation/
│       └── export/
│
├── scripts/
│
├── backend/
│   └── app/
│
├── frontend/
│   └── src/
│       ├── api/
│       ├── layout/
│       └── pages/
│
├── tests/
│
├── data_stub/
├── results_stub/
│
├── data/
├── models/
└── results/
```

### Important

Large datasets, trained models and generated artifacts are not committed to GitHub.

```text
data/
models/
results/
```

are locally generated/managed directories.

---

# 20. Team Responsibilities

The project is divided into two primary ownership areas.

## Person 1 — Data + ML + Evaluation

Responsible for:

* Data validation
* Temporal splitting
* Transaction features
* Account features
* Feature assembly
* XGBoost
* Class imbalance
* Threshold selection
* Model evaluation
* Bootstrap
* SHAP
* Grouped permutation
* Hard-subset analysis
* Model/result exports
* Data & Split page
* Model Results page

---

## Person 2 — Graph + Application

Responsible for:

* Graph node construction
* Graph snapshots
* Pair history
* Network features
* Graph exports
* FastAPI backend
* API routers
* React application shell
* Overview page
* Network page
* Transaction Inspector
* Limitations page

---

# 21. Data Handoff Contracts

The two development streams communicate through defined data contracts.

### Person 1 → Person 2

```text
transactions_clean.parquet
node_map.parquet
results/
```

### Person 2 → Person 1

```text
graph_node_day.parquet
graph_pair_txn.parquet
```

The schemas are defined in:

```text
docs/data_contract.md
configs/feature_sets.yaml
src/aml/common/schema.py
```

Contract changes require coordination between both developers.

---

# 22. Development Workflow

The `main` branch is protected.

Development happens through feature branches:

```text
p1/<topic>
p2/<topic>
contract/<change>
```

Typical workflow:

```bash
git checkout -b p1/data-pipeline

# make changes

git add .
git commit -m "Implement chronological data split"
git push -u origin p1/data-pipeline
```

Then open a Pull Request.

The other developer reviews the changes before merging.

---

# 23. Testing

The project includes tests for:

```text
Schema validation
ID consistency
Data leakage
Graph construction
API behavior
```

Important principle:

> **A model that performs well because of leakage is not a successful model.**

Therefore, leakage tests are treated as mandatory rather than optional.

---

# 24. Installation

Clone the repository:

```bash
git clone <repository-url>
cd pods_AML
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install project dependencies:

```bash
pip install -e .
```

The exact dependency configuration is maintained in:

```text
pyproject.toml
```

---

# 25. Running the Project

The project is designed around reproducible pipeline stages.

Conceptually:

```text
Raw Data
   ↓
Data Pipeline
   ↓
Feature Pipeline
   ↓
Graph Pipeline
   ↓
Model Experiments
   ↓
Evaluation
   ↓
Exports
   ↓
FastAPI
   ↓
React Dashboard
```

The corresponding scripts are located in:

```text
scripts/
├── run_data.py
├── run_features.py
├── run_graph.py
└── run_experiments.py
```

Project-wide commands are maintained in the `Makefile`.

---

# 26. Reproducibility

The project uses controlled random seeds for model experiments.

Final model experiments use:

```text
42
43
44
```

Configuration is centralized in:

```text
configs/
```

and shared schemas are maintained in:

```text
src/aml/common/schema.py
```

This reduces differences between experiments and prevents different parts of the project from silently using different assumptions.

---

# 27. Key Design Principles

The project follows these principles:

### 1. Temporal correctness

Historical features must not use future information.

### 2. Transaction-level prediction

Transactions remain the prediction units even though graphs are used to provide context.

### 3. Simple baseline

M1 establishes what can be achieved from transaction information alone.

### 4. Historical context

M2 tests whether account and network history adds value.

### 5. No unnecessary complexity

Only graph features with reasonable computational cost and interpretability are included.

### 6. Honest evaluation

The project explicitly investigates dataset shortcuts and distribution shift.

### 7. Explainability

Model predictions should be inspectable rather than treated as unexplained scores.

### 8. Reproducibility

Data contracts, configuration, seeds and tests are part of the project rather than afterthoughts.

---

# 28. Limitations

This project should **not** be interpreted as a production AML system.

Important limitations include:

* The dataset may contain synthetic or artificial patterns.
* Strong dataset shortcuts can inflate predictive performance.
* Historical network information may capture dataset-specific behavior.
* The tail period exhibits distribution shift.
* A suspicious prediction does not prove criminal activity.
* The system does not replace human investigation.
* The dashboard is analytical and read-only.
* The system has not been validated against real-world banking operations.
* Correlated graph and account features make attribution difficult.
* Permutation importance can underestimate the importance of correlated feature groups.

These limitations are documented in greater detail in:

```text
docs/limitations.md
```

---

# 29. Expected Final Result

The final project answers one central question:

> **Does adding historical account and transaction-network context improve AML detection over transaction-level information alone?**

The answer is evaluated through:

```text
M1 vs M2
     ↓
PR-AUC comparison
     ↓
Paired bootstrap confidence interval
     ↓
No-shortcut analysis
     ↓
Hard-subset analysis
     ↓
Grouped permutation importance
     ↓
SHAP explanation
     ↓
Tail / distribution-shift analysis
```

The goal is therefore not simply to build a model with a high score.

The goal is to determine **whether the improvement is meaningful, robust, explainable and free from obvious temporal leakage.**

---

## Project Status

🚧 **In Development**

The repository is being developed incrementally through the data, graph, ML, evaluation and application stages.

---

## Team

**Person 1 — Data, ML & Evaluation**

**Person 2 — Graph, Backend & Frontend**

---

## License

Add the appropriate license for your project here.

---

**AML Graph Project — investigating whether historical transaction-network context can improve suspicious transaction detection.**
