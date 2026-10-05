# pods_AML — Graph-Based Anti-Money-Laundering Detection

**Question:** does historical account/network information improve transaction-level AML
classification vs. transaction information alone, under a leakage-safe chronological setup?

Dataset: IBM AML `HI-Small_Trans.csv` (not committed). Set `AML_DATA_DIR` to the folder holding it.

## Quick start
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install        # pip install -e .
make stub           # writes tiny schema-valid fake files into data_stub/
make test
```
Contracts: `docs/data_contract.md` · Feature lists: `configs/feature_sets.yaml` · Params: `configs/config.yaml`

## Ownership
P1: data, features, models, evaluation, Data&Split + Results pages. P2: graph, backend, frontend (other pages).
Branches: `p1/<topic>`, `p2/<topic>`, `contract/<change>`. No direct pushes to `main`.
