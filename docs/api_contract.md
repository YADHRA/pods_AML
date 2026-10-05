# API contract (SHARED) — FastAPI, read-only GET, prefix /api. P2 owns implementation.
Stub mode: AML_USE_STUB=1 reads results_stub/ + data_stub/.

/overview · /data/daily · /data/splits · /data/shortcuts
/graph/degree · /graph/hubs · /graph/new-pair-lift · /graph/subgraphs · /graph/subgraphs/{id}
/models/runs · /models/runs/{run_id} · /models/curves · /models/permutation · /models/shap · /models/hard-subset
/transactions?outcome=&model= · /transactions/{txn_id}

Response shapes: to be filled in Step 0.2 once results JSON shapes are fixed.
