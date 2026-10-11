"""
Prints the REAL shapes of your API and result files so the UI can be wired exactly.
Run from the project root (backend running on port 8001 is optional):
    python scripts/probe_shapes.py > probe.txt
Then attach probe.txt.   (It only reads; it changes nothing and prints no secrets.)
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8001"


def shape(x, d=0, maxd=4):
    if isinstance(x, dict):
        return "{...}" if d >= maxd else {k: shape(v, d + 1, maxd) for k, v in list(x.items())[:30]}
    if isinstance(x, list):
        return f"list[{len(x)}]" if not x else [f"list[{len(x)}]", shape(x[0], d + 1, maxd)]
    if isinstance(x, str):
        return x if len(x) <= 40 else x[:40] + "…"
    return x


def show(title, obj):
    print(f"\n=== {title}")
    print(json.dumps(obj, indent=1, default=str)[:3500])


# 1) endpoints
for name in ("openapi8001.json", "openapi.json"):
    p = ROOT / name
    if p.exists():
        spec = json.loads(p.read_text(encoding="utf-8"))
        print(f"### ENDPOINTS from {name}")
        for path, ops in spec.get("paths", {}).items():
            for m, op in ops.items():
                qs = [f"{q['name']}{'*' if q.get('required') else ''}" for q in op.get("parameters", []) if q.get("in") == "query"]
                ok = op.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {}).get("schema", {})
                print(f"{m.upper():5} {path}  query={qs}  -> {ok.get('$ref', ok.get('type', '?')).split('/')[-1]}")
        break

# 2) result JSON files (one sample per kind)
res = ROOT / "results"
seen = set()
for p in sorted(res.rglob("*.json")):
    kind = p.parent.name + "/" + (p.stem.split("_", 1)[1] if p.stem[:8].isdigit() else p.stem)
    if kind in seen:
        continue
    seen.add(kind)
    try:
        show(str(p.relative_to(ROOT)), shape(json.loads(p.read_text(encoding="utf-8"))))
    except Exception as e:
        print(f"\n=== {p.name}: unreadable ({e})")

# 3) parquet files: columns + 2 rows
try:
    import pyarrow.parquet as pq
    targets = list(res.glob("*.parquet")) + list((ROOT / "data" / "processed").glob("*.parquet"))
    for p in targets:
        pf = pq.ParquetFile(p)
        head = next(pf.iter_batches(batch_size=2)).to_pandas()
        print(f"\n=== {p.relative_to(ROOT)}  rows={pf.metadata.num_rows}")
        print(head.dtypes.to_string())
        print(head.head(2).to_string()[:1500])
except Exception as e:
    print("\n(parquet probe skipped:", e, ")")

# 4) live API (GET endpoints without path parameters)
try:
    paths = [p for p in spec["paths"] if "{" not in p and "get" in spec["paths"][p]]
    for p in paths:
        t = time.time()
        try:
            with urllib.request.urlopen(API + p, timeout=25) as r:
                show(f"LIVE GET {p}  ({time.time()-t:.1f}s)", shape(json.loads(r.read())))
        except urllib.error.HTTPError as e:
            print(f"\n=== LIVE GET {p}: HTTP {e.code} {e.read()[:300]!r}")
        except Exception as e:
            print(f"\n=== LIVE GET {p}: {e}")
except Exception as e:
    print("\n(live probe skipped:", e, ")")

# 5) docs
for d in ("docs/api_contract.md",):
    p = ROOT / d
    if p.exists():
        print(f"\n### {d}\n" + p.read_text(encoding="utf-8")[:6000])
