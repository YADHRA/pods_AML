import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_stubs import build  # noqa: E402
from aml.common import schema as S  # noqa: E402


def test_stub_files_match_contract(tmp_path):
    build(out=tmp_path)
    S.assert_transactions_clean(pd.read_parquet(tmp_path / "transactions_clean.parquet"))
    S.assert_graph_node_day(pd.read_parquet(tmp_path / "graph_node_day.parquet"))
    S.assert_graph_pair_txn(pd.read_parquet(tmp_path / "graph_pair_txn.parquet"))
    nm = pd.read_parquet(tmp_path / "node_map.parquet")
    assert set(nm.columns) == set(S.NODE_MAP_DTYPES)


def test_no_warmup_rows_in_pair_features(tmp_path):
    build(out=tmp_path)
    tx = pd.read_parquet(tmp_path / "transactions_clean.parquet")
    pair = pd.read_parquet(tmp_path / "graph_pair_txn.parquet")
    warm = set(tx.loc[tx.split == "warmup", "txn_id"])
    assert not (set(pair.txn_id) & warm)
