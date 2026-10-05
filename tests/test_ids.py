import pandas as pd
from aml.common.config import load_config
from aml.data import load, split, validate

HEADER = load.EXPECTED_HEADER
ROWS = [  # deliberately UNSORTED, with an exact duplicate, leading-zero banks, same account string under 2 banks
    "2022/09/03 10:00,010,ACC1,020,ACC2,100.0,US Dollar,100.0,US Dollar,ACH,1",
    "2022/09/01 00:05,010,ACC1,010,ACC1,5.0,US Dollar,5.0,US Dollar,Reinvestment,0",
    "2022/09/03 10:00,010,ACC1,020,ACC2,100.0,US Dollar,100.0,US Dollar,ACH,1",   # exact duplicate of row 0
    "2022/09/01 00:05,020,ACC1,010,ACC9,7.0,Euro,6.0,US Dollar,Wire,0",            # ACC1 under bank 020 != ACC1 under 010
    "2022/09/11 01:00,010,ACC9,020,ACC2,9.0,US Dollar,9.0,US Dollar,ACH,1",
]


def _clean(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    raw = load.read_raw(p)
    return raw, *split.build_clean(raw, load_config())


def test_txn_id_is_original_row_index_and_survives_sorting(tmp_path):
    raw, clean, _ = _clean(tmp_path)
    assert list(raw.txn_id) == [0, 1, 2, 3, 4]
    assert list(clean.txn_id) == [1, 3, 0, 2, 4]              # sorted by (ts, txn_id); ties by txn_id
    orig = pd.read_csv(tmp_path / "t.csv", header=0, usecols=[5], names=["x"])  # amount_received by position
    amt = clean.set_index("txn_id")["amount_received"]
    assert amt.loc[3] == 7.0 and amt.loc[0] == 100.0           # row identity not lost


def test_duplicates_and_leading_zeros_kept(tmp_path):
    raw, clean, nm = _clean(tmp_path)
    assert len(clean) == 5 and validate.audit(raw)["exact_duplicate_extra_rows"] == 1
    assert "010" in set(clean.src_bank) and "10" not in set(clean.src_bank)


def test_node_identity_is_bank_plus_account(tmp_path):
    _, clean, nm = _clean(tmp_path)
    acc1 = nm[nm.account == "ACC1"]
    assert len(acc1) == 2 and set(acc1.bank) == {"010", "020"}
    r = clean.set_index("txn_id")
    assert r.loc[1, "is_self_loop"] and not r.loc[0, "is_self_loop"]


def test_splits_by_date(tmp_path):
    _, clean, _ = _clean(tmp_path)
    r = clean.set_index("txn_id")["split"]
    assert r.loc[1] == "warmup" and r.loc[0] == "train" and r.loc[4] == "tail"
