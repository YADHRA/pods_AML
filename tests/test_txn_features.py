import pandas as pd
from aml.features import transaction as T


def _df():
    return pd.DataFrame({
        "txn_id": [0, 1, 2, 3, 4],
        "ts": pd.to_datetime(["2022-09-02 03:10"] * 3 + ["2022-09-07 15:00", "2022-09-08 23:59"]),
        "amount_paid": [10.0, 100.0, 1000.0, 50.0, 5.0],
        "cur_paid": ["US Dollar"] * 3 + ["US Dollar", "Yen"],
        "cur_recv": ["US Dollar"] * 3 + ["Euro", "Yen"],
        "payment_format": ["ACH", "ACH", "Cheque", "ACH", "Wire"],
        "src_bank": ["010", "010", "020", "010", "030"],
        "dst_bank": ["010", "020", "020", "020", "030"],
        "is_self_loop": [True, False, False, False, True],
        "split": ["train", "train", "train", "val", "val"],
    })


def test_basic_flags():
    f, _ = T.build_txn_features(_df())
    assert list(f.t_hour) == [3, 3, 3, 15, 23]
    assert list(f.t_is_fx) == [0, 0, 0, 1, 0]
    assert list(f.t_same_bank) == [1, 0, 1, 0, 1]
    assert list(f.t_is_self_loop) == [1, 0, 0, 0, 1]
    assert abs(f.t_log_amount_paid[2] - 3.0) < 1e-6


def test_stats_use_train_rows_only():
    a, sa = T.build_txn_features(_df())
    d2 = _df(); d2.loc[3:, "amount_paid"] = [9e9, 9e9]          # change val rows drastically
    b, sb = T.build_txn_features(d2)
    assert sa["US Dollar"] == sb["US Dollar"]                   # stats unchanged -> no val/test leakage
    assert abs(a.t_amount_z_in_currency[0] - b.t_amount_z_in_currency[0]) < 1e-9


def test_unseen_currency_is_nan_and_no_label_column():
    f, _ = T.build_txn_features(_df())
    assert pd.isna(f.t_amount_z_in_currency[4])                 # Yen never in train
    assert "is_laundering" not in f.columns