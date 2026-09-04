"""Tests for modeling/preprocessing.py: train-only fitting, missing-value
handling, categorical encoding, and artifact save/load (plan §15/§16/§25/§26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scm_dataset.modeling.preprocessing import (
    FeaturePreprocessor,
    _TypePreprocessor,
    full_time_fit_mask,
    supplier_fit_mask,
)
from scm_dataset.schema.nodes import NodeType


def test_numeric_scaler_is_fit_on_train_rows_only():
    df = pd.DataFrame({"x": [10.0, 12.0, 11.0, 1000.0, -1000.0]})  # last two are "test" outliers
    fit_mask = pd.Series([True, True, True, False, False])
    tp = _TypePreprocessor(numeric_columns=["x"], categorical_columns=[])
    tp.fit(df, fit_mask)

    # mean/std must reflect only the first three (train) rows: mean=11, std=sqrt(2/3)
    assert tp.numeric_stats["x"].mean == pytest.approx(11.0)
    assert tp.numeric_stats["x"].std == pytest.approx(np.std([10.0, 12.0, 11.0]))


def test_missing_value_gets_indicator_and_train_median_imputation():
    df = pd.DataFrame({"x": [10.0, np.nan, 30.0, np.nan]})
    fit_mask = pd.Series([True, True, True, False])  # train median over [10, NaN->dropped, 30] = 20
    tp = _TypePreprocessor(numeric_columns=["x"], categorical_columns=[])
    tp.fit(df, fit_mask)
    assert tp.numeric_stats["x"].median == pytest.approx(20.0)

    out = tp.transform(df)
    assert list(out["x_missing"]) == [0.0, 1.0, 0.0, 1.0]
    # row 1's imputed raw value should equal the train median (20.0) before scaling
    imputed_row1_scaled = out.loc[1, "x"]
    expected = (20.0 - tp.numeric_stats["x"].mean) / tp.numeric_stats["x"].std
    assert imputed_row1_scaled == pytest.approx(expected)


def test_constant_numeric_column_does_not_divide_by_zero():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    tp = _TypePreprocessor(numeric_columns=["x"], categorical_columns=[])
    tp.fit(df, pd.Series([True, True, True]))
    out = tp.transform(df)
    assert np.isfinite(out["x"]).all()


def test_categorical_vocab_fit_on_train_rows_only_and_unseen_maps_to_unknown():
    df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})  # "C" only appears in a non-fit row
    fit_mask = pd.Series([True, True, True, False])
    tp = _TypePreprocessor(numeric_columns=[], categorical_columns=["cat"])
    tp.fit(df, fit_mask)
    assert tp.categorical_vocab["cat"] == ["A", "B"]

    out = tp.transform(df)
    assert out.loc[3, "cat__A"] == 0.0
    assert out.loc[3, "cat__B"] == 0.0
    assert out.loc[3, "cat____UNKNOWN__"] == 1.0
    assert out.loc[0, "cat__A"] == 1.0


def test_output_dim_accounts_for_missing_indicators_and_unknown_bucket():
    tp = _TypePreprocessor(numeric_columns=["a", "b"], categorical_columns=["c"])
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": ["x", "y"]})
    tp.fit(df, pd.Series([True, True]))
    # 2 numeric * (value + missing) + (2 categories + UNKNOWN)
    assert tp.output_dim() == 2 * 2 + 3


def test_save_and_load_roundtrip_preserves_transform(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", "x"]})
    tp = _TypePreprocessor(numeric_columns=["a"], categorical_columns=["c"])
    tp.fit(df, pd.Series([True, True, True]))

    preprocessor = FeaturePreprocessor(by_type={NodeType.SUPPLIER: tp})
    preprocessor.save(str(tmp_path))
    loaded = FeaturePreprocessor.load(str(tmp_path))

    original = preprocessor.transform(NodeType.SUPPLIER, df)
    restored = loaded.transform(NodeType.SUPPLIER, df)
    pd.testing.assert_frame_equal(original, restored)


def test_full_time_fit_mask_selects_by_time_level_regardless_of_entity():
    index = pd.MultiIndex.from_tuples([("m0", 5), ("m0", 6), ("m1", 5), ("m1", 6)], names=["material_id", "time"])
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=index)
    mask = full_time_fit_mask(df, train_times={5})
    assert list(mask) == [True, False, True, False]


def test_supplier_fit_mask_selects_exact_supplier_time_pairs():
    index = pd.MultiIndex.from_tuples([("s0", 5), ("s0", 6), ("s1", 5)], names=["supplier_id", "time"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=index)
    train_examples = pd.DataFrame({"supplier_id": ["s0"], "time": [5]})
    mask = supplier_fit_mask(df, train_examples)
    assert list(mask) == [True, False, False]
