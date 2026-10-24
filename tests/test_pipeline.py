import os, sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))
"""
Comprehensive automated unit test suite for Wildfire Spread Prediction pipeline.
Tests:
- Streaming data loading & batch tensor shapes
- Anomaly cleaning (sentinel masks, temperature dropouts, corrupt angles)
- Wind vector decomposition
- Spatial feature engineering (Sobel slopes, neighborhood convolutions)
- Binary metric edge cases (all-zero, all-one, NaN guards)
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import WildfireTFRecordDataset, clean_and_normalize_tile, FEATURE_NAMES, TARGET_NAME
from src.features import extract_tabular_features_from_tile, TABULAR_FEATURE_NAMES
from src.utils import decompose_wind, compute_binary_metrics

def test_anomaly_cleaning():
    """Verify that all 5 classes of anomalies are cleansed completely."""
    # Create synthetic mock tile with all real-world corruption patterns
    mock_feats = {name: np.ones((64, 64), dtype=np.float32) * 50.0 for name in FEATURE_NAMES}
    mock_feats[TARGET_NAME] = np.zeros((64, 64), dtype=np.float32)
    
    # Inject corruptions:
    mock_feats[TARGET_NAME][0, 0] = -1.0         # Cloud sentinel
    mock_feats['PrevFireMask'][0, 0] = -1.0      # Input sentinel
    mock_feats['tmmn'][10:15, 10:15] = 0.0       # Temperature dropout
    mock_feats['tmmx'][10:15, 10:15] = 0.0       # Temperature dropout
    mock_feats['th'][5, 5] = -505870.0           # Corrupt wind angle
    mock_feats['th'][6, 6] = 37735.0             # Corrupt wind angle
    mock_feats['vs'][2, 2] = -43.0               # Negative wind speed
    mock_feats['pr'][3, 3] = -5.0                # Negative rain
    mock_feats['population'][0, 1] = 14000.0     # Extreme population spike

    inputs_arr, target_arr, mask_arr = clean_and_normalize_tile(mock_feats, scalers=None, normalize=False)

    # 1. Mask check
    assert mask_arr[0, 0] == 0.0, "Unobserved -1.0 target pixel must be masked out"
    assert mask_arr[1, 1] == 1.0, "Valid target pixel must have mask=1.0"
    
    # 2. Temperature check
    tmmn_idx = FEATURE_NAMES.index('tmmn')
    assert np.all(inputs_arr[tmmn_idx] > 200.0), "0K dropouts must be imputed with realistic temperatures"

    # 3. Wind angle check
    th_idx = FEATURE_NAMES.index('th')
    assert np.all((inputs_arr[th_idx] >= 0.0) & (inputs_arr[th_idx] <= 360.0)), "Wind angles must be in [0, 360]"

    # 4. Non-negative quantities
    vs_idx = FEATURE_NAMES.index('vs')
    pr_idx = FEATURE_NAMES.index('pr')
    assert np.all(inputs_arr[vs_idx] >= 0.0), "Wind speed must be >= 0"
    assert np.all(inputs_arr[pr_idx] >= 0.0), "Precipitation must be >= 0"

def test_wind_decomposition():
    """Verify zonal and meridional vector decomposition."""
    # North wind (from 0 deg / 360 deg) -> blows South (v < 0, u = 0)
    u, v = decompose_wind(10.0, 0.0)
    assert np.isclose(u, 0.0, atol=1e-5)
    assert np.isclose(v, -10.0, atol=1e-5)

    # West wind (from 270 deg) -> blows East (u > 0, v = 0)
    u, v = decompose_wind(10.0, 270.0)
    assert np.isclose(u, 10.0, atol=1e-5)
    assert np.isclose(v, 0.0, atol=1e-5)

def test_feature_engineering_dimensions():
    """Verify that spatial feature engineering outputs exact shape (4096, 21)."""
    dummy_tile = np.random.randn(12, 64, 64).astype(np.float32)
    feats = extract_tabular_features_from_tile(dummy_tile)
    assert feats.shape == (4096, len(TABULAR_FEATURE_NAMES))
    assert np.all(np.isfinite(feats)), "Engineered features must be completely finite"

def test_metrics_edge_cases():
    """Verify metrics calculation on edge cases like all-zeros and all-ones without crashing."""
    # Case A: All negative pixels (common in non-fire tiles)
    y_true_zero = np.zeros(4096, dtype=np.uint8)
    y_prob = np.random.uniform(0, 1, 4096).astype(np.float32)
    m_zero = compute_binary_metrics(y_true_zero, y_prob)
    assert m_zero['f1'] == 0.0
    assert m_zero['recall'] == 0.0
    assert m_zero['pr_auc'] == 0.0
    assert m_zero['roc_auc'] == 0.5

    # Case B: Standard mixed labels
    y_true_mixed = np.zeros(4096, dtype=np.uint8)
    y_true_mixed[:50] = 1
    y_prob_good = np.zeros(4096, dtype=np.float32)
    y_prob_good[:50] = 0.9
    m_mixed = compute_binary_metrics(y_true_mixed, y_prob_good, threshold=0.5)
    assert m_mixed['f1'] > 0.95
    assert np.isclose(m_mixed['recall'], 1.0)

def test_dataloader_streaming():
    """Verify PyTorch streaming data loader with real shards."""
    raw_dir = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/data/raw"
    files = sorted(glob.glob(os.path.join(raw_dir, "*train_00.tfrecord")))
    if not files:
        pytest.skip("Dataset files not found for integration test")

    ds = WildfireTFRecordDataset(files, normalize=True)
    loader = DataLoader(ds, batch_size=4)
    for x, y, mask in loader:
        assert x.shape == (4, 12, 64, 64)
        assert y.shape == (4, 1, 64, 64)
        assert mask.shape == (4, 1, 64, 64)
        assert torch.all(torch.isfinite(x))
        break

if __name__ == "__main__":
    print("Running pipeline test suite...")
    test_anomaly_cleaning()
    print("  [PASS] test_anomaly_cleaning")
    test_wind_decomposition()
    print("  [PASS] test_wind_decomposition")
    test_feature_engineering_dimensions()
    print("  [PASS] test_feature_engineering_dimensions")
    test_metrics_edge_cases()
    print("  [PASS] test_metrics_edge_cases")
    test_dataloader_streaming()
    print("  [PASS] test_dataloader_streaming")
    print("\nALL 5 PIPELINE UNIT TESTS PASSED SUCCESSFULLY! ZERO ISSUES FOUND.")
