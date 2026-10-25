"""
Automated unit test suite for Phase 4 Serving, Inference Engine, and FastAPI endpoints.
"""

import sys
from pathlib import Path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import base64
import numpy as np
from fastapi.testclient import TestClient

from src.inference import WildfireInferenceEngine
from src.dataset import FEATURE_NAMES
from src.api import app

def test_onnx_model_inference():
    """Verify ONNX runtime session loads and produces bounded output [0, 1]."""
    engine = WildfireInferenceEngine(backend="onnx")
    assert engine.backend == "onnx"
    assert engine.session is not None

    dummy_input = np.ones((12, 64, 64), dtype=np.float32)
    result = engine.predict_tile(dummy_input, decision_threshold=0.35)

    assert result["probability_map"].shape == (64, 64)
    assert np.all(result["probability_map"] >= 0.0)
    assert np.all(result["probability_map"] <= 1.0)
    assert result["binary_mask"].shape == (64, 64)
    assert result["risk_level"] in ["Low", "Moderate", "High", "Extreme"]
    assert "eastward_km" in result["centroid_shift_km"]
    assert "northward_km" in result["centroid_shift_km"]

def test_inference_dict_input():
    """Verify inference engine works with channel dictionary input."""
    engine = WildfireInferenceEngine(backend="onnx")
    tile_dict = {name: np.ones((64, 64), dtype=np.float32) * 10.0 for name in FEATURE_NAMES}
    tile_dict["PrevFireMask"] = np.zeros((64, 64), dtype=np.float32)
    tile_dict["PrevFireMask"][20:30, 20:30] = 1.0

    result = engine.predict_tile(tile_dict, decision_threshold=0.40)
    assert result["prior_burn_pixels"] == 100
    assert result["decision_threshold"] == 0.40
    assert isinstance(result["predicted_burn_pixels"], int)

def test_fastapi_rest_service():
    """Verify FastAPI endpoints: root, health, metadata, and predict."""
    client = TestClient(app)

    # 1. Root
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "operational"

    # 2. Health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    # 3. Channel Metadata
    r_meta = client.get("/metadata/channels")
    assert r_meta.status_code == 200
    assert len(r_meta.json()["input_channels"]) == 12

    # 4. Predict Tile via Base64 Payload
    mock_tile = np.zeros((12, 64, 64), dtype=np.float32)
    mock_tile[0] = 800.0   # elevation
    mock_tile[2] = 4000.0  # NDVI
    b64_str = base64.b64encode(mock_tile.tobytes()).decode("utf-8")

    r_pred = client.post("/predict/tile", json={"raw_tensor_base64": b64_str, "decision_threshold": 0.35})
    assert r_pred.status_code == 200
    resp_data = r_pred.json()
    assert "risk_level" in resp_data
    assert "predicted_burn_area_km2" in resp_data
    assert "probability_matrix_base64" in resp_data

    # 5. Predict Sample from validation shard
    r_sample = client.get("/predict/sample/0")
    assert r_sample.status_code == 200
    sample_data = r_sample.json()
    assert "centroid_shift_km" in sample_data

if __name__ == "__main__":
    print("Running Phase 4 serving unit test suite...")
    test_onnx_model_inference()
    print("  [PASS] test_onnx_model_inference")
    test_inference_dict_input()
    print("  [PASS] test_inference_dict_input")
    test_fastapi_rest_service()
    print("  [PASS] test_fastapi_rest_service")
    print("\nALL SERVING UNIT TESTS PASSED SUCCESSFULLY!")
