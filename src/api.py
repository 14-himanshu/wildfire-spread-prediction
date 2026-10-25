"""
Production FastAPI REST microservice for Wildfire Spread Prediction.
Provides low-latency model inference, channel catalog metadata,
and interactive evaluation on historical validation tiles.
"""

import base64
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from src.inference import WildfireInferenceEngine
    from src.dataset import FEATURE_NAMES, FEATURE_METADATA, WildfireTFRecordDataset
    from src.utils import PROJECT_ROOT, DATA_DIR, compute_binary_metrics
except ImportError:
    from inference import WildfireInferenceEngine
    from dataset import FEATURE_NAMES, FEATURE_METADATA, WildfireTFRecordDataset
    from utils import PROJECT_ROOT, DATA_DIR, compute_binary_metrics

# Initialize FastAPI App
app = FastAPI(
    title="Wildfire Spread Prediction API",
    description="Operational REST microservice for next-day wildfire spread forecasting using spatial deep learning.",
    version="1.0.0"
)

# Enable CORS for frontend web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Inference Engine Singleton
inference_engine: Optional[WildfireInferenceEngine] = None

def get_engine() -> WildfireInferenceEngine:
    global inference_engine
    if inference_engine is None:
        try:
            inference_engine = WildfireInferenceEngine(backend="onnx")
        except Exception as e:
            # Fallback to PyTorch backend if ONNX fails
            inference_engine = WildfireInferenceEngine(backend="pytorch")
    return inference_engine

# --- Pydantic Request & Response Schemas ---

class CentroidShift(BaseModel):
    eastward_km: float = Field(..., description="East-West spread propagation displacement in kilometers.")
    northward_km: float = Field(..., description="North-South spread propagation displacement in kilometers.")

class WildfirePredictionResponse(BaseModel):
    risk_level: str = Field(..., description="Hazard tier: Low, Moderate, High, or Extreme.")
    decision_threshold: float = Field(..., description="Classification threshold applied for active burn delineation.")
    predicted_burn_pixels: int = Field(..., description="Total pixel count classified as active burning front.")
    predicted_burn_area_km2: float = Field(..., description="Estimated fire spread area in square kilometers.")
    prior_burn_pixels: int = Field(..., description="Fire pixel count at Day T.")
    expansion_ratio: float = Field(..., description="Ratio of predicted spread to initial fire front.")
    centroid_shift_km: CentroidShift = Field(..., description="Directional displacement of burning front.")
    max_probability: float = Field(..., description="Maximum burn probability in the 64x64 tile.")
    backend_used: str = Field(..., description="Runtime backend engine: onnx or pytorch.")
    probability_matrix_base64: str = Field(..., description="Base64-encoded float32 numpy array [64, 64] of probabilities.")
    ground_truth_metrics: Optional[Dict[str, float]] = Field(None, description="Comparative accuracy metrics if ground truth is available.")

class RasterTileRequest(BaseModel):
    decision_threshold: Optional[float] = Field(0.35, ge=0.01, le=0.99, description="Custom decision threshold.")
    raw_tensor_base64: Optional[str] = Field(None, description="Base64 encoded float32 numpy array of shape [12, 64, 64].")
    channel_dict: Optional[Dict[str, List[List[float]]]] = Field(
        None,
        description="Dictionary mapping band names (elevation, NDVI, etc.) to 64x64 float matrices."
    )

class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str
    num_features: int

# --- API Endpoints ---

@app.get("/", tags=["System"])
def root():
    """Service status and API documentation entry point."""
    return {
        "service": "Wildfire Spread Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "healthcheck": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
def healthcheck():
    """Verify inference engine readiness and model configuration."""
    engine = get_engine()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "backend": engine.backend,
        "num_features": len(FEATURE_NAMES)
    }

@app.get("/metadata/channels", tags=["Metadata"])
def get_channel_metadata():
    """Retrieve catalog of all 12 input bands, physical units, and environmental categories."""
    return {
        "input_channels": FEATURE_NAMES,
        "resolution_km_per_pixel": 1.0,
        "tile_dimensions": [64, 64],
        "channel_metadata": FEATURE_METADATA
    }

@app.post("/predict/tile", response_model=WildfirePredictionResponse, tags=["Inference"])
def predict_raster_tile(request: RasterTileRequest):
    """
    Execute wildfire spread prediction on a user-submitted 12x64x64 raster tile.
    Accepts either base64-encoded numpy float32 buffer or a nested JSON dictionary.
    """
    engine = get_engine()

    if request.raw_tensor_base64:
        try:
            byte_data = base64.b64decode(request.raw_tensor_base64)
            array_data = np.frombuffer(byte_data, dtype=np.float32).reshape(12, 64, 64)
            tile_input = array_data
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to decode raw_tensor_base64: {str(e)}")
    elif request.channel_dict:
        try:
            tile_input = {}
            for name in FEATURE_NAMES:
                if name not in request.channel_dict:
                    raise ValueError(f"Missing required band '{name}' in channel_dict")
                tile_input[name] = np.array(request.channel_dict[name], dtype=np.float32)
                if tile_input[name].shape != (64, 64):
                    raise ValueError(f"Band '{name}' must have shape (64, 64), got {tile_input[name].shape}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid channel_dict: {str(e)}")
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must supply either 'raw_tensor_base64' or 'channel_dict'."
        )

    # Run inference
    result = engine.predict_tile(tile_input, decision_threshold=request.decision_threshold)
    prob_b64 = base64.b64encode(result["probability_map"].astype(np.float32).tobytes()).decode("utf-8")

    return WildfirePredictionResponse(
        risk_level=result["risk_level"],
        decision_threshold=result["decision_threshold"],
        predicted_burn_pixels=result["predicted_burn_pixels"],
        predicted_burn_area_km2=result["predicted_burn_area_km2"],
        prior_burn_pixels=result["prior_burn_pixels"],
        expansion_ratio=result["expansion_ratio"],
        centroid_shift_km=CentroidShift(**result["centroid_shift_km"]),
        max_probability=result["max_probability"],
        backend_used=result["backend_used"],
        probability_matrix_base64=prob_b64,
        ground_truth_metrics=None
    )

@app.get("/predict/sample/{tile_index}", response_model=WildfirePredictionResponse, tags=["Inference"])
def predict_sample_tile(
    tile_index: int,
    decision_threshold: float = Query(0.35, ge=0.01, le=0.99, description="Active burn decision threshold")
):
    """
    Retrieve and evaluate a historical validation wildfire event by index.
    Calculates accuracy metrics against the ground-truth Day T+1 satellite observation.
    """
    engine = get_engine()
    eval_files = sorted(glob.glob(str(DATA_DIR / "*eval_*.tfrecord")))
    if not eval_files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No evaluation TFRecord files found on system.")

    ds = WildfireTFRecordDataset(eval_files, normalize=False)
    
    selected_sample = None
    for idx, (inputs, target, mask) in enumerate(ds):
        if idx == tile_index:
            selected_sample = (inputs.numpy(), target.numpy()[0], mask.numpy()[0])
            break
        if idx > tile_index:
            break

    if selected_sample is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tile index {tile_index} out of range.")

    inputs_np, target_np, mask_np = selected_sample
    result = engine.predict_tile(inputs_np, decision_threshold=decision_threshold)

    # Compute comparative accuracy metrics against ground truth
    valid_mask = mask_np.astype(bool)
    y_true = target_np[valid_mask]
    y_prob = result["probability_map"][valid_mask]

    gt_metrics = None
    if len(y_true) > 0:
        raw_metrics = compute_binary_metrics(y_true, y_prob, threshold=decision_threshold)
        gt_metrics = {
            "ground_truth_burn_pixels": int(np.sum(y_true == 1)),
            "f1_score": float(raw_metrics.get("f1", 0.0)),
            "precision": float(raw_metrics.get("precision", 0.0)),
            "recall": float(raw_metrics.get("recall", 0.0)),
            "iou_jaccard": float(raw_metrics.get("iou", 0.0)),
            "dice_score": float(raw_metrics.get("dice", 0.0))
        }

    prob_b64 = base64.b64encode(result["probability_map"].astype(np.float32).tobytes()).decode("utf-8")

    return WildfirePredictionResponse(
        risk_level=result["risk_level"],
        decision_threshold=result["decision_threshold"],
        predicted_burn_pixels=result["predicted_burn_pixels"],
        predicted_burn_area_km2=result["predicted_burn_area_km2"],
        prior_burn_pixels=result["prior_burn_pixels"],
        expansion_ratio=result["expansion_ratio"],
        centroid_shift_km=CentroidShift(**result["centroid_shift_km"]),
        max_probability=result["max_probability"],
        backend_used=result["backend_used"],
        probability_matrix_base64=prob_b64,
        ground_truth_metrics=gt_metrics
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
