"""
High-level production inference engine for Wildfire Spread Prediction.
Supports both ONNX Runtime (default, low-latency CPU) and PyTorch backends.
Applies automated physical anomaly cleaning, dynamic standardization,
and calculates operational wildfire risk indicators.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import json
import numpy as np

try:
    from src.dataset import clean_and_normalize_tile, FEATURE_NAMES, FEATURE_METADATA
    from src.utils import PROJECT_ROOT, MODELS_DIR, CONFIGS_DIR
except ImportError:
    from dataset import clean_and_normalize_tile, FEATURE_NAMES, FEATURE_METADATA
    from utils import PROJECT_ROOT, MODELS_DIR, CONFIGS_DIR

class WildfireInferenceEngine:
    """
    Production inference engine for predicting next-day wildfire spread.
    
    Accepts raw multi-spectral environmental tiles [12, 64, 64], cleans physical
    anomalies, applies calibrated feature normalization, and executes deep learning
    forward passes using ONNX Runtime or PyTorch.
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        backend: str = "onnx",
        default_threshold: float = 0.35
    ):
        self.backend = backend.lower()
        self.default_threshold = default_threshold

        # Load normalization scalers
        scaler_file = Path(scaler_path or (CONFIGS_DIR / "scaler_params.json"))
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler parameters not found at: {scaler_file}")
        with open(scaler_file, "r") as f:
            self.scalers = json.load(f)

        # Initialize backend
        if self.backend == "onnx":
            self.model_path = Path(model_path or (MODELS_DIR / "wildfire_unet.onnx"))
            if not self.model_path.exists():
                # Fallback to PyTorch if ONNX file is not yet exported
                pth_path = MODELS_DIR / "wildfire_unet_best.pth"
                if pth_path.exists():
                    self.backend = "pytorch"
                    self.model_path = pth_path
                else:
                    raise FileNotFoundError(f"Model file not found at {self.model_path} or {pth_path}")

        if self.backend == "onnx":
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.model = None
        else:
            import torch
            from models.unet import WildfireUNet
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = WildfireUNet(in_channels=12, out_channels=1, base_channels=32)
            ckpt = torch.load(self.model_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.session = None

    def predict_tile(
        self,
        raw_tile: Union[np.ndarray, Dict[str, np.ndarray]],
        decision_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run inference on a single 64x64 multi-spectral tile.

        Parameters
        ----------
        raw_tile : np.ndarray or dict
            Multi-spectral array of shape [12, 64, 64] or dictionary mapping channel names to [64, 64] arrays.
        decision_threshold : float, optional
            Binary classification threshold for active burn delineation. Defaults to self.default_threshold (0.35).

        Returns
        -------
        dict containing:
            - probability_map: 64x64 numpy array of burn probabilities [0.0, 1.0]
            - binary_mask: 64x64 binary prediction {0, 1}
            - risk_level: Categorical hazard assessment ('Low', 'Moderate', 'High', 'Extreme')
            - predicted_burn_pixels: Total pixel count classified as active fire
            - predicted_burn_area_km2: Predicted burn area in square kilometers
            - prior_burn_pixels: Fire pixel count at Day T
            - expansion_ratio: Spread expansion ratio relative to prior fire
            - centroid_shift_km: Directional shift vector (dx, dy) in kilometers
            - max_probability: Peak burn probability in tile
        """
        threshold = decision_threshold if decision_threshold is not None else self.default_threshold

        # Convert dict input to array if needed
        if isinstance(raw_tile, dict):
            tile_dict = raw_tile
        elif isinstance(raw_tile, np.ndarray):
            if raw_tile.shape != (12, 64, 64):
                raise ValueError(f"Expected input array shape (12, 64, 64), got {raw_tile.shape}")
            tile_dict = {name: raw_tile[i] for i, name in enumerate(FEATURE_NAMES)}
        else:
            raise TypeError(f"Unsupported tile type: {type(raw_tile)}")

        # Automated Anomaly Defense and Standardization
        cleaned_inputs, _, _ = clean_and_normalize_tile(tile_dict, scalers=self.scalers, normalize=True)
        input_tensor_np = np.expand_dims(cleaned_inputs, axis=0).astype(np.float32)  # [1, 12, 64, 64]

        # Model Forward Pass
        if self.backend == "onnx":
            prob_output = self.session.run([self.output_name], {self.input_name: input_tensor_np})[0]
            probability_map = prob_output[0, 0]  # [64, 64]
        else:
            import torch
            with torch.no_grad():
                tensor_in = torch.from_numpy(input_tensor_np).to(self.device)
                logits = self.model(tensor_in)
                probs = torch.sigmoid(logits)
                probability_map = probs.cpu().numpy()[0, 0]

        # Operational Metrics Calculation
        binary_mask = (probability_map >= threshold).astype(np.uint8)
        predicted_burn_pixels = int(np.sum(binary_mask))
        predicted_burn_area_km2 = float(predicted_burn_pixels * 1.0)  # 1 km x 1 km pixel resolution

        # Prior fire reference
        prior_fire_raw = tile_dict.get('PrevFireMask', np.zeros((64, 64)))
        prior_fire_mask = (prior_fire_raw > 0.5).astype(np.uint8)
        prior_burn_pixels = int(np.sum(prior_fire_mask))

        expansion_ratio = float(predicted_burn_pixels / max(1, prior_burn_pixels))

        # Centroid Shift (Propagation Direction)
        if predicted_burn_pixels > 0 and prior_burn_pixels > 0:
            pred_y, pred_x = np.where(binary_mask == 1)
            prior_y, prior_x = np.where(prior_fire_mask == 1)
            dx = float(np.mean(pred_x) - np.mean(prior_x))
            dy = float(np.mean(pred_y) - np.mean(prior_y))
            centroid_shift_km = {"eastward_km": round(dx, 2), "northward_km": round(-dy, 2)}
        else:
            centroid_shift_km = {"eastward_km": 0.0, "northward_km": 0.0}

        # Risk Classification Tier
        max_prob = float(np.max(probability_map))
        if predicted_burn_pixels > 200 or max_prob > 0.85:
            risk_level = "Extreme"
        elif predicted_burn_pixels > 75 or max_prob > 0.65:
            risk_level = "High"
        elif predicted_burn_pixels > 15 or max_prob > 0.40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "probability_map": probability_map,
            "binary_mask": binary_mask,
            "risk_level": risk_level,
            "decision_threshold": threshold,
            "predicted_burn_pixels": predicted_burn_pixels,
            "predicted_burn_area_km2": predicted_burn_area_km2,
            "prior_burn_pixels": prior_burn_pixels,
            "expansion_ratio": round(expansion_ratio, 2),
            "centroid_shift_km": centroid_shift_km,
            "max_probability": round(max_prob, 4),
            "backend_used": self.backend
        }
