"""
Export trained PyTorch WildfireUNet model checkpoint to optimized ONNX format.
Validates numerical parity between PyTorch and ONNX Runtime and benchmarks inference latency.
"""

import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

try:
    from src.models.unet import WildfireUNet
    from src.utils import PROJECT_ROOT, MODELS_DIR
except ImportError:
    from models.unet import WildfireUNet
    from utils import PROJECT_ROOT, MODELS_DIR

class WildfireInferenceWrapper(nn.Module):
    """Wraps WildfireUNet with Sigmoid activation for direct probability output in ONNX."""
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core_model = core_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.core_model(x)
        return torch.sigmoid(logits)

def export_to_onnx(checkpoint_path=None, output_path=None):
    checkpoint_path = Path(checkpoint_path or (MODELS_DIR / "wildfire_unet_best.pth"))
    output_path = Path(output_path or (MODELS_DIR / "wildfire_unet.onnx"))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PyTorch model checkpoint not found at: {checkpoint_path}")

    print(f"Loading PyTorch checkpoint from: {checkpoint_path}")
    model = WildfireUNet(in_channels=12, out_channels=1, base_channels=32)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    inference_model = WildfireInferenceWrapper(model)
    inference_model.eval()

    dummy_input = torch.randn(1, 12, 64, 64, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting model to ONNX at: {output_path}")

    torch.onnx.export(
        inference_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_raster"],
        output_names=["burn_probability"],
        dynamic_axes={
            "input_raster": {0: "batch_size"},
            "burn_probability": {0: "batch_size"}
        }
    )

    # Verify ONNX model integrity
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified successfully.")

    # Numerical validation with ONNX Runtime
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    
    with torch.no_grad():
        pytorch_output = inference_model(dummy_input).numpy()

    onnx_inputs = {"input_raster": dummy_input.numpy()}
    onnx_output = session.run(["burn_probability"], onnx_inputs)[0]

    max_abs_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"Maximum absolute difference between PyTorch and ONNX Runtime: {max_abs_diff:.6e}")
    assert max_abs_diff < 1e-4, f"Numerical parity validation failed with diff: {max_abs_diff}"
    print("Numerical parity check passed: PyTorch and ONNX Runtime outputs match within tolerance.")

    # Latency benchmark
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(["burn_probability"], onnx_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = np.mean(latencies[5:])  # Skip warmups
    print(f"ONNX Runtime CPU Inference Latency: {avg_latency:.2f} ms per 64x64 tile")
    print(f"Successfully exported and validated ONNX model: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch WildfireUNet to ONNX format.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch .pth checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Destination path for .onnx model")
    args = parser.parse_args()

    export_to_onnx(checkpoint_path=args.checkpoint, output_path=args.output)
