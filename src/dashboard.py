"""
Interactive GIS Web Dashboard for Wildfire Spread Prediction.
Built with Streamlit and Matplotlib.
Allows operators and researchers to inspect multi-spectral environmental rasters,
visualize physical spread fronts, evaluate spatial deep learning predictions,
and analyze operational hazard tiers.
"""

import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

try:
    from src.inference import WildfireInferenceEngine
    from src.dataset import FEATURE_NAMES, FEATURE_METADATA, WildfireTFRecordDataset
    from src.utils import PROJECT_ROOT, DATA_DIR, decompose_wind, compute_binary_metrics
except ImportError:
    from inference import WildfireInferenceEngine
    from dataset import FEATURE_NAMES, FEATURE_METADATA, WildfireTFRecordDataset
    from utils import PROJECT_ROOT, DATA_DIR, decompose_wind, compute_binary_metrics

# Streamlit Page Configuration
st.set_page_config(
    page_title="Wildfire Spread Prediction | GIS Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Zero emojis, modern publication typography)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    """Cache inference engine instance for fast multi-tile querying."""
    try:
        return WildfireInferenceEngine(backend="onnx")
    except Exception:
        return WildfireInferenceEngine(backend="pytorch")

@st.cache_data
def load_validation_sample(tile_idx: int):
    """Cache and load a single validation tile from held-out TFRecords."""
    eval_files = sorted(glob.glob(str(DATA_DIR / "*eval_*.tfrecord")))
    if not eval_files:
        return None
    ds = WildfireTFRecordDataset(eval_files, normalize=False)
    for i, (inputs, target, mask) in enumerate(ds):
        if i == tile_idx:
            return inputs.numpy(), target.numpy()[0], mask.numpy()[0]
        if i > tile_idx:
            break
    return None

def main():
    st.markdown('<div class="main-title">Next Day Wildfire Spread Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Operational 24-hour spatial forecasting using multi-spectral satellite imagery and deep learning (PyTorch U-Net / ONNX Runtime).</div>',
        unsafe_allow_html=True
    )

    engine = load_engine()

    # --- Sidebar Controls ---
    st.sidebar.header("Configuration & Controls")
    
    st.sidebar.markdown(f"**Runtime Backend:** `{engine.backend.upper()}`")
    st.sidebar.markdown("**Spatial Resolution:** `1.0 km / pixel (64x64 km)`")

    # Interesting historical wildfire events from validation shard
    preset_tiles = {
        "Incident #1 (Active Fire Front - Shard Tile 28)": 28,
        "Incident #2 (Ridge Propagation - Shard Tile 42)": 42,
        "Incident #3 (Complex Topography - Shard Tile 105)": 105,
        "Incident #4 (High Wind Spread - Shard Tile 7)": 7,
        "Custom Tile Index": -1
    }
    selected_preset = st.sidebar.selectbox("Select Historical Wildfire Event", list(preset_tiles.keys()))

    if preset_tiles[selected_preset] == -1:
        tile_index = st.sidebar.number_input("Enter Tile Index (0 to 500)", min_value=0, max_value=500, value=28, step=1)
    else:
        tile_index = preset_tiles[selected_preset]

    threshold = st.sidebar.slider(
        "Decision Threshold (Burn Delineation)",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Probability cutoff for classifying a 1 km2 pixel as actively burning."
    )

    channel_to_inspect = st.sidebar.selectbox(
        "Inspect Multi-Spectral Environmental Band",
        FEATURE_NAMES,
        index=0,
        help="Select any of the 12 input satellite and atmospheric raster channels."
    )

    # --- Load Data Sample ---
    sample_data = load_validation_sample(tile_index)
    if sample_data is None:
        st.error(f"Could not load validation tile at index {tile_index}. Verify raw TFRecord data paths.")
        return

    inputs_np, target_np, mask_np = sample_data

    # Execute Prediction
    result = engine.predict_tile(inputs_np, decision_threshold=threshold)
    prob_map = result["probability_map"]
    pred_binary = result["binary_mask"]
    risk_level = result["risk_level"]

    # Calculate Ground Truth Evaluation Metrics
    valid_mask = mask_np.astype(bool)
    y_true = target_np[valid_mask]
    y_prob = prob_map[valid_mask]
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)

    # --- Operational Metric Cards ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    risk_colors = {
        "Extreme": "#dc2626",
        "High": "#ea580c",
        "Moderate": "#d97706",
        "Low": "#16a34a"
    }
    risk_color = risk_colors.get(risk_level, "#475569")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hazard Level</div>
            <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Burn Area</div>
            <div class="metric-value">{result['predicted_burn_area_km2']:.0f} km²</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prior Fire (Day T)</div>
            <div class="metric-value">{result['prior_burn_pixels']} km²</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expansion Ratio</div>
            <div class="metric-value">{result['expansion_ratio']:.2f}x</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        dx = result["centroid_shift_km"]["eastward_km"]
        dy = result["centroid_shift_km"]["northward_km"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Spread Vector (E, N)</div>
            <div class="metric-value">{dx:+.1f}, {dy:+.1f} km</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Section 1: Wildfire Spread Comparison Heatmaps ---
    st.subheader("Spatial Wildfire Spread Forecasting")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.25)

    # 1. Day T Fire
    prev_fire = inputs_np[11]
    axes[0].imshow(prev_fire, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title(f"Day T: Initial Fire Boundary\n(Burned: {int(prev_fire.sum())} km²)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Day T+1 Ground Truth
    gt_burned = int(target_np[valid_mask].sum())
    axes[1].imshow(target_np, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
    axes[1].set_title(f"Day T+1: Satellite Observation\n(Burned: {gt_burned} km²)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Continuous Probability Heatmap
    im_prob = axes[2].imshow(prob_map, cmap="inferno", interpolation="bicubic", vmin=0, vmax=1)
    axes[2].set_title(f"Spatial U-Net Probability Map\n(Max: {result['max_probability']:.2f})", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04, label="Burn Probability")

    # 4. Binary Delineation & Contour
    axes[3].imshow(pred_binary, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    if prev_fire.sum() > 0:
        axes[3].contour(prev_fire, levels=[0.5], colors="red", linewidths=1.8, linestyles="dashed")
    axes[3].set_title(f"Active Front Delineation\n(Threshold = {threshold:.2f}, Area = {result['predicted_burn_pixels']} km²)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    st.pyplot(fig)
    plt.close()

    # --- Section 2: Environmental Multi-Spectral Layers ---
    st.subheader("Environmental & Atmospheric Drivers")
    col_env1, col_env2 = st.columns([1, 1])

    with col_env1:
        # Display Selected Band
        ch_idx = FEATURE_NAMES.index(channel_to_inspect)
        ch_meta = FEATURE_METADATA.get(channel_to_inspect, {})
        ch_data = inputs_np[ch_idx]

        fig_ch, ax_ch = plt.subplots(figsize=(8, 6), facecolor="#ffffff")
        im_ch = ax_ch.imshow(ch_data, cmap="viridis", interpolation="nearest")
        ax_ch.set_title(
            f"{ch_meta.get('title', channel_to_inspect)} ({ch_meta.get('unit', '')})\nCategory: {ch_meta.get('category', 'Driver')}",
            fontsize=12,
            fontweight="bold"
        )
        ax_ch.axis("off")
        fig_ch.colorbar(im_ch, ax=ax_ch, fraction=0.046, pad=0.04)
        st.pyplot(fig_ch)
        plt.close()

    with col_env2:
        # Topography & Wind Vectors Overlaid
        elev_data = inputs_np[0]
        wind_dir = inputs_np[5]
        wind_spd = inputs_np[8]

        u_wind, v_wind = decompose_wind(wind_spd, wind_dir)

        fig_wind, ax_wind = plt.subplots(figsize=(8, 6), facecolor="#ffffff")
        im_topo = ax_wind.imshow(elev_data, cmap="terrain", interpolation="bicubic")
        
        # Subsample quiver arrows for clean visualization
        step = 6
        y_grid, x_grid = np.mgrid[0:64:step, 0:64:step]
        ax_wind.quiver(
            x_grid, y_grid,
            u_wind[::step, ::step], v_wind[::step, ::step],
            color="black", scale=40, width=0.005, alpha=0.8
        )
        if prev_fire.sum() > 0:
            ax_wind.contour(prev_fire, levels=[0.5], colors="red", linewidths=2.0)

        ax_wind.set_title("Terrain Elevation with Wind Velocity Vectors (Quiver)", fontsize=12, fontweight="bold")
        ax_wind.axis("off")
        fig_wind.colorbar(im_topo, ax=ax_wind, fraction=0.046, pad=0.04, label="Elevation (m)")
        st.pyplot(fig_wind)
        plt.close()

    # --- Section 3: Ground Truth Evaluation Metrics ---
    if gt_burned > 0:
        st.subheader("Incident Evaluation vs Ground Truth")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Recall (Fire Catch Rate)", f"{metrics['recall'] * 100:.1f}%")
        col_m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        col_m3.metric("F1-Score", f"{metrics['f1']:.3f}")
        col_m4.metric("IoU (Jaccard)", f"{metrics['iou']:.3f}")
        col_m5.metric("Dice Coefficient", f"{metrics['dice']:.3f}")

if __name__ == "__main__":
    main()
