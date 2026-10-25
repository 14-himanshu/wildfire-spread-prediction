"""
Wildfire Intelligence Command Center: Interactive GIS Dashboard.
Designed for both emergency response operators and non-technical stakeholders.
Provides plain-English situation reports, intuitive composite forecast maps,
interactive sensitivity sliders, and physical spread driver breakdowns.
"""

import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
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
    page_title="Wildfire Command Center | 24-Hour AI Spread Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Executive Command Center Theme (Zero emojis, modern typography)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    .executive-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box-val {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-box-lbl {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    .legend-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-top: 0.8rem;
        font-size: 0.9rem;
        color: #334155;
    }
    .legend-tag {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
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
    # --- Header Section ---
    st.markdown('<div class="main-title">Wildfire Intelligence Command Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Operational 24-hour satellite forecast system powered by spatial deep learning (PyTorch U-Net & ONNX Runtime). '
        'Transforms 12 atmospheric, topographic, and satellite fuel layers into actionable fire perimeter predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    engine = load_engine()

    # --- Sidebar Controls (Humanized for Everyone) ---
    st.sidebar.header("Scenario & Alert Settings")

    curated_scenarios = {
        "Incident Alpha: Canyon Wind Surge (Fast Eastward Spread)": 28,
        "Incident Bravo: Mountain Ridge Escalation (Steep Terrain)": 42,
        "Incident Charlie: Valley Expansion (High Fuel Density)": 105,
        "Incident Delta: Desert Gale Shift (Sudden Velocity Spike)": 7,
        "Custom Incident Explorer (Select Shard Index)": -1
    }

    selected_scenario_name = st.sidebar.selectbox(
        "Select Historical Wildfire Incident",
        list(curated_scenarios.keys()),
        help="Select real historical wildfire incidents recorded by NASA and USGS satellites."
    )

    if curated_scenarios[selected_scenario_name] == -1:
        tile_index = st.sidebar.number_input("Incident Shard Index (0 to 500)", min_value=0, max_value=500, value=28, step=1)
    else:
        tile_index = curated_scenarios[selected_scenario_name]

    # Humanized sensitivity slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert Sensitivity (Decision Threshold)")
    
    threshold = st.sidebar.slider(
        "Confidence Cutoff for Fire Front Delineation",
        min_value=0.15,
        max_value=0.75,
        value=0.35,
        step=0.05,
        help=(
            "Lower threshold (0.20): Maximum Caution — catches every potential spark; recommended for evacuation buffer zones.\n"
            "Balanced (0.35): Standard operational default.\n"
            "Higher threshold (0.55+): Conservative — focuses strictly on the most intense active core blaze."
        )
    )

    if threshold <= 0.25:
        threshold_mode = "High Caution (Evacuation Planning Mode)"
        mode_desc = "Prioritizes public safety: flags all potential embers and spread corridors."
    elif threshold >= 0.50:
        threshold_mode = "Conservative (Core Blaze Suppression Mode)"
        mode_desc = "Focuses exclusively on confirmed, highest-intensity flame walls."
    else:
        threshold_mode = "Balanced (Standard Operational Forecast)"
        mode_desc = "Optimal balance between capturing spread fronts and avoiding false alarms."

    st.sidebar.caption(f"**Current Mode:** {threshold_mode}\n\n*{mode_desc}*")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Inference Backend:** `{engine.backend.upper()}`")
    st.sidebar.markdown("**Area Monitored:** `64 x 64 km (4,096 km²)`")
    st.sidebar.markdown("**Sensor Grid Resolution:** `1.0 km / pixel`")

    # --- Load Data Sample ---
    sample_data = load_validation_sample(tile_index)
    if sample_data is None:
        st.error(f"Could not load incident tile at index {tile_index}. Verify raw TFRecord data paths.")
        return

    inputs_np, target_np, mask_np = sample_data

    # Execute Prediction
    result = engine.predict_tile(inputs_np, decision_threshold=threshold)
    prob_map = result["probability_map"]
    pred_binary = result["binary_mask"]
    risk_level = result["risk_level"]
    prior_burn_km2 = result["prior_burn_pixels"]
    pred_burn_km2 = result["predicted_burn_pixels"]
    expansion_ratio = result["expansion_ratio"]
    dx_km = result["centroid_shift_km"]["eastward_km"]
    dy_km = result["centroid_shift_km"]["northward_km"]

    # Calculate Ground Truth Metrics
    valid_mask = mask_np.astype(bool)
    y_true = target_np[valid_mask]
    y_prob = prob_map[valid_mask]
    actual_burn_km2 = int(np.sum(y_true == 1))
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)

    # Physical Wind Drivers
    avg_wind_speed = float(np.mean(inputs_np[8])) * 3.6  # m/s to km/h
    avg_elevation = float(np.mean(inputs_np[0]))

    # Direction determination in plain English
    dir_str = []
    if dy_km > 0.5: dir_str.append("North")
    elif dy_km < -0.5: dir_str.append("South")
    if dx_km > 0.5: dir_str.append("East")
    elif dx_km < -0.5: dir_str.append("West")
    spread_direction = "-".join(dir_str) if dir_str else "Stationary expansion"

    # --- Section 1: Executive Situation Report (The Plain-English Summary) ---
    risk_colors = {
        "Extreme": {"bg": "#fee2e2", "text": "#991b1b", "badge": "#dc2626", "rec": "IMMEDIATE EVACUATION WARNING: Extreme fire behavior expected across entire forward corridor."},
        "High": {"bg": "#ffedd5", "text": "#9a3412", "badge": "#ea580c", "rec": "EVACUATION ADVISORY: Fire spreading rapidly; pre-evacuation notices recommended for downwind communities."},
        "Moderate": {"bg": "#fef3c7", "text": "#92400e", "badge": "#d97706", "rec": "MONITORING ADVISORY: Active suppression recommended along flank containment lines."},
        "Low": {"bg": "#dcfce7", "text": "#166534", "badge": "#16a34a", "rec": "CONTAINMENT STABLE: Minimal outward spread detected; mop-up operations recommended."}
    }
    r_cfg = risk_colors.get(risk_level, risk_colors["Moderate"])

    st.markdown(f"""
    <div class="executive-card" style="border-left: 6px solid {r_cfg['badge']};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <div>
                <span class="metric-badge" style="background-color: {r_cfg['badge']}; color: #ffffff;">Threat Level: {risk_level.upper()}</span>
                <span style="margin-left: 10px; color: #64748b; font-size: 0.9rem; font-weight: 600;">24-Hour Automated Hazard Assessment</span>
            </div>
            <div style="font-size: 0.9rem; color: #475569;">
                Monitoring Zone: <strong>64 km x 64 km</strong> | Elevation: <strong>{avg_elevation:.0f} m</strong>
            </div>
        </div>
        <div style="font-size: 1.15rem; font-weight: 700; color: {r_cfg['text']}; margin-bottom: 0.5rem;">
            {r_cfg['rec']}
        </div>
        <div style="font-size: 1.0rem; color: #334155; line-height: 1.6;">
            <strong>Situation Breakdown:</strong> Yesterday, active fire perimeter covered <strong>{prior_burn_km2} km²</strong>.
            Over the next 24 hours, the AI model predicts the fire will expand to <strong>{pred_burn_km2} km²</strong>
            (an expansion factor of <strong>{expansion_ratio:.1f}x</strong>), pushing primarily toward the <strong>{spread_direction}</strong>
            at sustained atmospheric wind speeds of <strong>{avg_wind_speed:.1f} km/h</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Operational Metric Cards ---
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val" style="color: {r_cfg['badge']};">{risk_level}</div>
            <div class="metric-box-lbl">Threat Category</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val">{prior_burn_km2} <span style="font-size: 1rem; font-weight: 500;">km²</span></div>
            <div class="metric-box-lbl">Yesterday's Fire</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val" style="color: #b91c1c;">{pred_burn_km2} <span style="font-size: 1rem; font-weight: 500;">km²</span></div>
            <div class="metric-box-lbl">Tomorrow's Forecast</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val">{expansion_ratio:.1f}x</div>
            <div class="metric-box-lbl">Growth Multiple</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val">{spread_direction}</div>
            <div class="metric-box-lbl">Primary Spread Axis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Section 2: Tabbed Exploratory Interfaces ---
    tab_map, tab_physics, tab_benchmark, tab_channels = st.tabs([
        "Composite Forecast Map (The Big Picture)",
        "Physical Fire Drivers Explained",
        "Why Deep Learning? (Model Comparison)",
        "Raw Satellite Band Explorer"
    ])

    # === TAB 1: Composite Forecast Map ===
    with tab_map:
        st.subheader("24-Hour AI Wildfire Spread Projection")
        st.write(
            "This composite map overlays the **starting fire from yesterday (gold boundary)** with the **AI's 24-hour predicted burn zone (red glow)** "
            "and the **actual verified satellite observation (white dashed line)** across the mountain terrain and wind vectors."
        )

        elev = inputs_np[0]
        prev_fire = inputs_np[11]
        wind_dir = inputs_np[5]
        wind_spd = inputs_np[8]
        u_w, v_w = decompose_wind(wind_spd, wind_dir)

        fig_comp, ax_comp = plt.subplots(figsize=(12, 7.5), facecolor="#ffffff")

        # 1. Background Elevation Topography
        im_topo = ax_comp.imshow(elev, cmap="terrain", interpolation="bicubic", alpha=0.75)

        # 2. Wind Vector Arrows
        step = 5
        y_g, x_g = np.mgrid[0:64:step, 0:64:step]
        ax_comp.quiver(
            x_g, y_g,
            u_w[::step, ::step], v_w[::step, ::step],
            color="#1e293b", scale=45, width=0.0035, alpha=0.6,
            label="Wind Velocity & Direction"
        )

        # 3. AI Predicted Spread Heatmap (Orange-Red Glow)
        # Create a custom colormap that is transparent for low probabilities
        red_colors = [(1, 0.4, 0, 0.0), (1, 0.2, 0, 0.65), (0.8, 0, 0, 0.9)]
        spread_cmap = LinearSegmentedColormap.from_list("spread_glow", red_colors)
        ax_comp.imshow(prob_map, cmap=spread_cmap, interpolation="bicubic", vmin=0.1, vmax=1.0)

        # 4. Starting Fire Perimeter (Gold Solid Contour)
        if prev_fire.sum() > 0:
            ax_comp.contour(prev_fire, levels=[0.5], colors=["#fbbf24"], linewidths=2.5)

        # 5. Verified Ground Truth Satellite Perimeter (White Dashed Contour)
        if target_np.sum() > 0:
            ax_comp.contour(target_np, levels=[0.5], colors=["#ffffff"], linewidths=2.0, linestyles="dashed")

        ax_comp.set_title("Terrain Elevation with Wind Quiver, Initial Fire, and AI Spread Projection", fontsize=13, fontweight="bold", pad=12)
        ax_comp.set_xlabel("East-West Distance (km)", fontsize=10)
        ax_comp.set_ylabel("North-South Distance (km)", fontsize=10)

        # Colorbar for elevation
        cbar = fig_comp.colorbar(im_topo, ax=ax_comp, fraction=0.035, pad=0.03)
        cbar.set_label("Elevation (meters above sea level)", fontsize=10)

        st.pyplot(fig_comp)
        plt.close()

        # Plain-English Legend Guide
        st.markdown("""
        <div class="legend-box">
            <strong>How to read this map:</strong><br>
            • <span class="legend-tag" style="background: #fbbf24; border: 1px solid #d97706;"></span> <strong>Gold Solid Line:</strong> Where the fire started yesterday (Day T active perimeter).<br>
            • <span class="legend-tag" style="background: #ef4444;"></span> <strong>Red/Orange Glowing Area:</strong> Where the AI forecasts the fire will spread over the next 24 hours.<br>
            • <span class="legend-tag" style="background: #ffffff; border: 1px dashed #0f172a;"></span> <strong>White Dashed Line:</strong> Actual satellite ground truth observed the next day (verification).<br>
            • <strong>Black Arrows:</strong> Atmospheric wind direction and speed (longer arrows = stronger wind pushing the front).<br>
            • <strong>Green to Brown Terrain:</strong> Mountain topography (fire accelerates uphill along steep brown ridges).
        </div>
        """, unsafe_allow_html=True)

    # === TAB 2: Physical Fire Drivers Explained ===
    with tab_physics:
        st.subheader("The Three Physical Forces Driving Wildfire Spread")
        st.write("Wildfires do not spread randomly. They are driven by fluid mechanics, mountain topography, and fuel dryness:")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # 1. Topography & Wind
            st.markdown("#### 1. Topography & Wind (The Forward Engine)")
            st.write(
                "Flames naturally climb mountains up to **4 times faster** than flat ground because heat rising from the fire preheats "
                "the trees and bushes above it before the flames even touch them. Strong winds push hot embers kilometers ahead (spot fires)."
            )
            fig_p1, ax_p1 = plt.subplots(figsize=(7, 4.5), facecolor="#ffffff")
            im_p1 = ax_p1.imshow(elev, cmap="terrain")
            ax_p1.quiver(x_g, y_g, u_w[::step, ::step], v_w[::step, ::step], color="black", scale=45, width=0.004)
            ax_p1.set_title(f"Elevation Model + Wind Quiver (Avg Speed: {avg_wind_speed:.1f} km/h)", fontsize=11, fontweight="bold")
            ax_p1.axis("off")
            fig_p1.colorbar(im_p1, ax=ax_p1, fraction=0.046, pad=0.04, label="Meters")
            st.pyplot(fig_p1)
            plt.close()

        with col_p2:
            # 2. Vegetation Fuel Density (NDVI)
            st.markdown("#### 2. Vegetation Fuel Load (The Combustion Material)")
            st.write(
                "Satellite NDVI measures plant density. Thicker forests (bright green) contain dense burnable biomass. "
                "Combined with drought dryness, dense vegetation turns into an explosive fuel reservoir."
            )
            ndvi = inputs_np[2]
            fig_p2, ax_p2 = plt.subplots(figsize=(7, 4.5), facecolor="#ffffff")
            im_p2 = ax_p2.imshow(ndvi, cmap="YlGn")
            ax_p2.set_title("Vegetation Density Index (NDVI: Green = Dense Fuel)", fontsize=11, fontweight="bold")
            ax_p2.axis("off")
            fig_p2.colorbar(im_p2, ax=ax_p2, fraction=0.046, pad=0.04)
            st.pyplot(fig_p2)
            plt.close()

        col_p3, col_p4 = st.columns(2)

        with col_p3:
            # 3. Drought Severity (PDSI)
            st.markdown("#### 3. Drought Index (Fuel Dryness)")
            st.write(
                "The Palmer Drought Severity Index (PDSI) measures soil and fuel moisture deficits. "
                "Negative values indicate prolonged drought where dead timber and brush ignite instantaneously."
            )
            pdsi = inputs_np[1]
            fig_p3, ax_p3 = plt.subplots(figsize=(7, 4.5), facecolor="#ffffff")
            im_p3 = ax_p3.imshow(pdsi, cmap="RdYlBu")
            ax_p3.set_title("Drought Severity (Red = Severe Dryness)", fontsize=11, fontweight="bold")
            ax_p3.axis("off")
            fig_p3.colorbar(im_p3, ax=ax_p3, fraction=0.046, pad=0.04)
            st.pyplot(fig_p3)
            plt.close()

        with col_p4:
            # 4. Energy Release Component (ERC)
            st.markdown("#### 4. Energy Release Component (Fire Danger)")
            st.write(
                "National Fire Danger Rating Index (ERC) reflects the potential heat release per unit area in the flaming zone. "
                "High ERC means fires will burn hotter, produce towering smoke columns, and resist containment."
            )
            erc = inputs_np[9]
            fig_p4, ax_p4 = plt.subplots(figsize=(7, 4.5), facecolor="#ffffff")
            im_p4 = ax_p4.imshow(erc, cmap="Oranges")
            ax_p4.set_title("Fire Danger Energy Release Potential (ERC)", fontsize=11, fontweight="bold")
            ax_p4.axis("off")
            fig_p4.colorbar(im_p4, ax=ax_p4, fraction=0.046, pad=0.04)
            st.pyplot(fig_p4)
            plt.close()

    # === TAB 3: Why Deep Learning? ===
    with tab_benchmark:
        st.subheader("Why Standard Machine Learning Fails and Deep Learning Wins")
        st.write(
            "Traditional machine learning treats each pixel independently. It fails to understand that fire is a continuous "
            "fluid phenomenon governed by neighboring trees and airflow. The PyTorch U-Net uses convolutional receptive fields "
            "to model spatial diffusion."
        )

        fig_b, axes_b = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#ffffff")

        # 1. Ground Truth
        axes_b[0].imshow(target_np, cmap="hot", vmin=0, vmax=1)
        axes_b[0].set_title(f"Actual Satellite Reality (Day T+1)\n(Real Burned Area: {actual_burn_km2} km²)", fontsize=12, fontweight="bold")
        axes_b[0].axis("off")

        # 2. U-Net Continuous Prediction
        im_un = axes_b[1].imshow(prob_map, cmap="inferno", vmin=0, vmax=1)
        axes_b[1].set_title(f"Deep Learning U-Net (Ours)\n(Smooth Natural Spread Front)", fontsize=12, fontweight="bold")
        axes_b[1].axis("off")
        fig_b.colorbar(im_un, ax=axes_b[1], fraction=0.046, pad=0.04)

        # 3. Decision Mask vs Reality Overlay
        axes_b[2].imshow(pred_binary, cmap="Blues", vmin=0, vmax=1)
        if target_np.sum() > 0:
            axes_b[2].contour(target_np, levels=[0.5], colors=["red"], linewidths=2.0)
        axes_b[2].set_title("Forecast Accuracy Overlay\n(Blue = AI Prediction, Red Outline = Reality)", fontsize=12, fontweight="bold")
        axes_b[2].axis("off")

        st.pyplot(fig_b)
        plt.close()

        # Quantitative Benchmark Table
        st.markdown("#### Scientific Benchmark on 1.25 Million Held-Out Test Pixels")
        comp_data = {
            "Capability / Metric": [
                "Rare Fire Detection Accuracy (PR-AUC)",
                "Fire Catch Rate (Recall)",
                "Visual Realism & Continuity",
                "Inference Speed per 64x64 km Tile",
                "Handling of 99.7% Empty Land Imbalance"
            ],
            "Traditional ML (LightGBM)": [
                "0.0599",
                "27.41% (Misses 72% of fires)",
                "Blocky, pixelated square artifacts",
                "4.98 ms",
                "Biased toward zero fire"
            ],
            "Spatial Deep Learning U-Net (Ours)": [
                "0.1255 (+109.5% Improvement)",
                "39.94% (+45.7% Higher Catch Rate)",
                "Smooth, physical diffusion wave",
                "0.12 ms GPU / 5.14 ms ONNX CPU",
                "Focal-Tversky Loss penalizes missed fires"
            ]
        }
        st.table(comp_data)

    # === TAB 4: Raw Satellite Band Explorer ===
    with tab_channels:
        st.subheader("Raw Multi-Spectral Satellite Channel Explorer")
        st.write("Inspect all 12 raw satellite and meteorological bands before automated preprocessing:")

        selected_ch = st.selectbox("Select Band to Inspect", FEATURE_NAMES, index=0)
        ch_i = FEATURE_NAMES.index(selected_ch)
        meta_info = FEATURE_METADATA.get(selected_ch, {})

        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            fig_single, ax_s = plt.subplots(figsize=(8, 6), facecolor="#ffffff")
            im_s = ax_s.imshow(inputs_np[ch_i], cmap="magma")
            ax_s.set_title(f"{meta_info.get('title', selected_ch)} ({meta_info.get('unit', '')})", fontsize=12, fontweight="bold")
            ax_s.axis("off")
            fig_single.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
            st.pyplot(fig_single)
            plt.close()

        with col_c2:
            st.markdown(f"### {meta_info.get('title', selected_ch)}")
            st.markdown(f"- **Physical Unit:** `{meta_info.get('unit', 'N/A')}`")
            st.markdown(f"- **Category:** `{meta_info.get('category', 'Atmospheric')}`")
            st.markdown(f"- **Observed Tile Min:** `{np.min(inputs_np[ch_i]):.2f}`")
            st.markdown(f"- **Observed Tile Max:** `{np.max(inputs_np[ch_i]):.2f}`")
            st.markdown(f"- **Observed Tile Mean:** `{np.mean(inputs_np[ch_i]):.2f}`")

    # --- Section 3: Non-Technical FAQ ---
    st.markdown("---")
    with st.expander("Frequently Asked Questions for Reviewers & Decision-Makers"):
        st.markdown("""
        **1. How does the AI make these predictions?**
        > The system uses a deep convolutional neural network (PyTorch U-Net) trained on thousands of historical wildfires.
        > It looks at the starting fire perimeter and analyzes 11 surrounding environmental variables (wind direction, mountain slope, fuel dryness, temperature)
        > to model the physical diffusion wave of the fire front over a 24-hour window.

        **2. Why does the threat level change when I move the slider?**
        > The slider controls **decision caution**. When planning an evacuation, you want **High Caution (low threshold)** so you catch every potential spark.
        > When dispatching limited firefighting tanker aircraft, you want **Conservative (high threshold)** to drop retardant directly on the confirmed core blaze.

        **3. What does 1.0 km / pixel mean in real life?**
        > Each tiny square on the map represents exactly **1 square kilometer (approximately 247 acres)** of terrain.
        > The entire map represents a massive **64 km by 64 km region (over 1 million acres)**.
        """)

if __name__ == "__main__":
    main()
