# Data Dictionary: Next Day Wildfire Spread Dataset

Every raster band, feature transformation, and physical anomaly in the Next Day Wildfire Spread dataset, documented before and after preprocessing.

All figures below were measured on the official dataset split (18,545 spatial tiles, each $64 \times 64\text{ km}$ at $1\text{ km/pixel}$ resolution) using `src/compute_stats.py` and validated with `tests/test_pipeline.py`.

---

## 1. Source & Attribution

| Field | Value |
| :--- | :--- |
| **Dataset Name** | Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote Sensing Data |
| **Provider / Paper** | Google Research — Huot et al. (2022), arXiv:[2112.02447](https://arxiv.org/abs/2112.02447) |
| **Data Repository** | Kaggle / Google Earth Engine — [Next Day Wildfire Spread](https://www.kaggle.com/datasets/google/next-day-wildfire-spread) |
| **License** | **CC BY 4.0** (Open Access, reuse permitted with attribution) |
| **File Format** | Multi-shard Protocol Buffer TFRecords (`.tfrecord`) |
| **Total Samples** | **18,545 spatial tiles** ($64 \times 64 = 4,096$ pixels per tile) |
| **Partitioning** | Train: 14,979 tiles (15 shards) \| Validation: 1,877 tiles (2 shards) \| Test: 1,689 tiles (2 shards) |
| **Temporal Horizon**| Given environmental state at Day $T$, predict active fire front at Day $T+1$ (24-hour advance forecast) |

---

## 2. Multimodal Input Channels (12 Bands)

Each sample is a 2D spatial raster grid of size $12 \times 64 \times 64$.

| # | Channel | Physical Meaning | Native Unit | Raw Min / Max | Training Mean | Training Std | Sentinel / Anomaly | Preprocessing Transformation |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `elevation` | Topographic digital elevation model (DEM) | meters | -8.0 to 4,193.0 | 1,049.08 m | 876.46 m | None | Z-score standardized: $(x - \mu)/\sigma$ |
| 2 | `pdsi` | Palmer Drought Severity Index (dryness) | dimensionless | -17.70 to 28.69 | -1.11 | 2.39 | None | Z-score standardized: $(x - \mu)/\sigma$ |
| 3 | `NDVI` | Normalized Difference Vegetation Index (fuel) | scaled ($10^4$) | -9,531 to 9,951 | 5,260.30 | 2,196.00 | None | Z-score standardized: $(x - \mu)/\sigma$ |
| 4 | `pr` | Daily accumulated precipitation | mm | -5.71 to 42.95 | 0.28 mm | 1.50 mm | Negative nodata sentinels | Clipped: $\max(0.0, x)$, then standardized |
| 5 | `sph` | Specific humidity (air moisture content) | kg/kg | -0.02 to 0.05 | 0.006 kg/kg | 0.004 kg/kg | Negative nodata sentinels | Clipped: $\max(0.0, x)$, then standardized |
| 6 | `th` | Wind direction azimuth (origin of wind) | degrees ($0^\circ - 360^\circ$) | -505,870 to 37,736 | 198.25° | 70.51° | Corrupt reprojection sentinels | Modulo wrapped: $x \pmod{360^\circ}$, vector decomposed |
| 7 | `tmmn` | Minimum daily temperature | Kelvin | 0.0 to 331.5 K | 282.52 K | 17.13 K | 0.0 K station dropouts | Imputed if $\le 200\text{ K}$ with regional median (282.5 K) |
| 8 | `tmmx` | Maximum daily temperature | Kelvin | 0.0 to 366.9 K | 298.14 K | 17.99 K | 0.0 K station dropouts | Imputed if $\le 200\text{ K}$ with regional median (298.1 K) |
| 9 | `vs` | Wind velocity at 10m height | m/s | -43.74 to 91.09 | 3.58 m/s | 1.27 m/s | Negative nodata sentinels | Clipped: $\max(0.0, x)$, then standardized |
| 10 | `erc` | Energy Release Component (NFDRS fire danger)| dimensionless | -672.97 to 294.84 | 57.13 | 26.15 | Negative nodata sentinels | Clipped: $\max(0.0, x)$, then standardized |
| 11 | `population`| Human population density | persons / km² | 0.0 to 14,087.4 | 21.29 | 167.54 | Heavy positive skew | Log-transformed: $\log(1 + \max(0, x))$, then standardized |
| 12 | `PrevFireMask`| Active fire detection mask at Day $T$ | binary $\{0, 1\}$ | -1.0 to 1.0 | 0.002 | 0.046 | `-1.0` (unobserved/clouds) | Clipped: $[0.0, 1.0]$ (unobserved treated as unburned) |

---

## 3. Target Channel (Ground Truth)

| Channel | Physical Meaning | Native Unit | Valid Values | Class Balance | Operational Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FireMask`** | Active fire detection mask at Day $T+1$ | binary $\{0, 1\}$ | `0.0`: Unburned<br>`1.0`: Active Fire<br>`-1.0`: Cloud / Unobserved | Active fire is **$<0.3\%$** of observable pixels | **Dynamic Masking:** Pixels with `-1.0` generate `valid_mask = 0.0`. All loss functions and metrics ignore unobserved pixels. |

---

## 4. Engineered Spatial Neighborhood Features (Tabular Baseline)

In addition to the 12 direct channels, the tabular baseline (`src/features.py`) derives **9 spatial interaction features**:

| Engineered Feature | Derivation / Formula | Physical Mechanism |
| :--- | :--- | :--- |
| `wind_u` | $u = -\text{vs} \cdot \sin(\text{rad}(\text{th}))$ | Zonal wind vector (West-to-East propagation pressure) |
| `wind_v` | $v = -\text{vs} \cdot \cos(\text{rad}(\text{th}))$ | Meridional wind vector (South-to-North propagation pressure) |
| `slope_x` | $\text{Sobel}_x(\text{elevation}) / 8.0$ | Terrain gradient along the East-West axis |
| `slope_y` | $\text{Sobel}_y(\text{elevation}) / 8.0$ | Terrain gradient along the North-South axis |
| `slope_magnitude` | $\sqrt{\text{slope}_x^2 + \text{slope}_y^2}$ | Overall terrain steepness (fire climbs steeper slopes faster) |
| `elevation_diff_3x3` | $\text{elevation} - \text{mean}_{3\times3}(\text{elevation})$ | Local convexity (positive = ridge, negative = ravine/valley) |
| `prev_fire_3x3_mean` | $\text{mean}_{3\times3}(\text{PrevFireMask})$ | Local fire density in immediate 8-neighbor adjacency |
| `prev_fire_5x5_mean` | $\text{mean}_{5\times5}(\text{PrevFireMask})$ | Medium-range fire pressure within 2 km radius |
| `ndvi_3x3_mean` | $\text{mean}_{3\times3}(\text{NDVI})$ | Surrounding fuel availability |
| `erc_3x3_mean` | $\text{mean}_{3\times3}(\text{erc})$ | Local potential energy release |

---

## 5. Data Quality Audits & Upstream Preprocessing

A diagnostic scan across **6.14 million pixels (1,500 tiles)** identified 5 critical real-world artifacts:

1. **Target Cloud Sentinels (`FireMask == -1.0` in 32.7% of tiles):**
   * *Problem:* Satellite sensors cannot detect thermal anomalies beneath thick cloud decks.
   * *Solution:* Dynamic loss masking (`valid_mask = raw_target >= 0.0`). The loss denominator only counts observable pixels:
     $$\mathcal{L} = \frac{\sum (\mathcal{L}_{\text{pixel}} \cdot \text{valid\_mask})}{\sum \text{valid\_mask} + \epsilon}$$
2. **Temperature Dropouts (`0.0 K` in 41 tiles):**
   * *Problem:* Absolute zero ($0.0\text{ K} = -273.15^\circ\text{C}$) indicates a missing weather station interpolation node.
   * *Solution:* Imputed with the tile spatial median or climatological baseline ($282.5\text{ K}$ min / $298.1\text{ K}$ max).
3. **Reprojection Wind Angle Outliers:**
   * *Problem:* Negative sentinels (e.g. $-505,870^\circ$) from GDAL reprojection voids.
   * *Solution:* Modulo arithmetic $x \pmod{360^\circ}$ maps angles to valid circular intervals $[0^\circ, 360^\circ]$.
4. **Physical Non-Negativity:**
   * *Problem:* Numerical precision underflow caused negative values in rain (`pr`), wind speed (`vs`), and fire danger (`erc`).
   * *Solution:* Clamped to $\ge 0.0$.
5. **Class Imbalance Distribution:**
   * Active burning pixels represent **0.1% - 0.5%** of all terrain pixels.
   * Addressed via **Masked Focal-Tversky Loss** ($\alpha=0.3, \beta=0.7, \gamma=0.75$), penalizing missed fires $2.33\times$ higher than false alarms.
