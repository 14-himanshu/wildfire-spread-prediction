# Next Day Wildfire Spread Prediction via Spatial Deep Learning & AML

An end-to-end multimodal machine learning system that predicts next-day wildfire propagation from remote sensing imagery, atmospheric weather grids, and digital elevation models.

## 📊 Dataset Specifications
- **Source:** Google Research / Kaggle Next Day Wildfire Spread
- **File Format:** Multi-shard TFRecords (`.tfrecord`)
- **Total Samples:** 18,545 tiles (64 x 64 km at 1 km/pixel resolution)
  - **Train:** 14,979 samples (15 shards: `train_00` to `train_14`)
  - **Eval (Val):** 1,877 samples (2 shards: `eval_00` to `eval_01`)
  - **Test:** 1,689 samples (2 shards: `test_00` to `test_01`)
- **Memory Footprint:** ~3.95 GB uncompressed

### 🛰️ Input Channels (12 Bands)
1. `elevation` (DEM elevation in meters)
2. `pdsi` (Palmer Drought Severity Index)
3. `NDVI` (Normalized Difference Vegetation Index / fuel density)
4. `pr` (Precipitation in mm)
5. `sph` (Specific humidity in kg/kg)
6. `th` (Wind direction azimuth in degrees: 0° - 360°)
7. `tmmn` (Daily minimum temperature in Kelvin)
8. `tmmx` (Daily maximum temperature in Kelvin)
9. `vs` (Wind velocity / speed at 10m in m/s)
10. `erc` (Energy Release Component / NFDRS fire danger index)
11. `population` (Human population density per km²)
12. `PrevFireMask` (Active fire binary mask on Day T)

### 🎯 Target Channel
- `FireMask` (Active fire binary mask on Day T+1, shape: 64x64)
- **Class Imbalance:** Active burn pixels account for only ~0.1% - 0.5% of total land area.

## 📁 Repository Structure
```text
wildfire-spread-prediction/
├── data/
│   └── raw/                       # Symlink to /Users/himanshupandey/Downloads/archive
├── src/
│   ├── dataset.py                 # Pure-PyTorch streaming TFRecord data loader
│   ├── baseline_lgbm.py           # Spatial neighborhood feature extraction & LightGBM
│   └── unet_model.py              # PyTorch U-Net with Focal-Tversky Loss
├── notebooks/                     # Exploratory Data Analysis & visual inspections
├── configs/                       # Hyperparameter & channel scaling configurations
├── tests/                         # PyTest validation suite
└── README.md
```

## 🚀 Quick Start
Run the streaming PyTorch data loader test:
```bash
/Users/himanshupandey/Downloads/.venv/bin/python src/dataset.py
```
