"""
Spatial neighborhood feature engineering pipeline for Wildfire Spread Tabular Baseline.
Transforms 2D spatial raster grids into tabular rows with engineered spatial lag features:
- Topographic slope gradients (Sobel filter)
- Spatial neighborhood fire density (3x3 and 5x5 filters)
- Local vegetation fuel density
- Zonal and meridional wind vector decomposition (u, v)
- Upwind directional fire alignment
"""

import numpy as np
from scipy.ndimage import uniform_filter, sobel
from utils import decompose_wind
from dataset import FEATURE_NAMES

TABULAR_FEATURE_NAMES = [
    # Direct point features
    'elevation', 'pdsi', 'NDVI', 'pr', 'sph', 
    'tmmn', 'tmmx', 'vs', 'erc', 'population', 'PrevFireMask',
    # Decomposed wind vectors
    'wind_u', 'wind_v',
    # Topographic gradients (slope)
    'slope_x', 'slope_y', 'slope_magnitude', 'elevation_diff_3x3',
    # Spatial neighborhood aggregates
    'prev_fire_3x3_mean', 'prev_fire_5x5_mean',
    'ndvi_3x3_mean', 'erc_3x3_mean'
]

def extract_tabular_features_from_tile(tile_tensor, return_grid=False):
    """
    Args:
        tile_tensor: numpy array of shape [12, 64, 64] (raw or normalized)
        return_grid: if True, returns [num_features, 64, 64], else returns flattened [4096, num_features]
    Returns:
        Feature matrix for all 4,096 pixels
    """
    feature_dict = {name: tile_tensor[idx] for idx, name in enumerate(FEATURE_NAMES)}
    
    elevation = feature_dict['elevation']
    prev_fire = feature_dict['PrevFireMask']
    ndvi = feature_dict['NDVI']
    erc = feature_dict['erc']
    vs = feature_dict['vs']
    th = feature_dict['th']

    # 1. Wind vector decomposition
    u, v = decompose_wind(vs, th)

    # 2. Topographic slope gradients via Sobel filters
    slope_x = sobel(elevation, axis=1) / 8.0
    slope_y = sobel(elevation, axis=0) / 8.0
    slope_mag = np.sqrt(slope_x**2 + slope_y**2)
    
    # Elevation convexity (is this pixel elevated above its local surroundings?)
    elev_mean_3x3 = uniform_filter(elevation, size=3, mode='reflect')
    elevation_diff_3x3 = elevation - elev_mean_3x3

    # 3. Spatial neighborhood fire expansion potential
    prev_fire_3x3 = uniform_filter(prev_fire, size=3, mode='reflect')
    prev_fire_5x5 = uniform_filter(prev_fire, size=5, mode='reflect')
    
    # 4. Spatial fuel & energy averages
    ndvi_3x3 = uniform_filter(ndvi, size=3, mode='reflect')
    erc_3x3 = uniform_filter(erc, size=3, mode='reflect')

    # Assemble all 21 spatial channels
    channel_list = [
        feature_dict['elevation'], feature_dict['pdsi'], feature_dict['NDVI'],
        feature_dict['pr'], feature_dict['sph'], feature_dict['tmmn'],
        feature_dict['tmmx'], feature_dict['vs'], feature_dict['erc'],
        feature_dict['population'], feature_dict['PrevFireMask'],
        u, v,
        slope_x, slope_y, slope_mag, elevation_diff_3x3,
        prev_fire_3x3, prev_fire_5x5,
        ndvi_3x3, erc_3x3
    ]

    grid_features = np.stack(channel_list, axis=0) # [21, 64, 64]

    if return_grid:
        return grid_features

    # Flatten to [4096, 21]
    flattened = grid_features.reshape(len(TABULAR_FEATURE_NAMES), -1).T
    return flattened

if __name__ == "__main__":
    dummy_tile = np.random.randn(12, 64, 64).astype(np.float32)
    # Give prev fire some binary values
    dummy_tile[FEATURE_NAMES.index('PrevFireMask')] = (dummy_tile[0] > 1.0).astype(np.float32)
    dummy_tile[FEATURE_NAMES.index('th')] = np.random.uniform(0, 360, (64, 64)).astype(np.float32)
    dummy_tile[FEATURE_NAMES.index('vs')] = np.random.uniform(1, 15, (64, 64)).astype(np.float32)

    feats = extract_tabular_features_from_tile(dummy_tile)
    print("Tabular feature engineering test:")
    print(f"  Extracted matrix shape: {feats.shape} (Expected: [4096, {len(TABULAR_FEATURE_NAMES)}])")
    print(f"  Feature count: {len(TABULAR_FEATURE_NAMES)}")
    print("  Feature list:", TABULAR_FEATURE_NAMES)
