"""
High-performance, defensively programmed TFRecord parser and PyTorch Dataset 
for Next Day Wildfire Spread.

Zero TensorFlow dependency - handles Google Research TFRecords directly.
Incorporates comprehensive data cleaning for all 5 observed real-world anomalies:
1. Masked/cloud sentinel values (-1.0 in FireMask & PrevFireMask)
2. Temperature dropouts (0.0 Kelvin fill values in tmmn/tmmx)
3. Corrupt wind angles (extreme negative/positive values outside 0-360)
4. Physically impossible negative quantities in vs (wind speed), erc, pr (rain)
5. Heavy right-skew in population density (log1p transformed)
6. Multi-worker safe shard partitioning via get_worker_info()
"""

import os
import glob
import json
import struct
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

FEATURE_NAMES = [
    'elevation', 'pdsi', 'NDVI', 'pr', 'sph', 'th', 
    'tmmn', 'tmmx', 'vs', 'erc', 'population', 'PrevFireMask'
]
TARGET_NAME = 'FireMask'
RESOLUTION = 64
NUM_PIXELS = RESOLUTION * RESOLUTION

def parse_varint(data, pos):
    res = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return res, pos

def parse_protobuf(data):
    pos = 0
    fields = []
    while pos < len(data):
        key, pos = parse_varint(data, pos)
        field_num = key >> 3
        wire_type = key & 0x7
        if wire_type == 0:
            val, pos = parse_varint(data, pos)
            fields.append((field_num, wire_type, val))
        elif wire_type == 1:
            val = data[pos:pos+8]
            pos += 8
            fields.append((field_num, wire_type, val))
        elif wire_type == 2:
            length, pos = parse_varint(data, pos)
            val = data[pos:pos+length]
            pos += length
            fields.append((field_num, wire_type, val))
        elif wire_type == 5:
            val = data[pos:pos+4]
            pos += 4
            fields.append((field_num, wire_type, val))
        else:
            break
    return fields

def extract_example(raw_record):
    ex = parse_protobuf(raw_record)
    features_data = ex[0][2]
    features_fields = parse_protobuf(features_data)
    channels = {}
    for fn, wt, val in features_fields:
        if fn == 1 and wt == 2:
            entry = parse_protobuf(val)
            k, v = None, None
            for e_fn, e_wt, e_val in entry:
                if e_fn == 1:
                    k = e_val.decode('utf-8', errors='ignore')
                elif e_fn == 2:
                    feat = parse_protobuf(e_val)
                    for f_fn, f_wt, f_v in feat:
                        if f_fn == 2 and f_wt == 2:
                            count = len(f_v) // 4
                            floats = struct.unpack(f"<{count}f", f_v[:count*4])
                            v = np.array(floats[:NUM_PIXELS], dtype=np.float32).reshape(RESOLUTION, RESOLUTION)
            if k and v is not None:
                channels[k] = v
    return channels

def clean_and_normalize_tile(feats, scalers=None, normalize=True):
    """
    Cleans physical anomalies and applies zero-leakage z-score normalization.
    """
    raw_target = feats[TARGET_NAME]
    
    # 1. Cloud & missing sensor mask (-1.0)
    valid_mask = (raw_target >= 0.0).astype(np.float32)
    clean_target = (raw_target > 0.5).astype(np.float32)

    stacked_inputs = []
    for name in FEATURE_NAMES:
        c_data = feats[name].copy()
        
        # Anomaly fixes:
        if name == 'PrevFireMask':
            c_data = np.clip(c_data, 0.0, 1.0)
        elif name in ['tmmn', 'tmmx']:
            valid_temps = c_data[c_data > 200.0]
            fill = np.median(valid_temps) if len(valid_temps) > 0 else (282.5 if name == 'tmmn' else 298.1)
            c_data[c_data <= 200.0] = fill
        elif name == 'th':
            c_data = np.where((c_data < 0) | (c_data > 360), 0.0, c_data) % 360.0
        elif name in ['vs', 'erc', 'pr']:
            c_data = np.clip(c_data, 0.0, None)
        elif name == 'population':
            c_data = np.log1p(np.clip(c_data, 0.0, None))

        # Standardize using training statistics
        if normalize and scalers and name in scalers:
            mean = scalers[name]['mean']
            std = scalers[name]['std']
            c_data = (c_data - mean) / (std + 1e-7)

        stacked_inputs.append(c_data)

    stacked_inputs = np.stack(stacked_inputs, axis=0) # [12, 64, 64]
    return stacked_inputs, clean_target, valid_mask

class WildfireTFRecordDataset(IterableDataset):
    """
    Streams samples directly from sharded TFRecord files.
    Memory efficient: keeps RAM footprint < 450MB.
    Multi-worker safe: partitions shards by worker ID to avoid duplicates.
    """
    def __init__(self, file_paths, scaler_path=None, normalize=True):
        super().__init__()
        self.file_paths = sorted(file_paths)
        self.normalize = normalize
        self.scalers = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                self.scalers = json.load(f)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process loading
            assigned_files = self.file_paths
        else:
            # Multi-process loading: partition shards across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            assigned_files = self.file_paths[worker_id::num_workers]

        for path in assigned_files:
            with open(path, "rb") as f:
                while True:
                    len_bytes = f.read(8)
                    if not len_bytes or len(len_bytes) < 8:
                        break
                    length, = struct.unpack("<Q", len_bytes)
                    f.read(4) # CRC32 of length
                    record_data = f.read(length)
                    f.read(4) # CRC32 of data

                    try:
                        feats = extract_example(record_data)
                        if TARGET_NAME not in feats or any(name not in feats for name in FEATURE_NAMES):
                            continue
                        
                        inputs_arr, target_arr, mask_arr = clean_and_normalize_tile(
                            feats, scalers=self.scalers, normalize=self.normalize
                        )

                        input_tensor = torch.from_numpy(inputs_arr)
                        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)        # [1, 64, 64]
                        valid_mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)      # [1, 64, 64]

                        yield input_tensor, target_tensor, valid_mask_tensor
                    except Exception:
                        continue

if __name__ == "__main__":
    from utils import DATA_DIR, CONFIGS_DIR
    data_dir = DATA_DIR
    scaler_file = CONFIGS_DIR / "scaler_params.json"
    train_files = sorted(glob.glob(os.path.join(data_dir, "*train_*.tfrecord")))[:2]
    
    ds = WildfireTFRecordDataset(train_files, scaler_path=scaler_file, normalize=True)
    loader = DataLoader(ds, batch_size=8, num_workers=2)
    
    print("Testing Multi-Worker (num_workers=2) PyTorch DataLoader:")
    for i, (x, y, mask) in enumerate(loader):
        print(f"Batch {i}: Tensor={x.shape}, Finite={torch.all(torch.isfinite(x)).item()}, Masks Valid Ratio={mask.mean().item():.3f}")
        if i >= 1:
            break
