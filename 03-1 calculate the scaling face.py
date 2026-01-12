import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# --- File Paths (Updated to Relative Paths) ---
original_path = "original_clouds"        # Folder containing original point clouds
processed_path = "processed_clouds"      # Folder containing processed/downsampled clouds
length_file = "stem_lengths_summary.txt" # Input: Summary of calculated stem lengths
output_file = "original_scale_lengths.txt" # Output: Rescaled summary file

def get_base_filename(filename):
    """Extracts the base name and removes downsampling suffixes."""
    base = os.path.splitext(filename)[0]
    if ' - Cloud_downsampled' in base:
        base = base.split(' - Cloud_downsampled')[0]
    return base

# Get file lists
original_files = [f for f in os.listdir(original_path) if f.endswith(('.txt', '.xyz'))]
processed_files = [f for f in os.listdir(processed_path) if f.endswith(('.txt', '.xyz'))]

original_base_files = {get_base_filename(f): f for f in original_files}
processed_base_files = {get_base_filename(f): f for f in processed_files}
common_base_files = set(original_base_files.keys()) & set(processed_base_files.keys())

if not common_base_files:
    raise ValueError("No matching point cloud files found in both directories!")

# ============ Calculate Scaling Factors via Nearest Neighbor Distance ============
scale_factors = {}
for base_name in common_base_files:
    orig_file = original_base_files[base_name]
    proc_file = processed_base_files[base_name]

    orig_full_path = os.path.join(original_path, orig_file)
    proc_full_path = os.path.join(processed_path, proc_file)

    try:
        # Load XYZ data (skipping extra columns)
        orig_data = np.loadtxt(orig_full_path, usecols=(0, 1, 2), ndmin=2)
        proc_data = np.loadtxt(proc_full_path, usecols=(0, 1, 2), ndmin=2)
    except Exception as e:
        print(f"Failed to read {base_name}: {e}")
        continue

    if len(orig_data) < 2 or len(proc_data) < 2:
        print(f"Too few points, skipping {base_name}")
        continue

    # Extract Y-column (index 1) to find the base/bottom point
    y_orig = orig_data[:, 1]
    y_proc = proc_data[:, 1]

    # Find the point with minimum Y value
    idx_orig_min = np.argmin(y_orig)
    idx_proc_min = np.argmin(y_proc)
    p_orig_min = orig_data[idx_orig_min]
    p_proc_min = proc_data[idx_proc_min]

    # Build KDTree to find nearest neighbor (excluding itself)
    tree_orig = KDTree(orig_data)
    tree_proc = KDTree(proc_data)

    # k=2 because the first neighbor is the point itself
    dist_orig, _ = tree_orig.query(p_orig_min, k=2)  
    dist_proc, _ = tree_proc.query(p_proc_min, k=2)

    if len(dist_orig) < 2 or len(dist_proc) < 2:
        print(f"Insufficient neighbors, skipping {base_name}")
        continue

    # Use the distance to the nearest neighbor (index 1)
    dist_orig_nearest = dist_orig[1]
    dist_proc_nearest = dist_proc[1]

    if dist_proc_nearest == 0:
        print(f"Processed distance is zero, skipping {base_name}")
        continue

    # Calculate scale factor: Original / Processed
    scale_factor = dist_orig_nearest / dist_proc_nearest
    scale_factors[base_name] = round(scale_factor, 3)
    
    print(f"{base_name}: Orig_Dist={dist_orig_nearest:.3f}, Proc_Dist={dist_proc_nearest:.3f}, Scale={scale_factor:.3f}")

# ============ Apply Scaling to Length Data ============
try:
    # Read the summary file
    length_df = pd.read_csv(length_file, sep='\s+', header=None)
    length_df.columns = ['filename', 'label'] + [f'length_{i}' for i in range(1, 7)]
except Exception as e:
    raise ValueError(f"Could not read stem length file: {e}")

# Convert columns to numeric (coerce errors to NaN)
for col in [f'length_{i}' for i in range(1, 7)]:
    length_df[col] = pd.to_numeric(length_df[col], errors='coerce')

# Map base filename for matching
length_df['base_filename'] = length_df['filename'].apply(get_base_filename)

# Apply scaling factor to each length column
for i in range(1, 7):
    col = f'length_{i}'
    length_df[col] = length_df.apply(
        lambda row: round(row[col] * scale_factors.get(row['base_filename'], 1.0), 3)
        if pd.notna(row[col]) and row['base_filename'] in scale_factors
        else row[col], axis=1
    )

# Clean up temporary column
length_df = length_df.drop(columns=['base_filename'])

# Save to output file
length_df.to_csv(output_file, sep='\t', index=False, float_format='%.3f')

print(f"\nSaved rescaled results to: {output_file}")
print(f"Successfully processed {len(scale_factors)} files.")