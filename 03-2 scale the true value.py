import pandas as pd
import numpy as np
import os
from warnings import warn
import time

# ==================== Configuration ====================
# Base directory path (Changed to relative path)
BASE_DIR = 'stem_experiment_data'

# Input file paths
STEM_LENGTH_PATH = os.path.join(BASE_DIR, 'stem_summary.txt')
SCALING_INFO_PATH = os.path.join(BASE_DIR, 'scaling_relations.txt')

# Output file path
OUTPUT_SCALED_PATH = os.path.join(BASE_DIR, 'stem_summary_original_scale.txt')

# Column definitions for input files
# stem_summary.txt (Assuming last 5 columns are Method_1 to Method_5)
STEM_COLS = ['file_basename', 'label_id', 'Method_1', 'Method_2', 'Method_3', 'Method_4', 'Method_5']
# scaling_relations.txt
SCALING_COLS = ['file_basename', 'label_id', 'GT_raw', 'GT_scaled']
# =======================================================

def calculate_file_scaling_ratio(filepath, scaling_cols):
    """
    Loads the scaling info file and calculates a uniform scaling ratio for each file.
    scaling_ratio = sum(GT_scaled) / sum(GT_raw) for each file_basename.
    """
    print(f"1. Loading scaling information from: {filepath}")

    try:
        # Assuming file is space/tab-separated without header
        df_scale = pd.read_csv(
            filepath, sep=r'\s+', header=None, names=scaling_cols, engine='python'
        )
    except Exception as e:
        warn(f"Failed to load scaling file: {e}")
        return None, None

    # Force conversion to numeric
    for col in ['GT_raw', 'GT_scaled']:
        df_scale[col] = pd.to_numeric(df_scale[col], errors='coerce')

    # Drop rows containing NaN
    df_scale = df_scale.dropna(subset=['GT_raw', 'GT_scaled'])

    # Extract required GT_scaled values for final output
    df_gt = df_scale[['file_basename', 'label_id', 'GT_scaled']].copy()

    # Group by filename to calculate uniform scaling ratio
    grouped = df_scale.groupby('file_basename').agg(
        GT_raw_sum=('GT_raw', 'sum'),
        GT_scaled_sum=('GT_scaled', 'sum')
    ).reset_index()

    # Calculate scaling ratio (GT_scaled_sum / GT_raw_sum)
    grouped['scaling_ratio'] = np.where(
        grouped['GT_raw_sum'] != 0,
        grouped['GT_scaled_sum'] / grouped['GT_raw_sum'],
        1.0  # Default to 1.0 to avoid division by zero
    )

    print(f"   -> Successfully calculated scaling ratios for {len(grouped)} files.")
    return grouped[['file_basename', 'scaling_ratio']], df_gt


def load_and_scale_stem_data(stem_path, scaling_ratios, df_gt):
    """
    Loads stem length summary data, applies scaling ratios, and merges with GT_scaled.
    """
    print(f"2. Loading stem length summary from: {stem_path}")

    try:
        df_stem = pd.read_csv(
            stem_path, sep=r'\s+', header=None, names=STEM_COLS, engine='python'
        )
    except Exception as e:
        warn(f"Failed to load stem summary file: {e}")
        return None

    # Identify method columns for scaling
    method_cols = STEM_COLS[2:]

    # Force all method columns to numeric type
    for col in method_cols:
        df_stem[col] = pd.to_numeric(df_stem[col], errors='coerce')

    # Merge stem data with scaling ratios based on file_basename
    df_merged = df_stem.merge(
        scaling_ratios,
        on='file_basename',
        how='inner'
    )

    print(f"   -> Successfully merged {len(df_merged)} rows of data.")

    # Apply scaling ratio to all measurement results
    scaled_method_cols = []
    for col in method_cols:
        new_col_name = f'{col}_Scaled'
        df_merged[new_col_name] = df_merged[col] * df_merged['scaling_ratio']
        scaled_method_cols.append(new_col_name)

    # Ensure label_id types match before final merging
    df_gt['label_id'] = df_gt['label_id'].astype(df_merged['label_id'].dtype)

    df_final = df_merged.merge(
        df_gt,
        on=['file_basename', 'label_id'],
        how='inner'
    )

    # Final output column order
    final_cols = ['file_basename', 'label_id', 'GT_scaled'] + scaled_method_cols

    return df_final[final_cols]


def main():
    start_time = time.time()

    # 1. Calculate file-level scaling ratios and get GT_scaled
    scaling_ratios, df_gt_scaled = calculate_file_scaling_ratio(SCALING_INFO_PATH, SCALING_COLS)

    if scaling_ratios is None or df_gt_scaled is None:
        print("Fatal Error: Could not load or calculate scaling info. Aborting.")
        return

    # 2. Load stem data, apply scaling, and merge with GT_scaled
    df_final_output = load_and_scale_stem_data(STEM_LENGTH_PATH, scaling_ratios, df_gt_scaled)

    if df_final_output is None or df_final_output.empty:
        print("Fatal Error: Could not process or merge stem data. Aborting.")
        return

    # 3. Write results to output file
    print(f"3. Writing results to: {OUTPUT_SCALED_PATH}")

    # Output header definition in English
    header = ['File_Name', 'Label_ID', 'GT_Scaled'] + [f'Method_{i + 1}_Scaled' for i in range(5)]

    df_final_output.to_csv(
        OUTPUT_SCALED_PATH,
        sep=' ',
        index=False,
        header=header,
        float_format="%.4f"
    )

    print("\n Scaling and writing completed!")
    print(f"   -> Final output contains {len(df_final_output)} rows.")
    print(f"   -> Results saved to: {OUTPUT_SCALED_PATH}")
    print(f"Total time elapsed: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    main()