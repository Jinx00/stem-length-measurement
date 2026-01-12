import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
from matplotlib.lines import Line2D

# --- File Path Configuration (Using Relative Paths) ---
BASE_DIR = "stem_experiment_data"
INPUT_FILE = os.path.join(BASE_DIR, "stem_summary_original_scale.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "stem_length_analysis_results.txt")

def load_and_clean_data(input_file):
    """
    Loads and cleans the dataset.
    Returns:
        data: Cleaned DataFrame
        methods: List of method names
        species: Species name extracted from filename
    """
    print(f"Loading data from: {input_file}")
    try:
        # Extract species name from filename
        species = os.path.basename(input_file).replace('.txt', '')
        print(f"Analyzing species: {species}")

        # Define column names (Translated to English)
        col_names = ['Filename', 'Label', 'GT_Truth', 'Our_Method', 'Geodesic', 'PCA', 'Curvature_PCA', 'RANSAC']
        numeric_cols = ['GT_Truth', 'Our_Method', 'Geodesic', 'PCA', 'Curvature_PCA', 'RANSAC']
        methods = ['Our_Method', 'Geodesic', 'PCA', 'Curvature_PCA', 'RANSAC']

        # Load data (Space separated)
        data = pd.read_csv(input_file, sep=' ', header=None, names=col_names)
        initial_rows = len(data)
        print(f"Initial rows: {initial_rows}")

        # Filter out invalid rows (e.g., placeholder '\')
        data = data[data['GT_Truth'] != '\\']
        filtered_rows = len(data)
        print(f"Filtered rows: {filtered_rows} ({initial_rows - filtered_rows} invalid rows removed)")

        # Convert to numeric
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Check and handle missing values
        if data.isna().any().any():
            na_sum = data.isna().sum()
            print("Warning: Missing values detected:")
            print(na_sum[na_sum > 0])
            data = data.dropna(subset=numeric_cols)
            print(f"Rows after dropping NaNs: {len(data)}")

        print("Data loading and cleaning completed.")
        return data, methods, species

    except Exception as e:
        print(f"Data processing error: {e}")
        exit(1)

def calculate_metrics(data, methods, target_col='GT_Truth'):
    """
    Calculates various evaluation metrics.
    Returns:
        metrics_df: DataFrame containing metrics
        fit_results: Dictionary with regression info (Equation, R2)
    """
    metrics = []
    fit_results = {}

    print("Calculating evaluation metrics...")
    for method in methods:
        # Basic metrics
        errors = data[method] - data[target_col]
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        relative_error = np.mean(np.abs(errors) / data[target_col])
        corr, _ = pearsonr(data[method], data[target_col])

        # Linear fitting (y = kx + b)
        coeffs = np.polyfit(data[target_col], data[method], 1)
        k, b = coeffs

        # R² Calculation
        pred = k * data[target_col] + b
        ss_tot = np.sum((data[method] - np.mean(data[method])) ** 2)
        ss_res = np.sum((data[method] - pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Store results
        metrics.append({
            'Method': method,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Relative_Error': relative_error,
            'Correlation': corr,
            'R2': r2
        })

        fit_results[method] = {
            'Slope': k,
            'Intercept': b,
            'R2': r2,
            'Equation': f"y = {k:.4f}x + {b:.4f}"
        }

    metrics_df = pd.DataFrame(metrics)
    print("Metric calculation completed.")
    return metrics_df, fit_results

def write_species_results(output_file, species, metrics_df, fit_results):
    """
    Writes analysis results to the output file (Append mode).
    """
    print(f"Writing analysis results for {species}...")
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Species: {species} Analysis Results\n")
            f.write(f"{'=' * 60}\n\n")

            # Quantitative metrics
            f.write("=== Quantitative Metrics Comparison ===\n")
            f.write("Method\tMSE\tRMSE\tMAE\tRel_Error\tCorr\tR2\n")
            for _, row in metrics_df.iterrows():
                f.write(f"{row['Method']}\t{row['MSE']:.6f}\t{row['RMSE']:.6f}\t"
                        f"{row['MAE']:.6f}\t{row['Relative_Error']:.6f}\t"
                        f"{row['Correlation']:.6f}\t{row['R2']:.6f}\n")

            # Regression equations
            f.write("\n=== Linear Regression Equations ===\n")
            for method, res in fit_results.items():
                f.write(f"{method}: {res['Equation']}\n")

            f.write("\n=== Goodness of Fit (R2) ===\n")
            for method, res in fit_results.items():
                f.write(f"{method}: {res['R2']:.4f}\n")

        print(f"Results for {species} saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    # Setup relative base directory
    base_dir = "stem_experiment_data"
    output_file = os.path.join(base_dir, "stem_length_analysis_results.txt")

    # List of species files (Mapped to relative paths)
    species_files = [
        ("All", os.path.join(base_dir, "stem_summary_original_scale.txt")),
        ("Soybean", os.path.join(base_dir, "soybean.txt")),
        ("Pepper", os.path.join(base_dir, "pepper.txt")),
        ("Tomato", os.path.join(base_dir, "tomato.txt")),
        ("Tobacco", os.path.join(base_dir, "tobacco.txt"))
    ]

    # Initialize output file (Clear old content)
    os.makedirs(base_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Comparative Analysis Results for Stem Length Calculation\n")
        f.write("=" * 60 + "\n")
        f.write("This file contains multi-species analysis results\n")

    for species_name, input_file in species_files:
        if not os.path.exists(input_file):
            print(f"Warning: File not found: {input_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Starting Analysis: {species_name}")
        print(f"{'=' * 60}")

        # Process
        data, methods, _ = load_and_clean_data(input_file)
        metrics_df, fit_results = calculate_metrics(data, methods)
        write_species_results(output_file, species_name, metrics_df, fit_results)

        print(f"{species_name} Analysis Finished.")

if __name__ == "__main__":
    main()