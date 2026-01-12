import numpy as np
import os
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
from numpy.linalg import linalg
from warnings import warn
# Ensure the som1D module is available in your environment
import som1D as som

np.random.seed(42)

# --- Global Constants and Configuration ---
PCA_POINT_THRESHOLD = 8        # Point threshold: use PCA if points are fewer than this
RANSAC_MIN_INLIER_RATIO = 0.8  # Minimum inlier ratio for successful RANSAC fitting
PCA_RATIO_THRESHOLD = 50.0     # PCA ratio threshold (L1/(L2+L3)) to judge straightness in SOM mode
SAMPLE_RATIO = 0.02            # Ratio of original points to sample for SOM when PCA ratio is high
LOW_RATIO_FIXED_NODES = PCA_POINT_THRESHOLD  # Fixed node count for SOM when PCA ratio is low
FIXED_NODES = 20
SOM_SIGMA = 1.5                # SOM training parameter (sigma)


def SOM_point_sample(data, resampleNum, sigma):
    """Generates ordered SOM skeleton nodes."""
    organs = []
    organs.append(data)
    assignNum = [resampleNum]
    accept_dict = som.getSkeleton(organs, assignNum, sigma)
    weight_centroid = accept_dict["weights"]
    weights = []
    for o in weight_centroid:
        for s in o:
            weights.append(s[0])
    return np.array(weights)


def get_pca_ratio(xyz):
    """Calculates the PCA eigenvalue ratio: L1 / (L2 + L3)."""
    N = len(xyz)
    if N < 3: return 0.0
    centered_xyz = xyz - np.mean(xyz, axis=0)
    cov_matrix = np.cov(centered_xyz, rowvar=False)
    eigenvalues = linalg.eigh(cov_matrix)[0]
    eigenvalues = np.sort(eigenvalues)[::-1]
    L1, L2, L3 = eigenvalues
    denominator = L2 + L3
    if denominator < 1e-9: return 1e9
    return L1 / denominator


def get_pca_endpoints(xyz):
    """
    Finds the pair of endpoints with the maximum Euclidean distance 
    by projecting points onto V1 and V2 principal axes.
    Returns [2, 3] array of endpoints.
    """
    N = len(xyz)
    if N < 2:
        return xyz if N == 1 else np.array([])

    centroid = np.mean(xyz, axis=0)
    centered_xyz = xyz - centroid
    cov_matrix = np.cov(centered_xyz, rowvar=False)
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
    sort_indices = np.argsort(eigenvalues)[::-1]

    # V1 axis endpoints and distance
    principal_axis_V1 = eigenvectors[:, sort_indices[0]]
    projections_V1 = np.dot(centered_xyz, principal_axis_V1)
    P_1a = xyz[np.argmin(projections_V1)]
    P_1b = xyz[np.argmax(projections_V1)]
    dist_V1 = np.linalg.norm(P_1a - P_1b)
    result_V1 = np.vstack([P_1a, P_1b])

    # V2 axis endpoints and distance
    dist_V2 = 0
    if len(sort_indices) > 1:
        principal_axis_V2 = eigenvectors[:, sort_indices[1]]
        projections_V2 = np.dot(centered_xyz, principal_axis_V2)
        P_2a = xyz[np.argmin(projections_V2)]
        P_2b = xyz[np.argmax(projections_V2)]
        dist_V2 = np.linalg.norm(P_2a - P_2b)
        result_V2 = np.vstack([P_2a, P_2b])
    else:
        result_V2 = np.array([])

    # Return the pair with the maximum Euclidean distance
    return result_V1 if dist_V1 >= dist_V2 else result_V2


def get_ordered_full_path(skeleton_xyz, endpoint_xyz):
    """Topologically connects PCA endpoints to SOM skeleton ends to construct a full path."""
    if skeleton_xyz.shape[0] < 1 or endpoint_xyz.shape[0] < 2:
        return skeleton_xyz

    S_start, S_end = skeleton_xyz[0], skeleton_xyz[-1]
    E_a, E_b = endpoint_xyz[0], endpoint_xyz[1]

    dist_a_start = np.linalg.norm(E_a - S_start)
    dist_b_end = np.linalg.norm(E_b - S_end)
    dist_b_start = np.linalg.norm(E_b - S_start)
    dist_a_end = np.linalg.norm(E_a - S_end)

    cost1 = dist_a_start + dist_b_end  # Path: E_a -> S_start -> ... -> S_end -> E_b
    cost2 = dist_b_start + dist_a_end  # Path: E_b -> S_start -> ... -> S_end -> E_a

    if cost1 <= cost2:
        ordered_path = np.vstack([E_a, skeleton_xyz, E_b])
    else:
        ordered_path = np.vstack([E_b, skeleton_xyz, E_a])

    return ordered_path


def calculate_stem_length(ordered_path_xyz):
    """Estimates stem length by summing Euclidean distances of the ordered path."""
    if ordered_path_xyz.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(ordered_path_xyz[1:] - ordered_path_xyz[:-1], axis=1)
    return np.sum(distances)


# --- RANSAC/PCA Fitting Tools ---

def compute_average_distance(points, k=2):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    return np.mean(distances[:, 1])


def ransac_line_fit(points, threshold, min_inlier_ratio=RANSAC_MIN_INLIER_RATIO, max_trials=5000):
    n_points = points.shape[0]
    best_inlier_mask = np.zeros(n_points, dtype=bool)
    best_line_params = None
    max_inliers = 0

    for _ in range(max_trials):
        sample_idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[sample_idx]
        line_dir = p2 - p1
        if np.linalg.norm(line_dir) == 0: continue
        line_dir /= np.linalg.norm(line_dir)

        diffs = points - p1
        dists = np.linalg.norm(diffs - np.outer(np.dot(diffs, line_dir), line_dir), axis=1)
        inlier_mask = dists < threshold
        num_inliers = np.sum(inlier_mask)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_line_params = (p1, line_dir)

    inlier_ratio = max_inliers / n_points
    success = inlier_ratio > min_inlier_ratio
    return success, best_inlier_mask, inlier_ratio, best_line_params


def extend_line_to_cover_points(points, line_start, line_dir):
    """Extends the fitted line to cover all points, returning two boundary endpoints."""
    projections = (points - line_start) @ line_dir
    min_proj = projections.min()
    max_proj = projections.max()
    extended_start = line_start + min_proj * line_dir
    extended_end = line_start + max_proj * line_dir
    return np.stack([extended_start, extended_end], axis=0)


def fit_line_with_pca(points):
    """Fits a line using PCA and returns two endpoints covering all points."""
    pca = PCA(n_components=1)
    pca.fit(points)
    line_dir = pca.components_[0]
    line_dir /= np.linalg.norm(line_dir)
    p1 = np.mean(points, axis=0)
    return extend_line_to_cover_points(points, p1, line_dir)


def get_ransac_endpoints_force(xyz, threshold, max_trials=5000):
    """Forces the use of the best RANSAC line found to determine endpoints."""
    N = xyz.shape[0]
    if N < 2:
        return xyz if N == 1 else np.array([])

    _, _, _, best_line_params = ransac_line_fit(xyz, threshold, min_inlier_ratio=0.0, max_trials=max_trials)

    if best_line_params is not None:
        p1, line_dir = best_line_params
        return extend_line_to_cover_points(xyz, p1, line_dir)
    else:
        print("Warning: RANSAC failed, falling back to PCA line fitting.")
        return fit_line_with_pca(xyz)


def process_one_label(points, label, avg_distance):
    """Core logic: determines fit type and calculates stem length for a single instance."""
    N_instance = len(points)
    if N_instance < 2:
        return np.array([]), 0.0, "Too Few Points"

    if N_instance < PCA_POINT_THRESHOLD:
        # Case 1: Few points, use PCA line fitting
        curve_points = fit_line_with_pca(points)
        length = np.linalg.norm(curve_points[1] - curve_points[0])
        fit_type = "Line (PCA)"
    else:
        # Case 2: Enough points, use RANSAC/SOM hybrid logic
        ransac_threshold = avg_distance * 2
        success, _, inlier_ratio, line_params = ransac_line_fit(
            points, threshold=ransac_threshold, min_inlier_ratio=RANSAC_MIN_INLIER_RATIO
        )

        if success:
            # Case 2A: RANSAC Success (Straight line)
            fit_type = "Line (RANSAC)"
            p1, line_dir = line_params
            curve_points = extend_line_to_cover_points(points, p1, line_dir)
            length = np.linalg.norm(curve_points[1] - curve_points[0])
        else:
            # Case 2B: RANSAC Failure (Curve), use SOM
            fit_type = "SOM (Curve Fit)"
            pca_ratio = get_pca_ratio(points)
            if pca_ratio > PCA_RATIO_THRESHOLD:
                sparse_target_num = min(FIXED_NODES, max(1, int(N_instance * SAMPLE_RATIO)))
            else:
                sparse_target_num = max(1, min(N_instance, LOW_RATIO_FIXED_NODES))

            sampled_xyz = SOM_point_sample(points, sparse_target_num, SOM_SIGMA)
            endpoint_xyz = get_ransac_endpoints_force(points, ransac_threshold)

            if sampled_xyz.shape[0] < 2:
                warn(f"SOM failed for label {label}, N={N_instance}. Falling back to PCA.")
                curve_points = fit_line_with_pca(points)
                length = np.linalg.norm(curve_points[1] - curve_points[0])
                fit_type = "Line (PCA Fallback)"
            else:
                ordered_path_xyz = get_ordered_full_path(sampled_xyz, endpoint_xyz)
                length = calculate_stem_length(ordered_path_xyz)
                curve_points = np.vstack([sampled_xyz, endpoint_xyz])

    return curve_points, length, fit_type


def process_file(file_path, output_folder):
    try:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        start_time = time.time()

        data = np.loadtxt(file_path)
        xyz = data[:, :3]
        labels = data[:, 3].astype(int)

        avg_distance = compute_average_distance(xyz)
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        save_curve_dir = os.path.join(output_folder, f"{file_basename}_curves")
        os.makedirs(save_curve_dir, exist_ok=True)

        lengths_info = []
        total_stems = 0
        total_length = 0.0

        for label in np.unique(labels):
            if label < 0: continue
            label_points = xyz[labels == label]
            if len(label_points) < 2: continue

            curve_points, length, fit_type = process_one_label(label_points, label, avg_distance)
            save_path = os.path.join(save_curve_dir, f"label_{label}.txt")
            np.savetxt(save_path, curve_points, fmt='%.6f')

            total_stems += 1
            total_length += length
            lengths_info.append((label, length, fit_type))

        # Save individual file results
        with open(os.path.join(save_curve_dir, "lengths.txt"), "w") as f:
            f.write(f"Total stems: {total_stems}\n")
            f.write(f"Average length: {total_length / total_stems if total_stems > 0 else 0:.2f} mm\n")
            for label, length, fit_type in lengths_info:
                f.write(f"Label {label}: {length:.3f} mm, Fit Type: {fit_type}\n")

        print(f"Process completed in {time.time() - start_time:.2f}s")
        return file_basename, total_stems, total_length, lengths_info

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def batch_process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_list = glob.glob(os.path.join(input_folder, "*.txt"))
    total_files = len(file_list)

    log_file = os.path.join(output_folder, "processing.log")
    with open(log_file, "w") as f:
        f.write(f"Process started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary_data = []
    all_lengths = []

    for idx, file_path in enumerate(file_list):
        print(f"\nProcessing file {idx + 1}/{total_files} ({((idx + 1) / total_files) * 100:.1f}%)")
        result = process_file(file_path, output_folder)
        if result:
            file_basename, total_stems, total_length, lengths_info = result
            summary_data.append((file_basename, total_stems, total_length))
            all_lengths.append((file_basename, lengths_info))

    # Generate detailed report
    with open(os.path.join(output_folder, "all_lengths.txt"), "w") as f:
        f.write("=== All Stems Length Details ===\n")
        for file_basename, lengths in all_lengths:
            f.write(f"\nFile: {file_basename}\n")
            for label, length, fit_type in lengths:
                f.write(f"  Label {label}: {length:.3f} mm ({fit_type})\n")

    # Generate summary report
    with open(os.path.join(output_folder, "summary.txt"), "w") as f:
        f.write("=== Summary Report ===\n")
        total_stems_sum = sum(x[1] for x in summary_data)
        total_length_sum = sum(x[2] for x in summary_data)
        f.write(f"Total files processed: {len(summary_data)}\n")
        f.write(f"Total stems counted: {total_stems_sum}\n")
        f.write(f"Average length per stem: {total_length_sum / total_stems_sum if total_stems_sum > 0 else 0:.2f} mm\n")
        f.write("\nFile Details:\n")
        for file_info in summary_data:
            f.write(f"{file_info[0]}: {file_info[1]} stems, total {file_info[2]:.2f} mm\n")

    with open(log_file, "a") as f:
        f.write(f"\nProcess completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    # Using relative paths based on the requested structure
    input_folder = 'predict_ins_relabeled'
    output_folder = 'stem_length_results'
    batch_process_folder(input_folder, output_folder)