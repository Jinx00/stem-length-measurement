import os
import numpy as np
from collections import defaultdict

def find_common_points(gt_data, pred_data, tol=1e-6):
    """
    Find points with identical coordinates in gt and pred point clouds (within tolerance).

    gt_data: GT point cloud data (N, 4), containing (x, y, z, ins_label)
    pred_data: Pred point cloud data (M, 4), containing (x, y, z, ins_label)
    tol: Tolerance for coordinate matching
    Returns: gt_indices, pred_indices (indices of common points)
    """
    gt_coords = gt_data[:, :3]  # (N, 3)
    pred_coords = pred_data[:, :3]  # (M, 3)

    # Use rounding and hashing to find common points quickly
    gt_coords_rounded = np.round(gt_coords / tol).astype(int)
    pred_coords_rounded = np.round(pred_coords / tol).astype(int)

    # Convert coordinates to tuples for comparison
    gt_coords_str = [tuple(coord) for coord in gt_coords_rounded]
    pred_coords_str = [tuple(coord) for coord in pred_coords_rounded]

    # Find intersection of coordinates
    common_coords = set(gt_coords_str) & set(pred_coords_str)

    # Get indices of common points
    gt_indices = [i for i, coord in enumerate(gt_coords_str) if coord in common_coords]
    pred_indices = [i for i, coord in enumerate(pred_coords_str) if coord in common_coords]

    # Ensure one-to-one mapping based on sorted coordinates
    common_coords_list = sorted(common_coords)
    gt_indices_sorted = []
    pred_indices_sorted = []
    
    # Create lookup dictionaries for speed
    gt_lookup = {coord: i for i, coord in enumerate(gt_coords_str)}
    pred_lookup = {coord: i for i, coord in enumerate(pred_coords_str)}

    for coord in common_coords_list:
        gt_indices_sorted.append(gt_lookup[coord])
        pred_indices_sorted.append(pred_lookup[coord])

    return np.array(gt_indices_sorted), np.array(pred_indices_sorted)


def compute_iou(gt_labels, pred_labels, gt_id, pred_id):
    """
    Compute IoU between gt and pred instances based on common points.

    gt_labels: Instance labels of common points in GT
    pred_labels: Instance labels of common points in Pred
    gt_id: Specific GT instance ID
    pred_id: Specific Pred instance ID
    """
    gt_mask = (gt_labels == gt_id)
    pred_mask = (pred_labels == pred_id)

    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)

    if union == 0:
        return 0.0
    return intersection / union


def relabel_predict_instances(gt_ins_folder, pred_ins_folder, target_ins_folder, tol=1e-6):
    """
    Relabel prediction instances based on GT labels using IoU of common points.

    gt_ins_folder: Path to filtered GT instance folder
    pred_ins_folder: Path to filtered prediction instance folder
    target_ins_folder: Path to save relabeled prediction instance files
    tol: Coordinate matching tolerance
    """
    os.makedirs(target_ins_folder, exist_ok=True)

    files = [f for f in os.listdir(pred_ins_folder) if f.endswith('.txt')]

    for file in files:
        gt_ins_path = os.path.join(gt_ins_folder, file)
        pred_ins_path = os.path.join(pred_ins_folder, file)

        if not os.path.exists(gt_ins_path):
            print(f"Warning: GT file {file} not found. Skipping...")
            continue

        # Load instance data
        gt_ins_data = np.loadtxt(gt_ins_path)
        pred_ins_data = np.loadtxt(pred_ins_path)

        # Find common points
        gt_indices, pred_indices = find_common_points(gt_ins_data, pred_ins_data, tol)

        if len(gt_indices) == 0:
            print(f"Warning: No common points found in {file}. Skipping...")
            continue

        # Extract instance labels for common points
        gt_common_labels = gt_ins_data[gt_indices, 3].astype(int)
        pred_common_labels = pred_ins_data[pred_indices, 3].astype(int)

        # Get unique instance IDs
        gt_unique_ids = np.unique(gt_common_labels)
        pred_unique_ids = np.unique(pred_ins_data[:, 3].astype(int))

        # Calculate IoU for each (pred_id, gt_id) pair
        iou_matrix = defaultdict(dict)
        for pred_id in pred_unique_ids:
            for gt_id in gt_unique_ids:
                iou_matrix[pred_id][gt_id] = compute_iou(gt_common_labels, pred_common_labels, gt_id, pred_id)

        # Greedy matching: Assign pred_id to the gt_id with the highest IoU
        matches = {}  # pred_id -> gt_id
        used_gt_ids = set()

        # Sort pairs by IoU in descending order
        iou_pairs = [(pred_id, gt_id, iou_matrix[pred_id][gt_id])
                     for pred_id in pred_unique_ids for gt_id in gt_unique_ids]
        iou_pairs.sort(key=lambda x: x[2], reverse=True)

        for pred_id, gt_id, iou_val in iou_pairs:
            if pred_id not in matches and gt_id not in used_gt_ids and iou_val > 0:
                matches[pred_id] = gt_id
                used_gt_ids.add(gt_id)

        # Assign new IDs for unmatched prediction instances
        max_gt_id = max(gt_unique_ids) if len(gt_unique_ids) > 0 else -1
        next_id = max_gt_id + 1
        for pred_id in pred_unique_ids:
            if pred_id not in matches:
                matches[pred_id] = next_id
                next_id += 1

        # Relabel pred_ins_data (all points)
        pred_labels = pred_ins_data[:, 3].astype(int)
        new_pred_labels = np.zeros_like(pred_labels)
        for pred_id, new_id in matches.items():
            new_pred_labels[pred_labels == pred_id] = new_id
        pred_ins_data[:, 3] = new_pred_labels

        # Save relabeled data
        target_path = os.path.join(target_ins_folder, file)
        np.savetxt(target_path, pred_ins_data, fmt='%.6f %.6f %.6f %d')

        print(f"{file}: Relabeling completed. Mapping: {matches}")
        print(f"    Common points: {len(gt_indices)}, Original IDs: {pred_unique_ids.tolist()}, New IDs: {np.unique(new_pred_labels).tolist()}")


if __name__ == "__main__":
    # Define paths relative to the current working directory
    # Assuming the script runs in the same environment as the previous processing step
    target_root = 'stem_only_relabel_ins'
    gt_ins_folder = os.path.join(target_root, 'gt', 'ins')
    pred_ins_folder = os.path.join(target_root, 'predict', 'ins')
    target_ins_folder = os.path.join(target_root, 'predict', 'ins_relabeled')

    # Execute relabeling
    relabel_predict_instances(gt_ins_folder, pred_ins_folder, target_ins_folder, tol=1e-6)

    print("All files have been relabeled successfully!")