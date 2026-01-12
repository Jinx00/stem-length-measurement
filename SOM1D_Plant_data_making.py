import os
import numpy as np
import warnings
import glob
from tqdm import tqdm
import som1D as som
import argparse
from numpy.linalg import linalg

warnings.filterwarnings('ignore')


def get_file_names(folder):
    file_names = glob.glob(folder + '/*')
    return file_names


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def read_pcd(pcd_path):
    try:
        data = np.genfromtxt(pcd_path, dtype=np.float32)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0 or data.shape[1] < 4:
            print(f"Warning: File {pcd_path} has less than 4 columns or is empty.")
            return np.array([])

        return data[:, :4]

    except Exception as e:
        print(f"Error reading {pcd_path} with numpy: {e}")
        return np.array([])


def farthest_point_sample(point, npoint):
    N, D = point.shape
    if N < npoint:
        return point

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)]


def SOM_point_sample(data, resampleNum, sigma):
    organs = []
    organs.append(data)
    assignNum = [resampleNum]
    accept_dict = som.getSkeleton(organs, assignNum, sigma)
    weight_centroid = accept_dict["weights"]

    weights = []
    for o in weight_centroid:
        for s in o:
            weights.append(s[0])
    weights = np.array(weights)  # 形状为 [M, 3]
    return weights


def get_pca_ratio(xyz):
    """
    Calculate the proportion of PCA eigenvalues of the point cloud to measure the structural straightness.
    Returns:: lambda1 / (lambda2 + lambda3)
    """
    N = len(xyz)
    if N < 3:
        return 0.0

    centered_xyz = xyz - np.mean(xyz, axis=0)
    cov_matrix = np.cov(centered_xyz, rowvar=False)
    eigenvalues = linalg.eigh(cov_matrix)[0]
    eigenvalues = np.sort(eigenvalues)[::-1]
    L1, L2, L3 = eigenvalues

    denominator = L2 + L3
    if denominator < 1e-9:
        return 1e9

    return L1 / denominator


def calculate_stem_length(ordered_path_xyz):
    """
    The stem length is estimated by accumulating the Euclidean distance of the ordered skeleton zigzagines.
    """
    if ordered_path_xyz.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(ordered_path_xyz[1:] - ordered_path_xyz[:-1], axis=1)

    total_length = np.sum(distances)
    return total_length


def get_ordered_full_path(skeleton_xyz, endpoint_xyz):
    """
    Topologically connect the PCA endpoints to both ends of the ordered SOM skeleton to construct a complete path.

    Args:
        skeleton_xyz (np.ndarray): [M, 3]。
        endpoint_xyz (np.ndarray):  [2, 3]。

    Returns:
        np.ndarray: [M+2, 3]。
    """
    if skeleton_xyz.shape[0] < 1 or endpoint_xyz.shape[0] < 2:
        return skeleton_xyz

    S_start = skeleton_xyz[0]
    S_end = skeleton_xyz[-1]
    E_a = endpoint_xyz[0]
    E_b = endpoint_xyz[1]

    dist_a_start = np.linalg.norm(E_a - S_start)
    dist_a_end = np.linalg.norm(E_a - S_end)
    dist_b_start = np.linalg.norm(E_b - S_start)
    dist_b_end = np.linalg.norm(E_b - S_end)

    cost1 = dist_a_start + dist_b_end
    cost2 = dist_b_start + dist_a_end

    if cost1 <= cost2:
        ordered_path = np.vstack([E_a, skeleton_xyz, E_b])
    else:
        ordered_path = np.vstack([E_b, skeleton_xyz, E_a])

    return ordered_path

def get_pca_endpoints(xyz):
    """
    Compare the farthest distances projected by the point cloud on the main axes V1 and V2, and select the pair of points with the greatest distance as the endpoints.
    """
    N = len(xyz)
    if N < 2:
        if N == 1:
            return xyz
        return np.array([])

    centroid = np.mean(xyz, axis=0)
    centered_xyz = xyz - centroid

    cov_matrix = np.cov(centered_xyz, rowvar=False)
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]

    dist_V2 = 0
    P_2a, P_2b = None, None

    principal_axis_V1 = eigenvectors[:, sort_indices[0]]
    projections_V1 = np.dot(centered_xyz, principal_axis_V1)

    idx_min_V1 = np.argmin(projections_V1)
    idx_max_V1 = np.argmax(projections_V1)
    P_1a = xyz[idx_min_V1]
    P_1b = xyz[idx_max_V1]
    dist_V1 = np.linalg.norm(P_1a - P_1b)

    if len(sort_indices) > 1:
        principal_axis_V2 = eigenvectors[:, sort_indices[1]]
        projections_V2 = np.dot(centered_xyz, principal_axis_V2)

        idx_min_V2 = np.argmin(projections_V2)
        idx_max_V2 = np.argmax(projections_V2)
        P_2a = xyz[idx_min_V2]
        P_2b = xyz[idx_max_V2]
        dist_V2 = np.linalg.norm(P_2a - P_2b)

    if dist_V1 >= dist_V2:
        return np.vstack([P_1a, P_1b])
    else:
        return np.vstack([P_2a, P_2b])


class InstanceDownSampler():
    def __init__(self, args):
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.ratio = args.sample_ratio
        self.sigma = args.sigma
        self.SOMform = args.use_som_sample
        self.normalize = args.normalize
        self.pca_ratio_threshold = args.pca_ratio_threshold
        self.low_ratio_fixed_nodes = args.low_ratio_fixed_nodes

        warnings.filterwarnings('ignore')
        os.makedirs(self.output_path, exist_ok=True)
        self.process_files()

    def process_files(self):
        datapath_list = get_file_names(self.data_path)
        print('The size of dataset is %d' % len(datapath_list))

        for fn in tqdm(datapath_list, desc="Processing Files"):
            dataname = os.path.basename(fn)
            savename = dataname.split('.')[0] + '_instance_sampled.txt'
            sn = os.path.join(self.output_path, savename)

            P = read_pcd(fn)

            if P.size == 0 or P.shape[1] < 4:
                print(f"Skipping file {dataname} due to invalid or insufficient data.")
                continue

            if self.normalize:
                Pnorm = pc_normalize(P[:, :3])
                point_set = np.hstack((Pnorm, P[:, 3].reshape(-1, 1)))
            else:
                point_set = P[:, :4]

            labels = point_set[:, 3]
            unique_labels = np.unique(labels.astype(int))

            sampled_points_list = []
            instance_lengths = []

            print(f"\nFile: {dataname}, Total Points: {len(point_set)}, Instances: {len(unique_labels)}")

            for label in unique_labels:
                instance_mask = (labels.astype(int) == label)
                instance_pc = point_set[instance_mask, :]
                instance_xyz = instance_pc[:, :3]  

                N_instance = len(instance_pc)
                if N_instance == 0: continue

                # --- 1. Calculate the PCA ratio and determine the number of sparse sampling points ---
                pca_ratio = get_pca_ratio(instance_xyz)

                if pca_ratio > self.pca_ratio_threshold:
                    sparse_target_num = max(1, int(N_instance * self.ratio))
                    structure_type = "Straight"

                else:
                    sparse_target_num = min(N_instance, self.low_ratio_fixed_nodes)
                    sparse_target_num = max(1, sparse_target_num)
                    structure_type = "Curved"

                print(
                    f"  -> Label {int(label)}: PCA Ratio {pca_ratio:.2f} ({structure_type}). Sparse Target: {sparse_target_num} pts.")

                # --- 2. SOM or FPS ---
                if self.SOMform:
                    sampled_xyz = SOM_point_sample(instance_xyz, sparse_target_num, self.sigma)
                    endpoint_xyz = get_pca_endpoints(instance_xyz)

                    ordered_path_xyz = get_ordered_full_path(sampled_xyz, endpoint_xyz)
                    stem_length = calculate_stem_length(ordered_path_xyz)
                    print(f"  -> Label {int(label)}: Estimated Stem Length: {stem_length:.6f}")
                    instance_lengths.append((int(label), stem_length))

                else:  # FPS mode
                    sparse_target_num = max(1, int(N_instance * self.ratio))
                    sampled_pc_with_label = farthest_point_sample(instance_pc, sparse_target_num)
                    sampled_xyz = sampled_pc_with_label[:, :3]
                    print(f"  -> Label {int(label)}: Stem Length calculation skipped (FPS mode, unordered points).")

                # --- 3. Extract endpoints (uniformly use the hybrid PCA strategy) ---
                endpoint_xyz = get_pca_endpoints(instance_xyz)

                if sampled_xyz.size == 0: continue

                # --- 4. Merge and output (no deduplication)---
                combined_xyz = np.vstack([sampled_xyz, endpoint_xyz])

                # 5. Assign the original instance labels to all points
                final_sampled_points = np.zeros((len(combined_xyz), 4))
                final_sampled_points[:, :3] = combined_xyz
                final_sampled_points[:, 3] = label

                sampled_points_list.append(final_sampled_points)

            # 6. Merge and save the results
            if sampled_points_list:
                final_sampled_data = np.vstack(sampled_points_list)

                np.savetxt(sn, final_sampled_data, fmt='%.8f %.8f %.8f %.0f')
                print(f"Saved {len(final_sampled_data)} sampled points to {sn}")
                if self.SOMform and instance_lengths:
                    log_savename = dataname.split('.')[0] + '_stem_lengths.csv'
                    log_sn = os.path.join(self.output_path, log_savename)

                    log_data = np.array(instance_lengths)
                    header = "Instance_Label,Estimated_Stem_Length"
                    np.savetxt(log_sn, log_data, fmt='%d,%.6f', header=header, delimiter=',')
                    print(f"Saved stem lengths for {len(instance_lengths)} instances to {log_sn}")
            else:
                print(f"No valid points sampled for file {dataname}.")


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('instance_downsampling')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    parser.add_argument('--sample_ratio', type=float, default=0.01,
                        help='Sampling ratio for final instance skeleton (e.g., 0.05) when PCA ratio is high.')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Control parameters for activating domain radius, recommended range[1.2, 1.5]')
    parser.add_argument('--use_som_sample', type=bool, default=True, help='use som sampling instead of FPS sampling')

    parser.add_argument('--pca_ratio_threshold', type=float, default=50.0,
                        help='Threshold for PCA ratio (L1 / (L2+L3)). If ratio is >= this value, use sample_ratio. Otherwise, use fixed nodes.')
    parser.add_argument('--low_ratio_fixed_nodes', type=int, default=10,
                        help='Fixed number of nodes to sample when PCA ratio is low (curved).')

    parser.add_argument('--data_path', type=str, default='./data', help='specify your point cloud path')
    parser.add_argument('--output_path', type=str, default='./output_instance_sampled', help='specify your output path')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='Normalize point cloud with coordinate origin (0, 0, 0)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    InstanceDownSampler(args=args)