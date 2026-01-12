import os
import numpy as np

def filter_and_relabel_stem_points(source_root, target_root, ins_subfolder, sem_subfolder):
    """
    source_root: 'GT' or 'Predict'
    target_root: New root directory for saving results
    ins_subfolder: Name of the instance-related subfolder
    sem_subfolder: Name of the semantic-related subfolder
    """
    ins_folder = os.path.join(source_root, ins_subfolder)
    sem_folder = os.path.join(source_root, sem_subfolder)

    target_ins_folder = os.path.join(target_root, os.path.basename(source_root), 'ins')
    target_sem_folder = os.path.join(target_root, os.path.basename(source_root), 'sem')

    os.makedirs(target_ins_folder, exist_ok=True)
    os.makedirs(target_sem_folder, exist_ok=True)

    files = os.listdir(sem_folder)
    files = [f for f in files if f.endswith('.txt')]

    for file in files:
        sem_path = os.path.join(sem_folder, file)
        ins_path = os.path.join(ins_folder, file)

        # Load data
        sem_data = np.loadtxt(sem_path)
        ins_data = np.loadtxt(ins_path)

        # Check if point counts are consistent
        assert sem_data.shape[0] == ins_data.shape[0], f"Point count mismatch in file {file}!"

        # Keep points with semantic labels 0, 2, 4, or 6
        mask = np.isin(sem_data[:, 3], [0, 2, 4, 6])

        sem_filtered = sem_data[mask]
        ins_filtered = ins_data[mask]

        # ------- Relabeling ---------

        # (2) Re-encode instance labels (continuous numbering starting from 0)
        original_ins_labels = ins_filtered[:, 3]
        new_ins_labels = np.zeros_like(original_ins_labels, dtype=int)
        unique_ins_values = np.unique(original_ins_labels)
        ins_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_ins_values))}
        
        for old_label, new_label in ins_mapping.items():
            new_ins_labels[original_ins_labels == old_label] = new_label
        ins_filtered[:, 3] = new_ins_labels

        # ------- Save ---------
        np.savetxt(os.path.join(target_sem_folder, file), sem_filtered, fmt='%.6f %.6f %.6f %d')
        np.savetxt(os.path.join(target_ins_folder, file), ins_filtered, fmt='%.6f %.6f %.6f %d')

        print(f"{file}: Original instance labels {unique_ins_values.tolist()} -> New labels {list(ins_mapping.values())}")
        print(f"    Points retained: {sem_filtered.shape[0]}")

if __name__ == "__main__":
    # Specify subfolder names for GT and Predict using relative paths
    # Assuming the script is run from the 'postprocessed_v3' directory
    tasks = [
        {
            'source_root': 'gt',
            'ins_subfolder': 'ins_gt',
            'sem_subfolder': 'sem_gt'
        },
        {
            'source_root': 'predict',
            'ins_subfolder': 'ins',
            'sem_subfolder': 'sem'
        }
    ]

    target_root = 'stem_only_relabel_ins'

    for task in tasks:
        filter_and_relabel_stem_points(
            source_root=task['source_root'],
            target_root=target_root,
            ins_subfolder=task['ins_subfolder'],
            sem_subfolder=task['sem_subfolder']
        )

    print("All processing completed!")