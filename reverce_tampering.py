import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

HEART_SEG_VALUE = 2

T4_inv = np.array(
    [
        [1, 0, 0, -50],
        [0, 1, 0, -40],
        [0, 0, 1, -15],
        [0, 0, 0, 1],
    ]
)

T3_inv = np.array(
    [
        [1, 0, 0, 275], 
        [0, 1, 0, 200], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]
    ]
)

phi = -(27 / 180) * np.pi
R2_inv = np.array(
    [
        [np.cos(phi), np.sin(phi), 0, 0],
        [-np.sin(phi), np.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

T1_inv = np.array(
    [
        [1, 0, 0, -275],
        [0, 1, 0, -200],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
)

REVERSE_TRANSFORM = T4_inv @ T3_inv @ R2_inv @ T1_inv


def fix_patient_heart_segmentation(patient_path: Path):
    # Load GT
    tampered_gt = nib.nifti1.load(str(patient_path / "GT.nii.gz"))
    gt_arr = np.array(tampered_gt.dataobj, dtype=np.uint8)

    # Extract heart segmentation
    tampered_heart_gt = (gt_arr == HEART_SEG_VALUE).astype(np.uint8)

    # Apply reverse affine transform
    heart_gt = affine_transform(tampered_heart_gt, REVERSE_TRANSFORM, order=0)

    # Remove tampered-with heart segmentation from GT
    gt_arr[tampered_heart_gt.astype(bool)] = 0

    # Add back fixed heart segmentation to GT
    gt_arr[heart_gt.astype(bool)] = 2

    # Save untampered GT
    fixed_gt = nib.nifti1.Nifti1Image(gt_arr, tampered_gt.affine, tampered_gt.header)
    nib.nifti1.save(fixed_gt, str(patient_path / "GT_fixed.nii.gz"))


def main(args: argparse.Namespace):
    start = datetime.now()
    patient_dirs = list(Path(args.source_dir).glob("*"))

    # Use all CPU cores
    with ProcessPoolExecutor() as executor:
        # Process all patients in parallel
        futures = [
            executor.submit(fix_patient_heart_segmentation, patient_path)
            for patient_path in patient_dirs
        ]

        # Track progress
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Reversing heart GT segmentation mask tampering"
            f" for {len(patient_dirs)} patients in '{args.source_dir}'...",
        ):
            pass

    end = datetime.now()
    print(f"Process took {(end - start).seconds} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse Heart GT Tampering")
    parser.add_argument("--source_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
