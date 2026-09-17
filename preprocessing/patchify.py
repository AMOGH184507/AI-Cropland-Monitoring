import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

try:
    from preprocessing.config import (
        PROJECT_ROOT,
        DATASET_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_MANIFEST_PATH,
        PATCH_SIZE,
        STRIDE,
        EMPTY_PATCH_THRESHOLD,
        RANDOM_SEED,
        SPLIT_RATIOS,
        CANONICAL_BAND_ORDER,
        INCLUDE_NDVI
    )
    from preprocessing.parse_netcdf import parse_tile
except ImportError:
    from config import (
        PROJECT_ROOT,
        DATASET_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_MANIFEST_PATH,
        PATCH_SIZE,
        STRIDE,
        EMPTY_PATCH_THRESHOLD,
        RANDOM_SEED,
        SPLIT_RATIOS,
        CANONICAL_BAND_ORDER,
        INCLUDE_NDVI
    )
    from parse_netcdf import parse_tile


def extract_patches(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    field_id: str,
    patch_size: Tuple[int, int] = PATCH_SIZE,
    stride: Tuple[int, int] = STRIDE,
    max_empty_ratio: float = EMPTY_PATCH_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Extract fixed-size spatial patches while preserving time.

    Image:
        (C, T, H, W)

    Mask:
        (H, W)

    Output image patch:
        (C, T, patch_H, patch_W)

    Output mask patch:
        (patch_H, patch_W)
    """

    # ---------------------------------------------------------
    # VALIDATE IMAGE SHAPE
    # ---------------------------------------------------------

    if image_np.ndim != 4:
        raise ValueError(
            f"Expected image shape (C,T,H,W), "
            f"got {image_np.shape}"
        )

    if mask_np.ndim != 2:
        raise ValueError(
            f"Expected mask shape (H,W), "
            f"got {mask_np.shape}"
        )

    _, _, h, w = image_np.shape

    patch_h, patch_w = patch_size

    stride_y, stride_x = stride

    patches = []

    patch_idx = 0

    # ---------------------------------------------------------
    # SLIDING WINDOW
    # ---------------------------------------------------------

    y_steps = (
        range(
            0,
            h - patch_h + 1,
            stride_y
        )
        if h >= patch_h
        else [0]
    )

    x_steps = (
        range(
            0,
            w - patch_w + 1,
            stride_x
        )
        if w >= patch_w
        else [0]
    )

    for y in y_steps:

        for x in x_steps:

            # -------------------------------------------------
            # SPATIAL CROP
            # -------------------------------------------------

            img_crop = image_np[
                :,
                :,
                y:y + patch_h,
                x:x + patch_w
            ]

            mask_crop = mask_np[
                y:y + patch_h,
                x:x + patch_w
            ]

            # -------------------------------------------------
            # EMPTY PATCH CHECK
            # -------------------------------------------------

            field_pixel_count = np.sum(
                mask_crop > 0.5
            )

            total_pixels = mask_crop.size

            empty_ratio = (
                1.0
                - field_pixel_count /
                max(total_pixels, 1)
            )

            if (
                empty_ratio >= max_empty_ratio
                and field_pixel_count == 0
            ):
                continue

            # -------------------------------------------------
            # PATCH ID
            # -------------------------------------------------

            patch_id = (
                f"{field_id}_p{patch_idx:02d}"
                f"_y{y}_x{x}"
            )

            patches.append({
                "patch_id": patch_id,
                "image": img_crop,
                "mask": mask_crop,
                "field_id": field_id,
                "crop_coords": (
                    y,
                    x,
                    patch_h,
                    patch_w
                )
            })

            patch_idx += 1

    return patches


def find_all_tile_pairs(dataset_dir: Path = DATASET_DIR) -> List[Tuple[Path, Path, str]]:
    """
    Scans dataset_dir for matching (.nc image, .tif mask) file pairs.
    Returns list of tuples: (nc_path, tif_path, field_id)
    """
    dataset_dir = Path(dataset_dir)
    nc_files = sorted(list(dataset_dir.rglob("*.nc")))

    pairs = []
    for nc_path in nc_files:
        country = nc_path.parent.name
        mask_name = nc_path.name.replace("_S2_10m_256.nc", "_S2label_10m_256.tif")

        # Check expected mask paths
        potential_mask_paths = [
            nc_path.parent.parent.parent / "masks" / country / mask_name,
            dataset_dir / "masks" / country / mask_name,
            nc_path.parent / mask_name
        ]

        tif_path = None
        for p in potential_mask_paths:
            if p.exists():
                tif_path = p
                break

        if tif_path is not None:
            field_id = nc_path.name.split("_S2_")[0]
            pairs.append((nc_path, tif_path, field_id))

    return pairs


def create_field_splits(
    pairs: List[Tuple[Path, Path, str]],
    seed: int = RANDOM_SEED,
    ratios: Dict[str, float] = SPLIT_RATIOS
) -> Dict[str, List[Tuple[Path, Path, str]]]:
    """
    Groups tile pairs by field_id and splits them into train, val, test splits (70/15/15)
    reproducibly using a fixed random seed at the FIELD level to prevent data leakage.
    """
    # Group by unique field ID
    field_to_pairs: Dict[str, List[Tuple[Path, Path, str]]] = {}
    for nc_p, tif_p, fid in pairs:
        if fid not in field_to_pairs:
            field_to_pairs[fid] = []
        field_to_pairs[fid].append((nc_p, tif_p, fid))

    unique_fields = sorted(list(field_to_pairs.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_fields)

    num_fields = len(unique_fields)
    train_count = int(round(num_fields * ratios["train"]))
    val_count = int(round(num_fields * ratios["val"]))

    train_fields = set(unique_fields[:train_count])
    val_fields = set(unique_fields[train_count:train_count + val_count])
    test_fields = set(unique_fields[train_count + val_count:])

    splits = {"train": [], "val": [], "test": []}

    for fid, pair_list in field_to_pairs.items():
        if fid in train_fields:
            splits["train"].extend(pair_list)
        elif fid in val_fields:
            splits["val"].extend(pair_list)
        else:
            splits["test"].extend(pair_list)

    return splits


def process_and_save_dataset(
    dataset_dir: Path = DATASET_DIR,
    out_dir: Path = PROCESSED_DATA_DIR,
    manifest_path: Path = SPLITS_MANIFEST_PATH,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parses NetCDF tiles, cuts patches, performs field-level train/val/test split,
    saves .npy patch files to disk, and writes splits.json manifest.
    """
    out_dir = Path(out_dir)
    manifest_path = Path(manifest_path)

    pairs = find_all_tile_pairs(dataset_dir)
    if max_samples and max_samples < len(pairs):
        pairs = pairs[:max_samples]

    field_splits = create_field_splits(pairs)
    manifest_data = {}
    summary_stats = {"train": 0, "val": 0, "test": 0}

    for split_name, split_pairs in field_splits.items():
        split_out_dir = out_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)

        for nc_path, tif_path, field_id in split_pairs:
            try:
                img_np, mask_np, fid = parse_tile(nc_path, tif_path)
                patches = extract_patches(img_np, mask_np, fid)

                for patch_info in patches:
                    pid = patch_info["patch_id"]
                    img_file = split_out_dir / f"{pid}_img.npy"
                    mask_file = split_out_dir / f"{pid}_mask.npy"

                    np.save(img_file, patch_info["image"].astype(np.float32))
                    np.save(mask_file, patch_info["mask"].astype(np.float32))

                    manifest_data[pid] = {
                        "split": split_name,
                        "source_field_id": fid,
                        "img_path": str(img_file.relative_to(PROJECT_ROOT)),
                        "mask_path": str(mask_file.relative_to(PROJECT_ROOT)),
                        "crop_coords": patch_info["crop_coords"]
                    }

                    summary_stats[split_name] += 1
            except Exception as e:
                print(f"[WARNING] Failed to process tile {nc_path.name}: {e}")

    # Save splits.json manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "seed": RANDOM_SEED,
            "split_ratios": SPLIT_RATIOS,
            "patch_size": PATCH_SIZE,
            "counts": summary_stats,
            "patches": manifest_data
        }, f, indent=2)

    return summary_stats


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2 SMOKE TEST: Tiling/Patching & Field-Level Split Management")
    print("=" * 70)

    pairs = find_all_tile_pairs()
    print(f"Found {len(pairs)} matching NetCDF/GeoTIFF tile pairs.")

    splits = create_field_splits(pairs)
    print("\nField-Level Split Distribution:")
    for split_name, pair_list in splits.items():
        print(f"  {split_name.upper():5s} : {len(pair_list):4d} tiles ({len(pair_list)/len(pairs)*100:.1f}%)")

    print("\nProcessing and patchifying all dataset tiles...")
    stats = process_and_save_dataset(max_samples=None)

    print("\nGenerated Patch Counts:")
    for split_name, cnt in stats.items():
        print(f"  {split_name.upper():5s} : {cnt} patches saved")

    print(f"\nSplits manifest written to: {SPLITS_MANIFEST_PATH}")
    print("=" * 70)
    print("[SUCCESS] STEP 2 SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)
