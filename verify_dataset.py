from pathlib import Path
import os
import sys
import xarray as xr
import rasterio

# Dynamic base directory resolution: env var > relative path > cwd > Windows fallback
DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "dataset" / "sentinel2"
ENV_BASE_DIR = os.environ.get("DATASET_DIR")

if ENV_BASE_DIR:
    BASE_DIR = Path(ENV_BASE_DIR)
elif DEFAULT_BASE_DIR.exists():
    BASE_DIR = DEFAULT_BASE_DIR
elif (Path.cwd() / "dataset" / "sentinel2").exists():
    BASE_DIR = Path.cwd() / "dataset" / "sentinel2"
else:
    BASE_DIR = Path(r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2")


def verify_split(split_name, expected_count=None):
    """
    Safely verifies all .nc image files and matching .tif masks for a given dataset split.
    Checks that every .nc file can be opened via xarray and contains required bands without modifying any file.
    """
    split_dir = BASE_DIR / split_name
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    print("=" * 70)
    print(f"VERIFYING '{split_name.upper()}' SPLIT IN {BASE_DIR}")
    print("=" * 70)

    if not image_dir.exists():
        print(f" ERROR: Image directory '{image_dir}' does not exist!")
        return 0, 0, []

    image_files = sorted(list(image_dir.rglob("*.nc")))
    print(f"Total .nc images found: {len(image_files)}")
    if expected_count is not None:
        print(f"Expected image count   : {expected_count}")

    successful = 0
    failed = 0
    corrupted_files = []

    for idx, image_path in enumerate(image_files, 1):
        image_name = image_path.name
        country = image_path.parent.name

        mask_name = image_name.replace("_S2_10m_256.nc", "_S2label_10m_256.tif")
        mask_path = mask_dir / country / mask_name

        is_valid = True
        error_msg = ""

        # 1. Check if mask exists
        if not mask_path.exists():
            is_valid = False
            error_msg = f"Mask missing: {mask_path}"

        # 2. Check if .nc image can be opened via xarray
        if is_valid:
            try:
                ds = xr.open_dataset(image_path)
                # Verify required bands
                for b in ["B2", "B3", "B4", "B8", "NDVI"]:
                    if b not in ds:
                        is_valid = False
                        error_msg = f"Missing band {b} in {image_name}"
                        break
                h = ds.sizes.get("y")
                w = ds.sizes.get("x")
                ds.close()
            except Exception as e:
                is_valid = False
                error_msg = f"NetCDF open error on {image_name}: {e}"

        # 3. Check mask readability
        if is_valid:
            try:
                with rasterio.open(mask_path) as src:
                    mh = src.height
                    mw = src.width
                if h != mh or w != mw:
                    is_valid = False
                    error_msg = f"Dimension mismatch ({h}x{w} vs {mh}x{mw})"
            except Exception as e:
                is_valid = False
                error_msg = f"Mask open error on {mask_name}: {e}"

        if is_valid:
            successful += 1
        else:
            failed += 1
            corrupted_files.append((str(image_path), error_msg))
            print(f"  [CORRUPTED/INVALID] Sample {idx}/{len(image_files)}: {image_name} -> {error_msg}")

    print("-" * 70)
    print(f"{split_name.upper()} SUMMARY:")
    print(f"  Total Images Verified : {len(image_files)}")
    print(f"  Valid Image-Mask Pairs: {successful}")
    print(f"  Failed/Corrupted Files: {failed}")
    print("-" * 70)

    return successful, failed, corrupted_files


def main():
    print("=" * 70)
    print("NETCDF & DATASET VERIFICATION STEP")
    print("=" * 70)
    print(f"Dataset Path: {BASE_DIR.resolve()}\n")

    train_succ, train_fail, train_corr = verify_split("train", expected_count=500)
    val_succ, val_fail, val_corr = verify_split("val", expected_count=100)
    test_succ, test_fail, test_corr = verify_split("test", expected_count=100)

    total_images = train_succ + train_fail + val_succ + val_fail + test_succ + test_fail
    total_valid = train_succ + val_succ + test_succ
    total_failed = train_fail + val_fail + test_fail

    print("\n" + "=" * 70)
    print("OVERALL DATASET INTEGRITY REPORT")
    print("=" * 70)
    print(f" Total Samples Checked : {total_images}")
    print(f" Total Valid Pairs     : {total_valid}")
    print(f" Total Corrupted/Failed: {total_failed}")

    if total_failed == 0:
        print("\n [SUCCESS] ALL TRAIN (500), VAL (100), AND TEST (100) NETCDF & MASK FILES ARE READABLE AND VALID!")
    else:
        print(f"\n [WARNING] DETECTED {total_failed} CORRUPTED/INVALID FILES:")
        all_corrupted = train_corr + val_corr + test_corr
        for file_path, reason in all_corrupted:
            print(f"   - {file_path}: {reason}")

    print("=" * 70)


if __name__ == "__main__":
    main()