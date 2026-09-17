import pandas as pd
import requests
from pathlib import Path

# ============================================================
# BASE PATH
# ============================================================

import os

DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "dataset" / "sentinel2"
ENV_BASE_DIR = os.environ.get("DATASET_DIR")
if ENV_BASE_DIR:
    BASE_DIR = Path(ENV_BASE_DIR)
elif DEFAULT_BASE_DIR.exists():
    BASE_DIR = DEFAULT_BASE_DIR
else:
    BASE_DIR = Path(
        r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2"
    )

# Number of pairs to download
SAMPLES = {
    "train": 500,
    "val": 100,
    "test": 100
}


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_file(url, output_path):

    # If file already exists, don't download again
    if output_path.exists():

        print("Already exists:", output_path.name)

        return True

    print("Downloading:", output_path.name)

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=60
        )

        if response.status_code != 200:

            print(
                "FAILED - HTTP status:",
                response.status_code
            )

            return False

        with open(output_path, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

        print("Downloaded successfully!")

        return True

    except Exception as e:

        print("Download error:", e)

        return False


# ============================================================
# DOWNLOAD ONE SPLIT
# ============================================================

def download_split(split_name, number_of_samples):

    print()
    print("=" * 70)
    print(
        f"STARTING {split_name.upper()} DATASET"
    )
    print(
        f"Number of pairs: {number_of_samples}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    csv_path = BASE_DIR / f"{split_name}.csv"

    df = pd.read_csv(csv_path)

    print(
        "Available samples:",
        len(df)
    )


    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    image_dir = (
        BASE_DIR /
        split_name /
        "images"
    )

    mask_dir = (
        BASE_DIR /
        split_name /
        "masks"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    successful = 0
    failed = 0


    # --------------------------------------------------------
    # Download samples
    # --------------------------------------------------------

    for index, row in df.head(number_of_samples).iterrows():

        image_name = row["image"]

        country = row["country"]


        # ----------------------------------------------------
        # Matching mask filename
        # ----------------------------------------------------

        mask_name = image_name.replace(
            "_S2_10m_256.nc",
            "_S2label_10m_256.tif"
        )


        # ----------------------------------------------------
        # Official URLs
        # ----------------------------------------------------

        image_url = (
            "https://jeodpp.jrc.ec.europa.eu/ftp/"
            "jrc-opendata/DRLL/AI4BOUNDARIES/"
            f"sentinel2/images/{country}/{image_name}"
        )

        mask_url = (
            "https://jeodpp.jrc.ec.europa.eu/ftp/"
            "jrc-opendata/DRLL/AI4BOUNDARIES/"
            f"sentinel2/masks/{country}/{mask_name}"
        )


        # ----------------------------------------------------
        # Local paths
        # ----------------------------------------------------

        image_output = (
            image_dir /
            country /
            image_name
        )

        mask_output = (
            mask_dir /
            country /
            mask_name
        )


        # Create country folders

        image_output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_output.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        print()
        print(
            f"[{index + 1}/{number_of_samples}]"
        )

        print(
            "Country:",
            country
        )

        print(
            "Image:",
            image_name
        )

        print(
            "Mask:",
            mask_name
        )


        # ----------------------------------------------------
        # Download image
        # ----------------------------------------------------

        image_ok = download_file(
            image_url,
            image_output
        )


        # ----------------------------------------------------
        # Download mask
        # ----------------------------------------------------

        mask_ok = download_file(
            mask_url,
            mask_output
        )


        # ----------------------------------------------------
        # Count pair
        # ----------------------------------------------------

        if image_ok and mask_ok:

            successful += 1

        else:

            failed += 1


    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"{split_name.upper()} COMPLETED"
    )

    print(
        "Successful pairs:",
        successful
    )

    print(
        "Failed pairs:",
        failed
    )

    print("-" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

for split, number in SAMPLES.items():

    download_split(
        split,
        number
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("ALL DOWNLOADS COMPLETED")
print("=" * 70)