import pandas as pd
import requests
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = Path(
    r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2\train.csv"
)

IMAGE_DIR = Path(
    r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2\images"
)

MASK_DIR = Path(
    r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2\masks"
)

# Number of image-mask pairs to download
NUMBER_OF_SAMPLES = 20


# ============================================================
# CREATE FOLDERS
# ============================================================

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ TRAINING CSV
# ============================================================

print("Reading train.csv...")

df = pd.read_csv(CSV_PATH)

print("Total training samples available:", len(df))
print("Samples we will download:", NUMBER_OF_SAMPLES)


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_file(url, output_path):

    # Don't download if file already exists
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

        # Save file
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
# DOWNLOAD FIRST 20 TRAINING SAMPLES
# ============================================================

successful = 0
failed = 0

for index, row in df.head(NUMBER_OF_SAMPLES).iterrows():

    # --------------------------------------------------------
    # Get information from CSV
    # --------------------------------------------------------

    image_name = row["image"]
    country = row["country"]

    # Example:
    # AT_1433_S2_10m_256.nc
    #
    # becomes:
    # AT_1433_S2label_10m_256.tif

    mask_name = image_name.replace(
        "_S2_10m_256.nc",
        "_S2label_10m_256.tif"
    )


    # --------------------------------------------------------
    # Create official JRC URLs
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Create local output paths
    # --------------------------------------------------------

    image_output = (
        IMAGE_DIR /
        country /
        image_name
    )

    mask_output = (
        MASK_DIR /
        country /
        mask_name
    )


    # --------------------------------------------------------
    # Create country folders
    # --------------------------------------------------------

    image_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Display current sample
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"Sample {index + 1}/{NUMBER_OF_SAMPLES}"
    )
    print("Country :", country)
    print("Image   :", image_name)
    print("Mask    :", mask_name)
    print("=" * 60)


    # --------------------------------------------------------
    # Download image
    # --------------------------------------------------------

    image_ok = download_file(
        image_url,
        image_output
    )


    # --------------------------------------------------------
    # Download mask
    # --------------------------------------------------------

    mask_ok = download_file(
        mask_url,
        mask_output
    )


    # --------------------------------------------------------
    # Count successful/failed pairs
    # --------------------------------------------------------

    if image_ok and mask_ok:

        successful += 1

    else:

        failed += 1


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("DOWNLOAD COMPLETED")
print("=" * 60)

print(
    "Successful image-mask pairs:",
    successful
)

print(
    "Failed image-mask pairs:",
    failed
)

print("=" * 60)