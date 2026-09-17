import os
import pandas as pd
from pathlib import Path

DEFAULT_BASE = Path(__file__).resolve().parent / "dataset" / "sentinel2"
ENV_BASE = os.environ.get("DATASET_DIR")
if ENV_BASE:
    BASE = Path(ENV_BASE)
elif DEFAULT_BASE.exists():
    BASE = DEFAULT_BASE
else:
    BASE = Path(
        r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2"
    )

for split in ["train", "val", "test"]:

    csv_path = BASE / f"{split}.csv"

    df = pd.read_csv(csv_path)

    print("=" * 50)
    print(f"{split.upper()} DATASET")
    print("=" * 50)

    print("Number of samples:", len(df))
    print("Countries:")

    print(df["country"].value_counts())

    print()