import xarray as xr
import rasterio
import numpy as np
from pathlib import Path
from typing import Tuple, List

try:
    from preprocessing.config import (
        CANONICAL_BAND_ORDER,
        INCLUDE_NDVI,
        NORM_SCALE,
        DATASET_DIR
    )
except ImportError:
    CANONICAL_BAND_ORDER = ["B4", "B3", "B2", "B8"]
    INCLUDE_NDVI = False
    NORM_SCALE = 10000.0
    DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "sentinel2"


def parse_tile(
    nc_path: Path,
    tif_path: Path,
    bands: List[str] = CANONICAL_BAND_ORDER,
    include_ndvi: bool = INCLUDE_NDVI,
    norm_scale: float = NORM_SCALE
) -> Tuple[np.ndarray, np.ndarray, str]:

    """
    Parses one Sentinel-2 NetCDF tile and matching GeoTIFF mask.

    IMPORTANT:
    Preserves the temporal dimension.

    Output image shape:
        (C, T, H, W)

    Example:
        (5, 6, 256, 256)

    Channels:
        B4, B3, B2, B8, NDVI

    Mask:
        (H, W), binary {0.0, 1.0}
    """

    nc_path = Path(nc_path)
    tif_path = Path(tif_path)

    if not nc_path.exists():
        raise FileNotFoundError(
            f"NetCDF tile not found: {nc_path}"
        )

    if not tif_path.exists():
        raise FileNotFoundError(
            f"GeoTIFF mask not found: {tif_path}"
        )

    field_id = nc_path.name.split("_S2_")[0]

    # ---------------------------------------------------------
    # OPEN NETCDF
    # ---------------------------------------------------------

    ds = xr.open_dataset(nc_path)

    try:

        # -----------------------------------------------------
        # VALIDATE CRS
        # -----------------------------------------------------

        nc_crs_str = ""

        if "spatial_ref" in ds.data_vars:
            if "spatial_ref" in ds.spatial_ref.attrs:
                nc_crs_str = str(
                    ds.spatial_ref.attrs["spatial_ref"]
                )

        elif hasattr(ds, "crs"):
            nc_crs_str = str(ds.crs)

        # -----------------------------------------------------
        # BAND LIST
        # -----------------------------------------------------

        band_list = list(bands)

        if include_ndvi and "NDVI" not in band_list:
            band_list.append("NDVI")

        channel_arrays = []

        # -----------------------------------------------------
        # EXTRACT BANDS
        # -----------------------------------------------------

        for band_name in band_list:

            if band_name not in ds.data_vars:
                raise KeyError(
                    f"Band '{band_name}' not found in "
                    f"{nc_path.name}. "
                    f"Available: {list(ds.data_vars)}"
                )

            var_data = ds[band_name]

            # -------------------------------------------------
            # PRESERVE TIME DIMENSION
            # -------------------------------------------------

            if "time" in var_data.dims:

                data = var_data.values.astype(
                    np.float32
                )

                # Expected:
                # (6, 256, 256)

                if data.ndim != 3:
                    raise ValueError(
                        f"Unexpected shape for {band_name}: "
                        f"{data.shape}"
                    )

            else:

                # If a band has no time dimension,
                # create a temporal dimension of length 1.
                data = var_data.values.astype(
                    np.float32
                )

                if data.ndim != 2:
                    raise ValueError(
                        f"Unexpected shape for {band_name}: "
                        f"{data.shape}"
                    )

                data = data[np.newaxis, ...]

            # -------------------------------------------------
            # NORMALIZATION
            # -------------------------------------------------

            if band_name != "NDVI":

                # Sentinel-2 reflectance:
                # 0–10000 → 0–1

                data = data / float(norm_scale)

                data = np.clip(
                    data,
                    0.0,
                    1.0
                )

            else:

                # NDVI already in [-1, 1]
                data = np.clip(
                    data,
                    -1.0,
                    1.0
                )

                # Convert NDVI to [0,1] so all model inputs
                # have a consistent range.
                data = (data + 1.0) / 2.0

            # -------------------------------------------------
            # NaN / INF
            # -------------------------------------------------

            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            )

            channel_arrays.append(data)

        # -----------------------------------------------------
        # STACK
        # -----------------------------------------------------

        # Each channel:
        # (T,H,W)
        #
        # Result:
        # (C,T,H,W)

        image_np = np.stack(
            channel_arrays,
            axis=0
        ).astype(np.float32)

    finally:

        ds.close()

    # =========================================================
    # LOAD FIELD-EXTENT MASK
    # =========================================================

    with rasterio.open(tif_path) as src:

        # IMPORTANT:
        # Band 1 is the binary field-extent label.
        mask_np = src.read(1).astype(
            np.float32
        )

        mask_crs_str = (
            str(src.crs)
            if src.crs
            else ""
        )

    # ---------------------------------------------------------
    # CLEAN MASK
    # ---------------------------------------------------------

    mask_np = np.nan_to_num(
        mask_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Convert to binary
    mask_np = (
        mask_np > 0.5
    ).astype(np.float32)

    # =========================================================
    # SPATIAL VALIDATION
    # =========================================================

    _, _, img_h, img_w = image_np.shape

    mask_h, mask_w = mask_np.shape

    assert (img_h, img_w) == (
        mask_h,
        mask_w
    ), (
        f"Spatial dimension mismatch for {field_id}: "
        f"Image {(img_h, img_w)} vs "
        f"Mask {(mask_h, mask_w)}"
    )

    # =========================================================
    # CRS VALIDATION
    # =========================================================

    if nc_crs_str and mask_crs_str:

        nc_ok = (
            "3035" in nc_crs_str
            or "ETRS89" in nc_crs_str
            or "LAEA" in nc_crs_str
        )

        mask_ok = (
            "3035" in mask_crs_str
            or "ETRS89" in mask_crs_str
            or "LAEA" in mask_crs_str
        )

        assert nc_ok and mask_ok, (
            f"CRS mismatch for {field_id}: "
            f"NetCDF CRS = {nc_crs_str} | "
            f"Mask CRS = {mask_crs_str}"
        )

    return image_np, mask_np, field_id


if __name__ == "__main__":

    print("=" * 70)
    print("TESTING TEMPORAL NETCDF PARSER")
    print("=" * 70)

    nc_files = sorted(
        list(DATASET_DIR.rglob("*.nc"))
    )

    if not nc_files:
        raise FileNotFoundError(
            "No NetCDF files found."
        )

    nc_path = nc_files[0]

    country = nc_path.parent.name

    mask_name = nc_path.name.replace(
        "_S2_10m_256.nc",
        "_S2label_10m_256.tif"
    )

    tif_path = (
        DATASET_DIR.parent
        / "sentinel2"
        / "train"
        / "masks"
        / country
        / mask_name
    )

    print("NetCDF:", nc_path)
    print("Mask  :", tif_path)

    image, mask, field_id = parse_tile(
        nc_path,
        tif_path
    )

    print()
    print("Field ID:", field_id)
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print(
        "Image min/max:",
        image.min(),
        image.max()
    )

    print()
    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask values:", np.unique(mask))

    print()
    print("=" * 70)
    print("EXPECTED:")
    print("(5, 6, 256, 256)")
    print("(256, 256)")
    print("=" * 70)