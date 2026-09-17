"""
Data Loader for Cropland Monitoring Dashboard.
Handles loading GeoJSON boundary features and NDVI time-series records,
computing monthly metrics (ndvi_delta, trend, health_label, is_anomaly, is_low_quality).
"""

import os
import json
import pandas as pd
import config

def get_data_directory():
    """Return data path depending on config.DATA_SOURCE ('mock' or 'real')."""
    if config.DATA_SOURCE == "real":
        return config.REAL_DATA_DIR
    return config.MOCK_DATA_DIR

def load_fields_geojson():
    """Load field boundary polygons from GeoJSON."""
    data_dir = get_data_directory()
    geojson_path = os.path.join(data_dir, config.FIELDS_FILENAME)
    
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Fields GeoJSON file not found at {geojson_path}. Run generate_mock_data.py first.")
        
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
        
    return geojson_data

def calculate_health_label(ndvi_val):
    """Categorize health based on NDVI thresholds defined in config.py."""
    if ndvi_val < config.NDVI_POOR_THRESHOLD:
        return "Poor"
    elif ndvi_val <= config.NDVI_MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Healthy"

def load_ndvi_data():
    """
    Load NDVI time series CSV and calculate month-over-month metrics:
    - ndvi_delta
    - trend ('Rising', 'Falling', 'Flat')
    - health_label ('Poor', 'Moderate', 'Healthy')
    - is_anomaly (True if drop >= config.ANOMALY_THRESHOLD)
    - is_low_quality (True if cloud_cover > config.CLOUD_COVER_THRESHOLD)
    """
    data_dir = get_data_directory()
    csv_path = os.path.join(data_dir, config.NDVI_FILENAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"NDVI CSV file not found at {csv_path}. Run generate_mock_data.py first.")

    df = pd.read_csv(csv_path)

    # Sort by field_id and month to ensure temporal order
    df = df.sort_values(by=["field_id", "month"]).reset_index(drop=True)

    # Compute month-over-month delta per field
    df["ndvi_delta"] = df.groupby("field_id")["ndvi"].diff().fillna(0.0).round(3)

    # Compute trend
    def get_trend(delta):
        if delta > 0.02:
            return "Rising"
        elif delta < -0.02:
            return "Falling"
        else:
            return "Flat"

    df["trend"] = df["ndvi_delta"].apply(get_trend)

    # Compute health label
    df["health_label"] = df["ndvi"].apply(calculate_health_label)

    # Anomaly flag (sharp drop in NDVI)
    df["is_anomaly"] = df["ndvi_delta"] <= -config.ANOMALY_THRESHOLD

    # Low quality flag (cloud cover above threshold)
    df["is_low_quality"] = df["cloud_cover"] > config.CLOUD_COVER_THRESHOLD

    return df

def get_field_options():
    """Return dictionary mapping field_id to field metadata."""
    geojson = load_fields_geojson()
    field_meta = {}
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        fid = props.get("field_id")
        if fid:
            field_meta[fid] = props
    return field_meta

if __name__ == "__main__":
    fields = load_fields_geojson()
    ndvi_df = load_ndvi_data()
    print("Loaded Fields:", len(fields["features"]))
    print(ndvi_df.head(12))
