"""
Mock Data Generator for Cropland Monitoring Dashboard.
Generates data/mock/fields.geojson (5 agricultural field boundary polygons)
and data/mock/ndvi.csv (10 months of NDVI and cloud cover data per field).
"""

import os
import json
import pandas as pd

import config

def ensure_dirs():
    os.makedirs(config.MOCK_DATA_DIR, exist_ok=True)
    os.makedirs(config.REAL_DATA_DIR, exist_ok=True)

def generate_fields_geojson():
    """Generate 5 field boundary polygons near an agricultural area (San Joaquin Valley, CA)."""
    fields_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F001",
                    "field_name": "North Wheat Polygon",
                    "area_ha": 45.2,
                    "crop_type": "Wheat"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-119.820, 36.450],
                        [-119.810, 36.450],
                        [-119.810, 36.442],
                        [-119.820, 36.442],
                        [-119.820, 36.450]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F002",
                    "field_name": "East Corn Plot",
                    "area_ha": 38.7,
                    "crop_type": "Corn"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-119.805, 36.450],
                        [-119.792, 36.450],
                        [-119.792, 36.442],
                        [-119.805, 36.442],
                        [-119.805, 36.450]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F003",
                    "field_name": "South Soybean Belt",
                    "area_ha": 52.1,
                    "crop_type": "Soybean"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-119.820, 36.438],
                        [-119.808, 36.438],
                        [-119.808, 36.428],
                        [-119.820, 36.428],
                        [-119.820, 36.438]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F004",
                    "field_name": "Central Cotton Field (Anomaly Test)",
                    "area_ha": 29.8,
                    "crop_type": "Cotton"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-119.805, 36.438],
                        [-119.792, 36.438],
                        [-119.792, 36.428],
                        [-119.805, 36.428],
                        [-119.805, 36.438]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F005",
                    "field_name": "West Alfalfa Tract",
                    "area_ha": 61.4,
                    "crop_type": "Alfalfa"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-119.835, 36.450],
                        [-119.823, 36.450],
                        [-119.823, 36.435],
                        [-119.835, 36.435],
                        [-119.835, 36.450]
                    ]]
                }
            }
        ]
    }

    filepath = os.path.join(config.MOCK_DATA_DIR, config.FIELDS_FILENAME)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(fields_geojson, f, indent=2)
    print(f"Generated {filepath}")

def generate_ndvi_csv():
    """
    Generate 10 months of NDVI time-series data for fields F001 to F005.
    F004 experiences a deliberate sharp drop at month 6 (0.68 -> 0.40, delta = -0.28) to test anomaly detection.
    Other fields feature normal seasonal senescence without drops >= 0.15.
    """
    months = [f"2024-{i:02d}" for i in range(1, 11)]

    # NDVI profiles across 10 months
    data = []

    # F001: Healthy wheat curve (Spring greenup, Summer peak, gradual harvest drop)
    f001_ndvi = [0.25, 0.38, 0.55, 0.72, 0.81, 0.78, 0.70, 0.62, 0.55, 0.48]
    f001_cloud = [5.0, 12.0, 18.0, 2.0, 8.0, 25.0, 4.0, 10.0, 15.0, 6.0] # month 6 high cloud cover

    # F002: Corn curve
    f002_ndvi = [0.20, 0.32, 0.48, 0.65, 0.76, 0.82, 0.75, 0.68, 0.60, 0.52]
    f002_cloud = [8.0, 6.0, 14.0, 9.0, 5.0, 11.0, 7.0, 22.0, 12.0, 8.0] # month 8 high cloud cover

    # F003: Soybean curve
    f003_ndvi = [0.22, 0.35, 0.50, 0.68, 0.75, 0.79, 0.72, 0.65, 0.57, 0.50]
    f003_cloud = [10.0, 15.0, 8.0, 6.0, 12.0, 9.0, 18.0, 14.0, 7.0, 5.0]

    # F004: Cotton field with SHARP ANOMALY drop at Month 6 (June)
    # Month 5: 0.68, Month 6: 0.40 -> Delta = -0.28 (>= 0.15 threshold -> Anomaly Flagged!)
    f004_ndvi = [0.21, 0.34, 0.52, 0.68, 0.68, 0.40, 0.35, 0.28, 0.22, 0.18]
    f004_cloud = [6.0, 8.0, 10.0, 5.0, 7.0, 12.0, 9.0, 6.0, 11.0, 14.0]

    # F005: Alfalfa curve (Multi-cut, high NDVI)
    f005_ndvi = [0.35, 0.48, 0.62, 0.78, 0.85, 0.82, 0.76, 0.70, 0.64, 0.58]
    f005_cloud = [4.0, 9.0, 11.0, 16.0, 8.0, 5.0, 10.0, 7.0, 9.0, 12.0]

    field_profiles = {
        "F001": (f001_ndvi, f001_cloud),
        "F002": (f002_ndvi, f002_cloud),
        "F003": (f003_ndvi, f003_cloud),
        "F004": (f004_ndvi, f004_cloud),
        "F005": (f005_ndvi, f005_cloud),
    }

    for field_id, (ndvi_list, cloud_list) in field_profiles.items():
        for month, ndvi_val, cloud_val in zip(months, ndvi_list, cloud_list):
            data.append({
                "field_id": field_id,
                "month": month,
                "ndvi": ndvi_val,
                "cloud_cover": cloud_val
            })

    df = pd.DataFrame(data)
    filepath = os.path.join(config.MOCK_DATA_DIR, config.NDVI_FILENAME)
    df.to_csv(filepath, index=False)
    print(f"Generated {filepath}")

if __name__ == "__main__":
    ensure_dirs()
    generate_fields_geojson()
    generate_ndvi_csv()
