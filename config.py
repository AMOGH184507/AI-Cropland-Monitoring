"""
Configuration file for AI-Based Intelligent Cropland Monitoring & Crop Health Analysis.
All threshold settings, colors, and dataset paths are controlled here.
"""

import os

# Data Source Selector ("mock" or "real")
# Change this single line to switch between mock data and real teammate output
DATA_SOURCE = "mock"

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_DATA_DIR = os.path.join(BASE_DIR, "data", "mock")
REAL_DATA_DIR = os.path.join(BASE_DIR, "data", "real")

FIELDS_FILENAME = "fields.geojson"
NDVI_FILENAME = "ndvi.csv"

# Health Classification Thresholds
# Poor: NDVI < 0.3
# Moderate: 0.3 <= NDVI <= 0.6
# Healthy: NDVI > 0.6
NDVI_POOR_THRESHOLD = 0.3
NDVI_MODERATE_THRESHOLD = 0.6

# Anomaly Detection Threshold
# A month-over-month drop of >= 0.15 in NDVI indicates a sharp health anomaly
ANOMALY_THRESHOLD = 0.15

# Cloud Cover Quality Threshold
# Passes with cloud cover > 20.0% are flagged as low quality
CLOUD_COVER_THRESHOLD = 20.0

# Health Status Colors (Hex Format)
HEALTH_COLORS = {
    "Poor": "#E74C3C",     # Red / Crimson
    "Moderate": "#F39C12", # Orange / Gold
    "Healthy": "#2ECC71"   # Emerald Green
}

# Trend Indicator Symbols / Labels
TREND_LABELS = {
    "Rising": "↗ Rising",
    "Falling": "↘ Falling",
    "Flat": "→ Flat"
}
