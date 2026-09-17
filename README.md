# AI-Based Intelligent Cropland Monitoring & Crop Health Analysis

Unified Repository combining **Sentinel-2 CTHBNet Field Parcel Segmentation** with the **Crop Health Dashboard & PDF Report Compiler**.

Remote Repository: [https://github.com/Abhi327561/MajorProject.git](https://github.com/Abhi327561/MajorProject.git)

---

## 1. System Architecture

```text
                               AI Cropland Monitoring System
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      ▼                                                                             ▼
CTHBNet Field Parcel Segmentation                                         Crop Health Dashboard
(PyTorch / Sentinel-2 AI4Boundaries)                                      (Streamlit / Folium / Plotly / PDF)
      │                                                                             │
      ├─ NetCDF Satellite Band Extraction                                           ├─ Interactive Field Boundary Map (Folium)
      ├─ CNN-Transformer Hybrid Encoder                                             ├─ Temporal NDVI Trajectory & Anomaly Detector
      ├─ Boundary Guidance Loss Heads                                               ├─ Schema Normalization & Bounds Validator
      └─ GeoJSON Boundary Exporter                                                  └─ Downloadable Farmer Health PDF Reports
```

---

## 2. Directory Layout & Integrated Modules

```text
AI_Cropland_Monitoring/
│
├── crop_health_dashboard/        # Integrated Streamlit Crop Health Dashboard
│   ├── app.py                    # Streamlit dashboard interface
│   ├── config.py                 # Health thresholds, colors, paths, mapping aliases
│   ├── data/                     # Polygon boundaries & NDVI time-series
│   │   ├── mock_fields.geojson
│   │   └── mock_ndvi_timeseries.csv
│   └── src/
│       ├── data_loader.py        # Data ingestion & schema validator
│       ├── ndvi_analysis.py      # Trajectory analysis, deltas & anomaly detection
│       ├── report_generator.py   # ReportLab PDF report compiler
│       ├── mock_generator.py     # Standalone dataset generator
│       └── test_processing.py    # 14-test regression and unit test suite
│
├── models/ / model/              # CTHBNet CNN-Transformer Model Implementations
│   ├── architecture.py           # Remote CTHBNet architecture
│   ├── cthbnet.py                # Local CTHBNet model
│   ├── baseline_cnn.py           # U-Net encoder/decoder baselines
│   ├── boundary_guidance.py      # Spatial boundary guidance heads
│   ├── export_geojson.py         # Export predictions to GeoJSON
│   └── ndvi_analysis.py          # Spatial NDVI analysis tools
│
├── preprocessing/                # AI4Boundaries dataset preprocessing
├── training/                     # PyTorch trainer implementations
├── evaluation/                   # Evaluator and metric calculators
│
├── app.py                        # Master launcher (Dashboard + Model Hub)
├── requirements.txt              # Combined dependency manifest
└── README.md                     # System documentation
```

---

## 3. Installation & Quick Start

### Step 1: Environment Setup
Ensure Python 3.10+ is installed and activate your virtual environment:
```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Install combined dependencies
pip install -r requirements.txt
```

### Step 2: Run Verification Test Suite
Run the 14 automated regression tests covering schema mappings, spatial bounds, anomaly detection, and PDF generation:
```powershell
python crop_health_dashboard/src/test_processing.py
```

### Step 3: Launch Streamlit Dashboard & Model Hub
```powershell
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 4. Key Functional Features

1. **Interactive Field Map (Folium):** Automatically centers on field boundaries and dynamically colors polygons based on real-time health classifications (Green: Healthy, Amber: Moderate, Crimson: Poor).
2. **Plotly Time-Series Analytics:** Plots historical vegetation greenness trajectories with color-shaded health bands, highlighting cloud-obscured acquisitions and sudden stress events.
3. **Automated PDF Health Reports:** Compiles downloadable field health summaries with metadata, stress alerts, and actionable agronomic advice.
4. **CTHBNet Deep Learning Model:** Combines CNN local feature extraction with Vision Transformer global context for high-precision field parcel boundary segmentation.
5. **Schema Validation Layer:** Guarantees referential integrity, range checks, date formatting, and missing value checks across teammate model outputs.

---

## 5. Integration with Remote Repository

The project has been synchronized with the remote repository:
```powershell
git remote add origin https://github.com/Abhi327561/MajorProject.git
```
All components from `MajorProject` (`crop_health_dashboard` and `model` scripts) are fully combined with the local `AI_Cropland_Monitoring` codebase.
