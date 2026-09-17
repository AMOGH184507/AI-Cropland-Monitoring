"""
AI-Based Intelligent Cropland Monitoring & Crop Health Analysis
Combined Master Application & Dashboard Launcher
"""

import sys
import os
import streamlit as st

# Add current folder and crop_health_dashboard folder to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(ROOT_DIR, "crop_health_dashboard")
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import crop health dashboard module
try:
    from crop_health_dashboard import app as dashboard_app
    from crop_health_dashboard.src import test_processing
    DASHBOARD_AVAILABLE = True
except Exception as e:
    DASHBOARD_AVAILABLE = False
    DASHBOARD_ERROR = str(e)


def run_model_hub():
    st.title("🧠 CTHBNet Model & Processing Pipeline Hub")
    st.markdown("---")

    st.subheader("1. Architecture Overview")
    st.markdown("""
    **CTHBNet** is a Convolutional Neural Network - Vision Transformer Hybrid model engineered specifically for field boundary parcel segmentation from multi-spectral temporal satellite imagery (Sentinel-2).

    - **CNN Encoder Branch:** Extracts high-resolution spatial feature maps for local detail & field edges.
    - **Transformer Branch:** Captures long-range spatial correlations and field geometry across large coverage regions.
    - **Boundary Guidance Module:** Uses dedicated loss maps to supervise precise boundary prediction.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input Channels", "4 (R, G, B, NIR)")
    with col2:
        st.metric("Patch Resolution", "256 x 256")
    with col3:
        st.metric("Temporal Bands", "6 Time Steps")

    st.divider()

    st.subheader("2. Available Model Scripts & Commands")
    st.markdown("""
    | Script | Path | Purpose |
    | :--- | :--- | :--- |
    | **Train Model** | `train.py` | Train CTHBNet on AI4Boundaries dataset patches |
    | **Evaluate Model** | `evaluate.py` | Compute IoU, F1-Score, and Boundary Dice metrics |
    | **Export GeoJSON** | `model/export_geojson.py` | Export predicted parcel boundaries to GeoJSON |
    | **NDVI Analysis** | `model/ndvi_analysis.py` | Compute multi-temporal NDVI & progression stats |
    | **Unit Tests** | `crop_health_dashboard/src/test_processing.py` | Run 14-test regression and schema validation suite |
    """)

    st.divider()
    st.subheader("3. Verification Test Suite Status")
    if st.button("🧪 Run Crop Health Test Suite Now"):
        with st.spinner("Running 14 automated unit and regression tests..."):
            try:
                import unittest
                suite = unittest.TestLoader().loadTestsFromModule(test_processing)
                runner = unittest.TextTestRunner(verbosity=2)
                result = runner.run(suite)
                if result.wasSuccessful():
                    st.success(f"✓ All {result.testsRun} tests passed successfully!")
                else:
                    st.error(f"❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests.")
            except Exception as ex:
                st.error(f"Error executing test suite: {ex}")


def main():
    # Sidebar navigation mode selector
    st.sidebar.title("🌱 AI Cropland Monitor")
    app_mode = st.sidebar.radio(
        "Navigation Mode",
        ["🌾 Crop Health Dashboard", "🧠 CTHBNet Model Hub"]
    )
    st.sidebar.divider()

    if app_mode == "🌾 Crop Health Dashboard":
        if DASHBOARD_AVAILABLE:
            dashboard_app.main()
        else:
            st.error(f"Could not load Crop Health Dashboard: {DASHBOARD_ERROR}")
    else:
        run_model_hub()


if __name__ == "__main__":
    main()
