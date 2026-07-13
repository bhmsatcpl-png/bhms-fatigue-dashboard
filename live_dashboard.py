import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
from sklearn.linear_model import LinearRegression
import rainflow
import plotly.graph_objects as go

# =============================================================
# CONSTANTS & METADATA CONFIGURATION
# =============================================================
YOUNGS_MODULUS = 210000  # MPa (Steel)
UTS = 460  # MPa
SN_M = 3.0  # Fatigue strength exponent

CATEGORY_DATABASE = {
    160: {"name": "Non-welded base metal / rolled profiles", "threshold": 40.0, "constant_C": 8.19e12},
    125: {"name": "Continuous longitudinal welds", "threshold": 32.0, "constant_C": 3.91e12},
    90:  {"name": "Transverse butt welds in plate configurations", "threshold": 23.0, "constant_C": 1.46e12},
    71:  {"name": "Attachments / Welded gussets on flanges", "threshold": 18.0, "constant_C": 7.16e11},
    50:  {"name": "Load-carrying cruciform joints", "threshold": 13.0, "constant_C": 2.50e11}
}

# Streamlit App UI Branding
st.set_page_config(page_title="Live BHMS Fatigue Dashboard", page_icon="🏗️", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏗️ Real-Time Bridge Health Monitoring</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563; margin-bottom: 30px;'>Automated Cloud-Linked Structural Fatigue Dashboard</h3>", unsafe_allow_html=True)

# -------------------------------------------------------------
# CLOUD REPOSITORY / DATA LINK SETUP
# -------------------------------------------------------------
st.sidebar.markdown("## 🌐 Live Data Feed Configuration")

DEFAULT_CLOUD_URL = "https://docs.google.com/spreadsheets/d/1X-YOUR_REAL_SHEET_ID_HERE/edit?usp=sharing"

cloud_link_input = st.sidebar.text_input(
    "Cloud Excel/CSV Shared Link:", 
    value=DEFAULT_CLOUD_URL,
    help="Provide the shared view link from Google Sheets or OneDrive."
)

selected_cat = st.sidebar.selectbox(
    "Select Structural Eurocode Detail Category:",
    options=list(CATEGORY_DATABASE.keys()),
    format_func=lambda x: f"Category {x} ({CATEGORY_DATABASE[x]['name']})"
)

auto_refresh_sec = st.sidebar.slider("Auto-Refresh Intermission Interval (Seconds):", min_value=10, max_value=600, value=60)

def convert_to_download_url(url):
    if "docs.google.com/spreadsheets" in url:
        if "/edit" in url:
            return url.split("/edit")[0] + "/export?format=xlsx"
    if "onedrive.live.com" in url and "download?" not in url:
        return url.replace("redir?", "download?").replace("1drv.ms", "onedrive.live.com/download")
    return url

def hampel_filter_notebook(series, window=5, n=3):
    new_series = series.copy()
    L = 1.4826
    for i in range(window, len(series) - window):
        x = series[i - window : i + window]
        median = np.median(x)
        mad = L * np.median(np.abs(x - median))
        if mad > 0 and np.abs(series[i] - median) > n * mad:
            new_series[i] = median
    return new_series

# -------------------------------------------------------------
# RUN-TIME CALCULATIONS & PIPELINE EVALUATION
# -------------------------------------------------------------
@st.cache_data(ttl=auto_refresh_sec)
def fetch_and_compute_pipeline(url, cat_num):
    download_url = convert_to_download_url(url)
    
    try:
        if "format=xlsx" in download_url or download_url.endswith(".xlsx"):
            df = pd.read_excel(download_url)
        else:
            df = pd.read_csv(download_url)
    except Exception as e:
        return None, f"Failed to pull live data from cloud source: {str(e)}"
        
    df.columns = df.columns.str.strip()
    
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['datetime', 'date time', 'timestamp', 'time']:
            rename_dict[col] = 'Datetime'
        elif col_lower in ['strain_raw', 'rawstrain', 'strain']:
            rename_dict[col] = 'RawStrain'
        elif col_lower in ['temperature', 'temp', 't']:
            rename_dict[col] = 'Temperature'
            
    df = df.rename(columns=rename_dict)
    
    for req_col in ['Datetime', 'RawStrain', 'Temperature']:
        if req_col not in df.columns:
            return None, f"Required column mismatch. Sheet headers found: {list(df.columns)}"

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    df['RawStrain'] = pd.to_numeric(df['RawStrain'], errors='coerce').interpolate()
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce').interpolate()
    
    df = df[(df['Temperature'] > -20) & (df['Temperature'] < 80)].copy().reset_index(drop=True)

    df['STRAIN'] = df['RawStrain']
    df['Digits'] = df['RawStrain'] * 0.1  
    df['Median'] = medfilt(df['RawStrain'], kernel_size=3)
    df['Hampel'] = hampel_filter_notebook(df['Median'].values, window=5, n=3)
    df['Smooth'] = savgol_filter(df['Hampel'], 11, 2)

    X_reg = df[['Temperature']]
    y_reg = df['Smooth']
    model = LinearRegression().fit(X_reg, y_reg)
    df['ThermalStrain'] = model.predict(X_reg)
    df['TempCorrected'] = df['Smooth'] - df['ThermalStrain']

    df['Trend'] = df['TempCorrected'].rolling(window=96, center=True).mean().ffill().bfill()
    df['Detrended'] = df['TempCorrected'] - df['Trend']

    dt = 15 *
