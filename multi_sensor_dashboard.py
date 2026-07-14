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
st.set_page_config(page_title="Cloud-Linked Fatigue Dashboard", page_icon="🏗️", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏗️ Real-Time Bridge Health Monitoring</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563; margin-bottom: 30px;'>Automated Multi-Sensor Fatigue Life Dashboard</h3>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
st.sidebar.markdown("## 🌐 Data Source Configuration")

data_source = st.sidebar.radio("Choose Data Source:", ["Google Sheets Link", "Upload Local Files"])

uploaded_files = []
google_sheet_url = ""

if data_source == "Google Sheets Link":
    google_sheet_url = st.sidebar.text_input(
        "Paste Shared Google Sheets Link:",
        placeholder="https://docs.google.com/spreadsheets/d/.../edit?usp=sharing",
        help="Make sure the Google Sheet sharing setting is set to 'Anyone with the link can view'."
    )
else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload Strain Gauge Files (Excel/CSV):",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
        help="Upload separate telemetry data files for each strain gauge sensor."
    )

selected_cat = st.sidebar.selectbox(
    "Select Structural Eurocode Detail Category:",
    options=list(CATEGORY_DATABASE.keys()),
    format_func=lambda x: f"Category {x} ({CATEGORY_DATABASE[x]['name']})"
)

def convert_to_download_url(url):
    if "docs.google.com/spreadsheets" in url:
        if "/edit" in url:
            return url.split("/edit")[0] + "/export?format=xlsx"
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
# PIPELINE EVALUATION FUNCTION
# -------------------------------------------------------------
def compute_sensor_pipeline(df, sensor_name, cat_num):
    try:
        # Strip trailing/leading spaces from columns
        df.columns = df.columns.str.strip()
        
        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower()
            # Explicitly match your exact headers from the Excel sheet
            if col_lower in ['date time (utc+05:30)', 'datetime', 'date time', 'timestamp', 'time']:
                rename_dict[col] = 'Datetime'
            elif col_lower in ['strain (microstrain)', 'strain_raw', 'rawstrain', 'strain']:
                rename_dict[col] = 'RawStrain'
            elif col_lower in ['temperature (c)', 'temperature', 'temp', 't']:
                rename_dict[col] = 'Temperature'
                
        df = df.rename(columns=rename_dict)
        
        # Verify required columns are present after renaming
        for req_col in ['Datetime', 'RawStrain', 'Temperature']:
            if req_col not in df.columns:
                return None, f"Required columns missing in '{sensor_name}'. Expecting: 'Date Time (UTC+05:30)', 'Temperature (C)', 'Strain (microstrain)'. Found: {list(df.columns)}"

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

        dt = 15 * 60  # 15 mins sample rate
        fs = 1 / dt
        cutoff = 1 / (6 * 3600)
        b, a = butter(4, cutoff / (fs / 2), btype='high')
        df['LiveLoad'] = filtfilt(b, a, df['Detrended'])
        
        stress_history = (df['LiveLoad'] * 1e-6) * YOUNGS_MODULUS
        
        cycles = list(rainflow.extract_cycles(stress_history.to_numpy()))
        time_delta = (df['Datetime'].iloc[-1] - df['Datetime'].iloc[0]).total_seconds()
        duration_hours = time_delta / 3600.0
        
        threshold_stress = CATEGORY_DATABASE[cat_num]['threshold']
        sn_c = CATEGORY_DATABASE[cat_num]['constant_C']
        
        cycle_records = []
        total_damage = 0.0
        ignored_cycles = 0
        
        for rng, mean, count, _, _ in cycles:
            rng_corrected = rng / (1.0 - (mean / UTS)) if mean < UTS else rng
            if rng_corrected < threshold_stress:
                ignored_cycles += 1
                continue
                
            N_allowed = sn_c / (rng_corrected ** SN_M)
            damage_i = count / N_allowed
            total_damage += damage_i
            
            cycle_records.append({
                'Goodman_Corrected_Range_MPa': rng_corrected,
                'Cycle_Count': count,
                'Damage_Contribution_Di': damage_i
            })
            
        rul_hours = max(0.0, (duration_hours / total_damage) - duration_hours) if total_damage > 0 else float('inf')
        rul_years = rul_hours / (24.0 * 365.25) if rul_hours != float('inf') else float('inf')
        
        summary_metrics = {
            "sensor_name": sensor_name,
            "cat_num": cat_num,
            "threshold": threshold_stress,
            "time_delta": time_delta,
            "cycles_pass": len(cycle_records),
            "cycles_fail": ignored_cycles,
            "total_damage": total_damage,
            "rul_hours": "Infinite" if rul_hours == float('inf') else round(rul_hours, 2),
            "rul_years": "Infinite" if rul_years == float('inf') else round(rul_years, 3)
        }
        
        return {"df": df, "summary": summary_metrics, "spectra": pd.DataFrame(cycle_records)}, None
    except Exception as e:
        return None, f"Processing Error: {str(e)}"

# -------------------------------------------------------------
# DATA LOADING & EXECUTION
# -------------------------------------------------------------
all_results = {}
failed_sensors = []

if data_source == "Google Sheets Link" and google_sheet_url:
    download_url = convert_to_download_url(google_sheet_url)
    try:
        xls = pd.ExcelFile(download_url)
        for sheet_name in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            res, err = compute_sensor_pipeline(df_sheet, sheet_name, selected_cat)
            if err:
                failed_sensors.append((sheet_name, err))
            else:
                all_results[sheet_name] = res
    except Exception as e:
        st.error(f"Failed to pull live data from Google Sheets: {str(e)}")
        st.info("💡 Ensure that your Google Sheet sharing setting is set to 'Anyone with the link can view'.")

elif data_source == "Upload Local Files" and uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith(".xlsx"):
                df_file = pd.read_excel(file)
            else:
                df_file = pd.read_csv(file)
            sensor_name = file.name.split('.')[0]
            res, err = compute_sensor_pipeline(df_file, sensor_name, selected_cat)
            if err:
                failed_sensors.append((sensor_name, err))
            else:
                all_results[sensor_name] = res
        except Exception as e:
            failed_sensors.append((file.name, str(e)))

# Show helper prompts if no data is loaded
if not all_results and not failed_sensors:
    if data_source == "Google Sheets Link":
        st.info("💡 Paste your shared Google Sheets link in the left panel to begin cloud-synchronized monitoring.")
    else:
        st.info("💡 Upload one or more strain gauge files (.xlsx or .csv) in the left panel to begin manual inspection.")

# Display failures if any
if failed_sensors:
    for sname, err in failed_sensors:
        st.error(f"Error processing sensor/tab '{sname}': {err}")

# Render Dashboard if results exist
if all_results:
    # Build comparison summary dataframe
    summary_rows = []
    for sname, data in all_results.items():
        summary_rows.append({
            "Sensor Name": sname,
            "Cumulative Damage (D)": f"{data['summary']['total_damage']:.2e}" if (0 < data['summary']['total_damage'] < 0.001) else f"{data['summary']['total_damage']:.5f}",
            "Remaining Life (Years)": data['summary']['rul_years'],
            "Remaining Life (Hours)": data['summary']['rul_hours'],
            "Logged Active Cycles": data['summary']['cycles_pass']
        })
    summary_df = pd.DataFrame(summary_rows)

    # 1. Master Multi-Sensor Comparison Panel
    st.markdown("## 📊 Sensor Fleet Fatigue Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Detail Analysis Dropdown for Individual Sensors
    st.markdown("## 🔍 Individual Sensor In-depth Analysis")
    selected_sensor = st.selectbox(
        "Select Sensor to inspect details:",
        options=list(all_results.keys())
    )

    sensor_data = all_results[selected_sensor]
    df_live = sensor_data["df"]
    metrics = sensor_data["summary"]
    spectra_df = sensor_data["spectra"]

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Structural Classification", value=f"FAT-{metrics['cat_num']}", delta=f"Threshold: {metrics['threshold']} MPa", delta_color="inverse")
    with col2:
        d_val = metrics['total_damage']
        val_str = f"{d_val:.2e}" if (0 < d_val < 0.001) else f"{d_val:.5f}"
        st.metric(label="Cumulative Damage (D)", value=val_str)
    with col3:
        st.metric(label="Remaining Fatigue Life", value=f"{metrics['rul_years']} Years", delta=f"{metrics['rul_hours']} Hours Total")
    with col4:
        st.metric(label="Active Load Cycles Logged", value=metrics['cycles_pass'])

    # Time-series graph
    st.markdown(f"### 📈 Live Dynamic Working Load Stress History: **{selected_sensor}**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_live['Datetime'], y=df_live['LiveLoad'], mode='lines', name='Live Dynamic Stress (MPa)', line=dict(color='#2563EB', width=1.5)))
    fig.update_layout(xaxis_title="Measurement Timeline", yaxis_title="Stress (MPa)", hovermode="x unified", margin=dict(l=40, r=40, t=10, b=40), height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Log & Histogram
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown(f"### 📋 Live Telemetry Log (Latest Records: {selected_sensor})")
        st.dataframe(df_live.tail(100).sort_values('Datetime', ascending=False), height=250, use_container_width=True)
    with right:
        st.markdown(f"### 📊 Rainflow Range Count Distribution Histogram ({selected_sensor})")
        if not spectra_df.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=spectra_df['Goodman_Corrected_Range_MPa'], nbinsx=15, marker_color='#DC2626', opacity=0.75))
            fig_hist.update_layout(xaxis_title="Goodman Corrected Range (MPa)", yaxis_title="Frequency Count", margin=dict(l=40, r=40, t=10, b=40), height=250, template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("No stress cycles have crossed the structural detail threshold yet.")
