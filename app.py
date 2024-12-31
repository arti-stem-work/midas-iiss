import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from scipy.signal import savgol_filter, detrend
from PIL import Image
import subprocess
import os
import tempfile
import base64

def add_background_image(image_file, opacity=0.12):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
        background_style = f"""
        <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), url(data:image/jpeg;base64,{encoded_string});
                background-size: 100vw 100vh;
                background-repeat: no-repeat;
                background-position: top left;
                background-attachment: local;
            }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

# Path to the R script
r_script_path = "install_hexView.R"
subprocess.run(["Rscript", r_script_path], check=True)

try:
    result = subprocess.run(["Rscript", "--version"], capture_output=True, text=True)
    print("Rscript Version:", result.stdout)
except Exception as e:
    print("Error checking Rscript version:", e)

model = joblib.load('multi_output_stacking_model.pkl')
scaler = joblib.load('preprocess_pipeline.pkl')

st.set_page_config(page_title="MIDAS", layout="wide")

add_background_image("background_image.jpg", opacity=0.12)

icar_logo = Image.open("ICAR Logo.png").resize((120, 120))
iiss_logo = Image.open("IISS Logo.png").resize((120, 120))
icraf_logo = Image.open("ICRAF Logo.png").resize((120, 120))
data_format = Image.open("Data Format.png")

col1, col2, col3, col4 = st.columns([1, 1, 6, 1])
with col1:
    st.image(icar_logo)
with col2:
    st.image(iiss_logo)
with col3:
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Edu VIC WA NT Beginner:wght@500&display=swap');
            h2 {
                font-family: 'Edu VIC WA NT Beginner', serif;
                text-align: center;
                font-size: 46px;
                text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff; /* White outline */
                font-weight: bold;
                margin-bottom: 10px;
                color: #092922;
            }
        </style>
        <h2>Mid-Infrared Spectroscopy Analysis System (MIDAS)<br> for Indian Agroecosystem</h2>
    """, unsafe_allow_html=True)

with col4:
    st.image(icraf_logo)

with st.popover("Read Instructions."):
    st.markdown("""
        ### Instructions
        1. Upload OPUS files. The system will convert them into a CSV.
        2. Ensure the spectral files are named using a prefix 3-letter followed by a six-digit number.
        3. The Spectra CSV created will have the following columns representing the spectra wavelengths (MIR) and rows representing the measurements.
    """)
    st.image(data_format, caption='Example of the Spectra CSV format')

# Organize tabs within a column for custom width control
main_col = st.columns([1, 6, 1])[1]  # Center the column with custom width

with main_col:
    tabs = st.tabs(["   📈 **SPECTRAL DATA**   ", "   🗃 **MODEL OUTPUT**   "])

    with tabs[0]:
        st.markdown('<span class="custom-file-uploader">Upload your OPUS files.</span>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("", type=[".0"], accept_multiple_files=True)

        if uploaded_files:
            max_file_size_mb = 10
            valid_files = []
            for uploaded_file in uploaded_files:
                if uploaded_file.size > max_file_size_mb * 1024 * 1024:
                    st.warning(f"File {uploaded_file.name} exceeds the maximum size of {max_file_size_mb} MB and will not be processed.")
                else:
                    valid_files.append(uploaded_file)

            if valid_files:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for valid_file in valid_files:
                        file_path = os.path.join(temp_dir, valid_file.name)
                        with open(file_path, "wb") as f:
                            f.write(valid_file.read())

                    r_script = "convert2csv.R"
                    output_csv = os.path.join(temp_dir, "Spectra.csv")
                    subprocess.run(["Rscript", r_script, temp_dir, temp_dir], check=True)

                    data = pd.read_csv(output_csv)

                    ids = data.iloc[:, 0]
                    spectra = data.iloc[:, 6:]

                    expected_columns = 1714
                    if spectra.shape[1] > expected_columns:
                        spectra = spectra.iloc[:, :expected_columns]
                    elif spectra.shape[1] < expected_columns:
                        padding = np.zeros((spectra.shape[0], expected_columns - spectra.shape[1]))
                        spectra = np.hstack([spectra, padding])
                        spectra = pd.DataFrame(spectra)

                    number_input_style = """
                    <style>
                    .st-eu {
                        font-weight: bold;
                        color: #000000; /* Jet black */
                        font-size: 28px;
                    }
                    </style>
                    """
                    st.markdown(number_input_style, unsafe_allow_html=True)

                    num_rows = st.number_input(
                        "**Enter the number of rows to preview spectral data:**",
                        min_value=1,
                        max_value=spectra.shape[0],
                        value=1,
                        step=1
                    )

                    selected_spectra = spectra.iloc[:num_rows, :]
                    wavelengths = selected_spectra.columns

                    fig = go.Figure()
                    for i in range(num_rows):
                        fig.add_trace(go.Scatter(
                            x=wavelengths,
                            y=selected_spectra.iloc[i, :],
                            mode='lines',
                            name=f"{ids.iloc[i]}",
                            line=dict(width=2)
                        ))
                    fig.update_layout(
                        title="Spectra Plot",
                        xaxis_title="Wavelength",
                        yaxis_title="Absorbance",
                        legend_title="Sample ID",
                        margin=dict(l=0, r=0, t=30, b=0),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig)

    with tabs[1]:
        property_selection = st.selectbox(
            'Select soil property to make predictions:',
            ['Cu', 'Zn', 'Fe', 'Mn', 'All']
        )

        if uploaded_files:
            X_smoothed = savgol_filter(spectra, window_length=11, polyorder=2, axis=1)
            X_corrected = detrend(X_smoothed, axis=1)

            spectra_scaled = scaler.transform(X_corrected)
            predictions = model.predict(spectra_scaled)

            if property_selection == 'All':
                results = pd.DataFrame(predictions, columns=['Cu', 'Zn', 'Fe', 'Mn'])
            else:
                property_index = ['Cu', 'Zn', 'Fe', 'Mn'].index(property_selection)
                results = pd.DataFrame(predictions[:, property_index], columns=[property_selection])

            results.insert(0, 'ID', ids)
            st.markdown("### Prediction Results")
            st.write(results)

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download as CSV File.", data=csv, file_name='predictions.csv', mime='text/csv')
