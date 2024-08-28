import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from scipy.signal import savgol_filter, detrend
from PIL import Image
import tempfile
import os
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.rinterface import RScript

# Function to check and install R packages
def install_r_package(package_name):
    try:
        robjects.r(f'if (!requireNamespace("{package_name}", quietly = TRUE)) install.packages("{package_name}")')
    except RRuntimeError as e:
        st.error(f"An error occurred while installing R package {package_name}: {e}")

# Install required R packages
install_r_package("hexView")

# Load your pre-trained model and pre-processing pipeline
model = joblib.load('multi_output_stacking_model.pkl')
scaler = joblib.load('preprocess_pipeline.pkl')

# Set page configuration
st.set_page_config(page_title="MIDAS", layout="wide")

# Add custom CSS to hide the "View Full Screen" option
hide_streamlit_style = """
    <style>
        button[title="View fullscreen"] {
            visibility: hidden;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load and resize the images
icar_logo = Image.open("ICAR Logo.png")
iiss_logo = Image.open("IISS Logo.png")
icraf_logo = Image.open("ICRAF Logo.png")
data_format = Image.open("Data Format.png")

# Resize the images to the same dimensions
icar_logo = icar_logo.resize((120, 120))
iiss_logo = iiss_logo.resize((120, 120))
icraf_logo = icraf_logo.resize((180, 120))

# Layout the title and logos using columns
col1, col2, col3, col4 = st.columns([1, 1, 6, 1])
with col1:
    st.image(icar_logo, use_column_width=False)
with col2:
    st.image(iiss_logo, use_column_width=False)
with col3:
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Edu VIC WA NT Beginner:wght@500&display=swap');
            h2 {
                font-family: 'Edu VIC WA NT Beginner', serif;
                text-align: center;
                font-size: 48px;
                text-shadow: 2px 2px 2px #aaa;
                font-weight: bold;
                margin-bottom: 10px;
                color: #092922;
            }
        </style>
        <h2>Mid-infrared (MIR) Spectroscopy Analysis System </h2>
    """, unsafe_allow_html=True)
with col4:
    st.image(icraf_logo, use_column_width=False)

# Sidebar with instructions and data example
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Upload OPUS files. The system will convert them into a CSV.
2. Ensure the spectral files are named using a prefix 3-letter followed by a six-digit number.
3. The Spectra CSV created will have following columns representing the spectra wavelengths (MIR), while rows representing the measurements.
""")

st.sidebar.image(data_format, caption='Example of the Spectra CSV format', use_column_width=True)

# Dropdown to select which property to predict
property_selection = st.selectbox(
    'Select soil property to make predictions:',
    ['Cu', 'Zn', 'Fe', 'Mn', 'All']
)

# File uploader for OPUS files
uploaded_files = st.file_uploader("Upload your OPUS files.", type=[".0"], accept_multiple_files=True)

if uploaded_files:
    # Create a temporary directory to save uploaded OPUS files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save each uploaded file to the temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        # Path to the R script
        r_script_path = "convert2csv.R"

        # Execute the R script using rpy2
        try:
            r_script = RScript(r_script_path)
            r_script.run([temp_dir, temp_dir])
        except RRuntimeError as e:
            st.error(f"An error occurred while running the R script: {e}")
            st.stop()

        # Load the converted CSV
        output_csv = os.path.join(temp_dir, "Spectra.csv")
        data = pd.read_csv(output_csv)
        
        # Ensure the first column is treated as ID
        ids = data.iloc[:, 0]
        spectra = data.iloc[:, 6:]

        # Handle varied number of columns by truncating or padding
        expected_columns = 1714  # Change to the expected number of features
        if spectra.shape[1] > expected_columns:
            spectra = spectra.iloc[:, :expected_columns]
        elif spectra.shape[1] < expected_columns:
            # Padding with zeros (or any other value like mean) for missing columns
            padding = np.zeros((spectra.shape[0], expected_columns - spectra.shape[1]))
            spectra = np.hstack([spectra, padding])
            spectra = pd.DataFrame(spectra)  # Convert back to DataFrame for easier handling

        # Number input to select the number of rows to plot
        num_rows = st.number_input("Enter the number of rows to preview spectral data:", min_value=1, max_value=spectra.shape[0], value=1, step=1)

        # Layout the buttons in a single row with improved spacing
        st.markdown("""
            <style>
                .stButton button {
                    width: 180px;
                    height: 40px;
                    font-size: 26px;
                    font-weight: bold;
                    border-radius: 5px;
                    margin: 0 10px;
                }
                .button-container {
                    display: flex;
                    justify-content: center;
                    margin: 40px 0;  
                }
            </style>
        """, unsafe_allow_html=True)

        # Container for buttons
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            plot_button = st.button("Preview Spectral Data")

        with col3:
            run_model_button = st.button("Run the Model")

        if plot_button:
            selected_spectra = spectra.iloc[:num_rows, :]

            # Extracting the wavelength headings from the columns
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
                template='plotly_white',  # Optional: 'plotly_dark' for dark mode

            )
            st.plotly_chart(fig)

        if run_model_button:
            # Apply Savitzky-Golay filter and baseline correction
            X_smoothed = savgol_filter(spectra, window_length=11, polyorder=2, axis=1)
            X_corrected = detrend(X_smoothed, axis=1)

            # Scale the data using the pre-trained scaler
            spectra_scaled = scaler.transform(X_corrected)

            # Make predictions using the model
            predictions = model.predict(spectra_scaled)

            # Create a DataFrame for the results
            if property_selection == 'All':
                result_df = pd.DataFrame(predictions, columns=['Cu', 'Zn', 'Fe', 'Mn'])
            else:
                property_index = ['Cu', 'Zn', 'Fe', 'Mn'].index(property_selection)
                results = pd.DataFrame(predictions[:, property_index], columns=[property_selection])
            
            results.insert(0, 'ID', ids)
            
            # Display the results
            st.markdown("### Prediction Results")
            st.write(results)
            
            # Option to download the results
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download the predictions as CSV File", data=csv, file_name='predictions.csv', mime='text/csv')
