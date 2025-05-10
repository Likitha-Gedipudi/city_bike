import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import zipfile
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from streamlit_folium import st_folium

from config import DATA_DIR
from inference import (
    get_model_predictions,
    load_batch_of_features_from_store,
    load_metrics_from_registry,
    load_model_from_registry,
)
from plot_utils import plot_aggregated_time_series

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False


def visualize_predicted_demand(shapefile_path, predicted_demand):
    """
    Visualizes the predicted number of rides on a map of NYC taxi zones.

    Parameters:
        shapefile_path (str): Path to the NYC taxi zones shapefile.
        predicted_demand (dict): A dictionary where keys are taxi zone IDs (or names)
                                and values are the predicted number of rides.

    Returns:
        None
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")

    # Ensure the taxi zone IDs in the shapefile match the keys in predicted_demand
    # Assuming the shapefile has a column 'zone_id' or 'LocationID' for taxi zones
    if "LocationID" not in gdf.columns:
        raise ValueError(
            "Shapefile must contain a 'LocationID' column to match taxi zones."
        )

    # Add a new column for predicted rides, defaulting to 0 if no prediction is available
    gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)

    # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(
        column="predicted_demand",  # Column to color by
        cmap="OrRd",  # Color map (e.g., 'OrRd' for orange-red gradient)
        linewidth=0.8,
        ax=ax,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
    )

    # Add title and labels
    ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
    ax.set_axis_off()  # Turn off axis for a cleaner map

    # Show the plot
    st.pyplot(fig)


import os
import tempfile

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium


def create_taxi_map(shapefile_path, prediction_data):
    """
    Create an interactive choropleth map of NYC taxi zones with predicted rides
    """
    # Load the NYC taxi zones shapefile
    nyc_zones = gpd.read_file(shapefile_path)

    # Merge with cleaned column names
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left",
    )

    # Fill NaN values with 0 for predicted demand
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)

    # Convert to GeoJSON for Folium
    nyc_zones = nyc_zones.to_crs(epsg=4326)

    # Create map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

    # Create color map
    colormap = LinearColormap(
        colors=[
            "#FFEDA0",
            "#FED976",
            "#FEB24C",
            "#FD8D3C",
            "#FC4E2A",
            "#E31A1C",
            "#BD0026",
        ],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max(),
    )

    colormap.add_to(m)

    # Define style function
    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(predicted_demand)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    # Convert GeoDataFrame to GeoJSON
    zones_json = nyc_zones.to_json()

    # Add the choropleth layer
    folium.GeoJson(
        zones_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style=(
                "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
            ),
        ),
    ).add_to(m)

    # Store the map in session state
    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m


def load_shape_data_file(
    data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True
):
    """
    Downloads, extracts, and loads a shapefile as a GeoDataFrame.

    Parameters:
        data_dir (str or Path): Directory where the data will be stored.
        url (str): URL of the shapefile zip file.
        log (bool): Whether to log progress messages.

    Returns:
        GeoDataFrame: The loaded shapefile as a GeoDataFrame.
    """
    # Ensure data directory exists
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    # Download the file if it doesn't already exist
    if not zip_path.exists():
        if log:
            print(f"Downloading file from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            with open(zip_path, "wb") as f:
                f.write(response.content)
            if log:
                print(f"File downloaded and saved to {zip_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")
    else:
        if log:
            print(f"File already exists at {zip_path}, skipping download.")

    # Extract the zip file if the shapefile doesn't already exist
    if not shapefile_path.exists():
        if log:
            print(f"Extracting files to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            if log:
                print(f"Files extracted to {extract_path}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file {zip_path}: {e}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

    # Load and return the shapefile as a GeoDataFrame
    if log:
        print(f"Loading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
        if log:
            print("Shapefile successfully loaded.")
        return gdf
    except Exception as e:
        raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")


# st.set_page_config(layout="wide")

current_date = pd.Timestamp.now(tz="Etc/UTC")
st.title(f"New York Yellow Taxi Cab Demand Next Hour")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 5


with st.spinner(text="Download shape file for taxi zones"):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)


with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Inference features fetched from the store")
    progress_bar.progress(2 / N_STEPS)


with st.spinner(text="Load model from registry"):
    model = load_model_from_registry()
    st.sidebar.write("Model was loaded from the registry")
    progress_bar.progress(3 / N_STEPS)

with st.spinner(text="Computing model predictions"):
    predictions = get_model_predictions(model, features)
    st.sidebar.write("Model predictions computed")
    progress_bar.progress(4 / N_STEPS)
    print(predictions)


shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

with st.spinner(text="Plot predicted rides demand"):
    # predictions_df = visualize_predicted_demand(
    #     shapefile_path, predictions["predicted_demand"]
    # )
    st.subheader("Taxi Ride Predictions Map")
    map_obj = create_taxi_map(shapefile_path, predictions)

    # Display the map
    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    # Display data statistics
    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Rides",
            f"{predictions['predicted_demand'].mean():.0f}",
        )
    with col2:
        st.metric(
            "Maximum Rides",
            f"{predictions['predicted_demand'].max():.0f}",
        )
    with col3:
        st.metric(
            "Minimum Rides",
            f"{predictions['predicted_demand'].min():.0f}",
        )

    # Show sample of the data
    st.sidebar.write("Finished plotting taxi rides demand")
    progress_bar.progress(5 / N_STEPS)

st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))
top10 = (
    predictions.sort_values("predicted_demand", ascending=False).head(10).index.tolist()
)
for location_id in top10:
    fig = plot_aggregated_time_series(
        features=features,
        targets=predictions["predicted_demand"],
        row_id=location_id,
        predictions=pd.Series(predictions["predicted_demand"]),
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
