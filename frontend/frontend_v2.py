import sys
from pathlib import Path
import zipfile
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# ------------------ SESSION STATE ------------------
if "map_created" not in st.session_state:
    st.session_state.map_created = False

# ------------------ SHAPE FILE ------------------
def load_citibike_shape_file(data_dir, url, log=True):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "citibike_shape.zip"
    extract_path = data_dir / "citibike_shape"

    if not zip_path.exists():
        if log:
            print(f"Downloading shape file from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)

    if not any(extract_path.glob("*.shp")):
        if log:
            print(f"Extracting shape file to {extract_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    shapefile = list(extract_path.glob("*.shp"))
    if not shapefile:
        raise FileNotFoundError("No .shp file found in extracted zip.")
    return gpd.read_file(shapefile[0]).to_crs("epsg:4326")

# ------------------ MAPPING ------------------
def create_citibike_map(shapefile_gdf, prediction_data):
    gdf = shapefile_gdf.copy()
    gdf = gdf.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="stationid",   # adjust based on JC shape file column
        right_on="pickup_location_id",
        how="left"
    )
    gdf["predicted_demand"] = gdf["predicted_demand"].fillna(0)
    gdf = gdf.to_crs(epsg=4326)

    m = folium.Map(location=[40.7178, -74.0431], zoom_start=13, tiles="cartodbpositron")

    colormap = LinearColormap(
        colors=["#FFEDA0", "#FEB24C", "#FC4E2A", "#BD0026"],
        vmin=gdf["predicted_demand"].min(),
        vmax=gdf["predicted_demand"].max(),
    )
    colormap.add_to(m)

    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(predicted_demand)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        gdf.to_json(),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["stationid", "predicted_demand"],
            aliases=["Station ID:", "Predicted Demand:"],
            localize=True
        )
    ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

# ------------------ STREAMLIT APP ------------------
current_date = pd.Timestamp.now(tz="America/New_York")
st.title("Jersey City Citi Bike Trip Prediction")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

progress_bar = st.sidebar.progress(0)
N_STEPS = 4

# Step 1: Download shapefile
with st.spinner("Downloading Citi Bike station shape file..."):
    shape_url = "https://data.jerseycitynj.gov/api/explore/v2.1/catalog/datasets/citi-bike-locations-phase-1-system-map-3/exports/shp?lang=en&timezone=America%2FNew_York"
    geo_df = load_citibike_shape_file(DATA_DIR, shape_url)
    st.sidebar.write("Shape file downloaded.")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Load features
with st.spinner("Loading batch of features from feature store..."):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Features loaded.")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Fetch predictions
with st.spinner("Fetching predictions from latest model..."):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Predictions ready.")
    progress_bar.progress(3 / N_STEPS)

# Step 4: Plot map
with st.spinner("Creating map..."):
    st.subheader("Predicted Citi Bike Trips by Station")
    map_obj = create_citibike_map(geo_df, predictions)
    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])
    progress_bar.progress(4 / N_STEPS)

# ------------------ Prediction Stats ------------------
st.subheader("Prediction Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Trips", f"{predictions['predicted_demand'].mean():.0f}")
col2.metric("Max Trips", f"{predictions['predicted_demand'].max():.0f}")
col3.metric("Min Trips", f"{predictions['predicted_demand'].min():.0f}")

# ------------------ Top 10 stations ------------------
st.subheader("Top 10 Stations by Predicted Demand")
top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
st.dataframe(top10[["pickup_location_id", "predicted_demand"]])

# ------------------ Dropdown + Plot ------------------
selected_id = st.selectbox("Select Station ID", predictions["pickup_location_id"].unique())

fig = plot_prediction(
    features=features[features["pickup_location_id"] == selected_id],
    prediction=predictions[predictions["pickup_location_id"] == selected_id],
)
st.plotly_chart(fig, theme="streamlit", use_container_width=True)
