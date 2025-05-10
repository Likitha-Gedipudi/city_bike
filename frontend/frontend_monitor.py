import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)


import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

st.title("Mean Absolute Error (MAE) by Pickup Hour")

# Sidebar for user input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,  # Minimum allowed value
    max_value=24 * 28,  # (Optional) Maximum allowed value
    value=12,  # Initial/default value
    step=1,  # Step size for increment/decrement
)

# Fetch data
st.write("Fetching data for the past", past_hours, "hours...")
df1 = fetch_hourly_rides(past_hours)
df2 = fetch_predictions(past_hours)
print("mnbdsjhv")
print(df1)
print("sdwdsd")
print(df2)

# Merge the DataFrames on 'pickup_location_id' and 'pickup_hour'
merged_df = pd.merge(df1, df2, on=["pickup_location_id", "pickup_hour"])

print(merged_df)

# Calculate the absolute error
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# Group by 'pickup_hour' and calculate the mean absolute error (MAE)
mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Create a Plotly plot
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Display the plot
st.plotly_chart(fig)
st.write(f'Average MAE: {mae_by_hour["MAE"].mean()}')
