
# import streamlit as st
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import folium
# from folium.plugins import HeatMap
# from streamlit_folium import folium_static  # Ensure this is imported

# # Load the trained XGBoost model
# model = xgb.XGBRegressor()
# model.load_model('model.json')  # Ensure you have the model saved as 'xgb_model.model'

# # Define possible values for categorical columns
# store_categories = ['Home & Kitchen', 'Sports', 'Beauty', 'Electrical', 'Daily Needs', 'Grocery']
# store_sizes = ['Small', 'Medium', 'Large']

# # Mapping for store size factor
# store_size_factor = {'Small': 1, 'Medium': 1.5, 'Large': 2}

# # Streamlit app
# st.title("Store Footfall Prediction and Heatmap")

# # Input fields
# st.sidebar.header("Input Store Details")
# store_category = st.sidebar.selectbox("Store Category", store_categories)
# store_size = st.sidebar.selectbox("Store Size", store_sizes)
# latitude = st.sidebar.number_input("Latitude", value=21.15)
# longitude = st.sidebar.number_input("Longitude", value=79.08)
# population_density = st.sidebar.number_input("Population Density", value=3000)
# average_income = st.sidebar.number_input("Average Income", value=50000)
# competitor_stores = st.sidebar.number_input("Competitor Stores", value=5)

# # Encode categorical variables
# label_encoder = LabelEncoder()
# store_category_encoded = label_encoder.fit_transform([store_category])[0]
# store_size_encoded = label_encoder.fit_transform([store_size])[0]

# # Normalize numerical features
# scaler = StandardScaler()
# numerical_features = np.array([[population_density, average_income, competitor_stores]])
# numerical_features_scaled = scaler.fit_transform(numerical_features)

# # Create input DataFrame
# input_data = pd.DataFrame({
#     "Store Category": [store_category_encoded],
#     "Store Size": [store_size_encoded],
#     "Latitude": [latitude],
#     "Longitude": [longitude],
#     "Population Density": [numerical_features_scaled[0][0]],
#     "Average Income": [numerical_features_scaled[0][1]],
#     "Competitor Stores": [numerical_features_scaled[0][2]],
# })

# # Predict footfall
# predicted_footfall = model.predict(input_data)[0]

# # Display predicted footfall
# st.write(f"Predicted Footfall: {predicted_footfall}")

# # Create a map with the predicted footfall
# m = folium.Map(location=[latitude, longitude], zoom_start=12)

# # Add heatmap
# # HeatMap([[latitude, longitude, predicted_footfall]], radius=15).add_to(m)
# HeatMap([[latitude, longitude, float(predicted_footfall)]], radius=15).add_to(m)


# # Display the map
# folium_static(m)  

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static  
import joblib

# Load the trained XGBoost model
model = xgb.XGBRegressor()

# model.load_model('model.json') 
model.load_model('model.json')

# Define possible values for categorical columns
store_categories = ['Home & Kitchen', 'Sports', 'Beauty', 'Electrical', 'Daily Needs', 'Grocery']
store_sizes = ['Small', 'Medium', 'Large']

# Streamlit app UI
st.title("Store Footfall Prediction and Heatmap")

# User Inputs
st.sidebar.header("Input Store Details")
store_category = st.sidebar.selectbox("Store Category", store_categories)
store_size = st.sidebar.selectbox("Store Size", store_sizes)
latitude = st.sidebar.number_input("Latitude", value=21.15)
longitude = st.sidebar.number_input("Longitude", value=79.08)
population_density = st.sidebar.number_input("Population Density", value=3000)
average_income = st.sidebar.number_input("Average Income", value=50000)
competitor_stores = st.sidebar.number_input("Competitor Stores", value=5)

# Encode categorical variables
label_encoder = LabelEncoder()
store_category_encoded = label_encoder.fit_transform([store_category])[0]
store_size_encoded = label_encoder.fit_transform([store_size])[0]

# Normalize numerical features
scaler = StandardScaler()
numerical_features = np.array([[population_density, average_income, competitor_stores]])
numerical_features_scaled = scaler.fit_transform(numerical_features)

# Prepare input DataFrame
input_data = pd.DataFrame({
    "Store Category": [store_category_encoded],
    "Store Size": [store_size_encoded],
    "Latitude": [latitude],
    "Longitude": [longitude],
    "Population Density": [numerical_features_scaled[0][0]],
    "Average Income": [numerical_features_scaled[0][1]],
    "Competitor Stores": [numerical_features_scaled[0][2]],
})

# Predict footfall
predicted_footfall = model.predict(input_data)[0]
# st.write(f"Predicted Footfall: {int(predicted_footfall)}")

# Create a Folium map centered at input location
m = folium.Map(location=[latitude, longitude], zoom_start=12)

# Generate hardcoded nearby heat zones
heatmap_data = []
for _ in range(150):  # Generate 150 points
    lat_offset = np.random.uniform(-0.02, 0.02)  # Random offset within ~2km
    lon_offset = np.random.uniform(-0.02, 0.02)
    intensity = float(predicted_footfall) * np.random.uniform(0.5, 1.5)  # Randomize intensity
    heatmap_data.append([latitude + lat_offset, longitude + lon_offset, intensity])

# Normalize footfall values to avoid extreme outliers
max_footfall = max([h[2] for h in heatmap_data])
heatmap_data = [[h[0], h[1], h[2] / max_footfall] for h in heatmap_data]  # Normalize intensity

# Add heatmap layer
HeatMap(heatmap_data, radius=20, blur=15, max_zoom=10).add_to(m)

# Display the map
folium_static(m)
