import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(
    page_title="Cognifyz Data Science Internship | Yash Chaudhari",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset .csv')
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    return df

# Helper to load model
@st.cache_resource
def load_model():
    if os.path.exists('rf_model.joblib'):
        return joblib.load('rf_model.joblib')
    return None

df = load_data()
model = load_model()

# Sidebar Navigation
st.sidebar.image("https://raw.githubusercontent.com/yashchaudhari372-lab/Cognifyz-Data-Science-Internship/main/visualizations/target_distribution.png" if not os.path.exists('visualizations/target_distribution.png') else 'visualizations/target_distribution.png', use_container_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Internship Section:",
    [
        "📌 Executive Overview",
        "🗺️ Level 1: EDA & Geospatial Analysis",
        "📊 Level 2: Business & Pricing Analysis",
        "🤖 Level 3: Machine Learning & Live Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Candidate:** Yash Devidas Chaudhari  
**Role:** Data Science Intern  
**Company:** Cognifyz Technologies  
**ID:** CTI/A1/C439738  
""")

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "📌 Executive Overview":
    st.markdown('<div class="main-header">Cognifyz Technologies - Data Science Internship</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive Analytics, Geospatial Intelligence & Predictive Modeling by Yash Devidas Chaudhari</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Restaurants", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{df.shape[1]}")
    with col3:
        st.metric("Countries Represented", f"{df['Country Code'].nunique()}")
    with col4:
        st.metric("Best Model R² Score", "0.9532 (RF)")
        
    st.markdown("---")
    
    st.markdown("""
    ### Project Objectives
    This application deploys and presents the complete solutions developed for the **Cognifyz Technologies Data Science Internship Program**.
    The project encompasses end-to-end analytical problem solving across 3 comprehensive levels:
    
    1. **Level 1: Exploration, Descriptive Statistics & Geospatial Intelligence**
       * Handled data cleansing, resolved missing values in cuisines, and identified the zero-inflation rating anomaly.
       * Visualized global restaurant coordinates using an interactive clustered map.
       * Applied one-way ANOVA hypothesis testing to establish the significance of city location on ratings ($p < 0.0001$).
       
    2. **Level 2: Service Capabilities, Pricing & Feature Engineering**
       * Assessed table booking and online delivery availability and proved that table-booking restaurants command significantly higher ratings ($3.44$ vs $2.56$, $p < 10^{-15}$).
       * Analyzed the monotonic relationship between price ranges and ratings.
       * Engineered text length features and encoded service indicators.
       
    3. **Level 3: Predictive Modeling & Customer Preferences**
       * Benchmarked Linear Regression, Decision Tree, and Random Forest algorithms.
       * Identified customer engagement (`Votes`) as the dominant predictor of restaurant ratings.
       * Deployed an interactive real-time rating prediction engine powered by our trained ensemble model.
    """)
    
    st.info("💡 **Tip:** Use the left sidebar to navigate through each Level's interactive analyses and test the live Machine Learning predictor!")

# ==========================================
# PAGE 2: LEVEL 1 TASKS
# ==========================================
elif page == "🗺️ Level 1: EDA & Geospatial Analysis":
    st.markdown('<div class="main-header">Level 1: Data Exploration, Descriptive Analysis & Geospatial Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["1. Data Exploration & Target Distribution", "2. Descriptive & Categorical Analysis", "3. Geospatial Mapping"])
    
    with tab1:
        st.subheader("Target Variable (`Aggregate rating`) Distribution")
        st.write("Exploration revealed that **2,148 restaurants (22.49%)** are unrated with an aggregate rating of `0.0`. When filtering for actively rated restaurants, ratings follow a near-normal distribution with a mean of **3.44**.")
        if os.path.exists('visualizations/target_distribution.png'):
            st.image('visualizations/target_distribution.png', use_container_width=True)
            
        with st.expander("🔍 View Raw Data Sample"):
            st.dataframe(df.head(10))
            
    with tab2:
        st.subheader("Top Cuisines & High-Density Cities")
        st.write("Over **90.6%** of restaurants belong to India, with the National Capital Region (NCR) representing the dense majority. **North Indian** and **Chinese** are the most widespread cuisines.")
        if os.path.exists('visualizations/top_cuisines_cities.png'):
            st.image('visualizations/top_cuisines_cities.png', use_container_width=True)
            
    with tab3:
        st.subheader("Interactive Geospatial Map (Folium)")
        st.write("Explore restaurant clusters across cities with interactive pinpoints, ratings, and cuisine details.")
        map_path = 'visualizations/interactive_restaurant_map.html'
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            components.html(html_data, height=550)
        else:
            st.warning("Map file not found.")
            
        if os.path.exists('visualizations/city_rating_boxplot.png'):
            st.image('visualizations/city_rating_boxplot.png', use_container_width=True)

# ==========================================
# PAGE 3: LEVEL 2 TASKS
# ==========================================
elif page == "📊 Level 2: Business & Pricing Analysis":
    st.markdown('<div class="main-header">Level 2: Service Capabilities, Price Range & Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["1. Table Booking & Online Delivery", "2. Price Range Analysis", "3. Feature Engineering & Correlation"])
    
    with tab1:
        st.subheader("Service Availability vs. Restaurant Ratings")
        st.write("While only **12.12%** of restaurants offer table booking, those that do score a **3.44** average rating compared to **2.56** for those without ($p < 10^{-15}$).")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('visualizations/table_booking_delivery.png'):
                st.image('visualizations/table_booking_delivery.png', use_container_width=True)
        with col2:
            if os.path.exists('visualizations/ratings_table_booking.png'):
                st.image('visualizations/ratings_table_booking.png', use_container_width=True)
                
    with tab2:
        st.subheader("Price Tier Dynamics & Rating Colors")
        st.write("Price Range 1 is the most common (46.5%), but average ratings scale upward directly with price tier. **Dark Green** represents the top rating tier with a 4.66 average.")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('visualizations/delivery_by_price_range.png'):
                st.image('visualizations/delivery_by_price_range.png', use_container_width=True)
        with col2:
            if os.path.exists('visualizations/rating_color_comparison.png'):
                st.image('visualizations/rating_color_comparison.png', use_container_width=True)
                
    with tab3:
        st.subheader("Engineered Features Correlation Heatmap")
        st.write("Extracted text characteristics (`Restaurant_Name_Length`, `Address_Length`, `Cuisine_Count`) and encoded binary service indicators.")
        if os.path.exists('visualizations/feature_engineering_correlation.png'):
            st.image('visualizations/feature_engineering_correlation.png', use_container_width=True)

# ==========================================
# PAGE 4: LEVEL 3 TASKS & LIVE PREDICTOR
# ==========================================
elif page == "🤖 Level 3: Machine Learning & Live Predictor":
    st.markdown('<div class="main-header">Level 3: Predictive Modeling & Live Restaurant Rating Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Live Rating Predictor", "📈 Algorithm Benchmarks", "🍽️ Customer Preferences"])
    
    with tab1:
        st.subheader("Interactive Restaurant Rating Predictor (Random Forest)")
        st.write("Test our trained machine learning model by configuring restaurant parameters below:")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_name = st.text_input("Restaurant Name", value="Royal Gourmet Bistro")
                input_address = st.text_input("Restaurant Address", value="123 Connaught Place, New Delhi")
                input_votes = st.slider("Total Customer Votes", min_value=0, max_value=5000, value=350, step=10)
                input_price_range = st.select_slider("Price Range", options=[1, 2, 3, 4], value=3)
            with col2:
                input_cuisines_count = st.slider("Number of Cuisines Offered", min_value=1, max_value=8, value=3)
                input_cost_for_two = st.number_input("Average Cost for Two (in local currency)", min_value=50, max_value=10000, value=1200, step=50)
                input_table_booking = st.selectbox("Offers Table Booking?", options=["Yes", "No"], index=0)
                input_online_delivery = st.selectbox("Offers Online Delivery?", options=["Yes", "No"], index=0)
                
            submit_btn = st.form_submit_button("Predict Aggregate Rating ⭐", use_container_width=True)
            
        if submit_btn and model is not None:
            # Prepare feature vector
            name_len = len(input_name)
            addr_len = len(input_address)
            tb_enc = 1 if input_table_booking == "Yes" else 0
            od_enc = 1 if input_online_delivery == "Yes" else 0
            
            feat_df = pd.DataFrame([{
                'Price range': input_price_range,
                'Votes': input_votes,
                'Average Cost for two': input_cost_for_two,
                'Has_Table_booking_Enc': tb_enc,
                'Has_Online_delivery_Enc': od_enc,
                'Restaurant_Name_Length': name_len,
                'Address_Length': addr_len,
                'Cuisine_Count': input_cuisines_count
            }])
            
            prediction = float(model.predict(feat_df)[0])
            prediction = min(max(prediction, 0.0), 5.0)
            
            # Determine color tier
            if prediction >= 4.5:
                color_name = "Dark Green"
                color_hex = "#006400"
                rating_text = "Excellent"
            elif prediction >= 4.0:
                color_name = "Green"
                color_hex = "#2ecc71"
                rating_text = "Very Good"
            elif prediction >= 3.5:
                color_name = "Yellow"
                color_hex = "#f1c40f"
                rating_text = "Good"
            elif prediction >= 2.5:
                color_name = "Orange"
                color_hex = "#e67e22"
                rating_text = "Average"
            elif prediction > 0.0:
                color_name = "Red"
                color_hex = "#e74c3c"
                rating_text = "Poor"
            else:
                color_name = "White"
                color_hex = "#bdc3c7"
                rating_text = "Not Rated"
                
            st.success(f"### Predicted Aggregate Rating: **{prediction:.2f} / 5.0** ⭐")
            st.markdown(f"""
            * **Rating Status:** **{rating_text}**  
            * **Assigned Color Tier:** <span style="background-color: {color_hex}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;">{color_name}</span>
            """, unsafe_allow_html=True)
            
    with tab2:
        st.subheader("Model Evaluation Benchmarks")
        benchmark_data = pd.DataFrame({
            'Algorithm': ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor'],
            'MSE': [1.6686, 0.1172, 0.1064],
            'RMSE': [1.2918, 0.3423, 0.3262],
            'MAE': [1.0781, 0.2240, 0.2127],
            'R² Score': [0.2669, 0.9485, 0.9532]
        })
        st.dataframe(benchmark_data, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('visualizations/model_performance_comparison.png'):
                st.image('visualizations/model_performance_comparison.png', use_container_width=True)
        with col2:
            if os.path.exists('visualizations/feature_importance_rf.png'):
                st.image('visualizations/feature_importance_rf.png', use_container_width=True)
                
    with tab3:
        st.subheader("Customer Preferences: Votes vs. Highest Ratings")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('visualizations/most_popular_cuisines_votes.png'):
                st.image('visualizations/most_popular_cuisines_votes.png', use_container_width=True)
        with col2:
            if os.path.exists('visualizations/highest_rated_cuisines.png'):
                st.image('visualizations/highest_rated_cuisines.png', use_container_width=True)
                
        if os.path.exists('visualizations/multivariate_relationships.png'):
            st.image('visualizations/multivariate_relationships.png', use_container_width=True)
