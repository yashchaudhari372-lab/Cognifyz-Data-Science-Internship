# Cognifyz Technologies - Data Science Internship Project

[![GitHub Pages](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-blue?style=for-the-badge&logo=github)](https://yashchaudhari372-lab.github.io/Cognifyz-Data-Science-Internship/)
[![Streamlit App](https://img.shields.io/badge/Interactive_App-Streamlit-red?style=for-the-badge&logo=streamlit)](https://share.streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-brightgreen?style=for-the-badge&logo=python)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Random_Forest_R²_0.9532-orange?style=for-the-badge)](https://scikit-learn.org)

🌐 **Live Deployed Web Portal:** [https://yashchaudhari372-lab.github.io/Cognifyz-Data-Science-Internship/](https://yashchaudhari372-lab.github.io/Cognifyz-Data-Science-Internship/)


**Intern Name:** Yash Devidas Chaudhari  
**Role:** Data Science Intern  
**Company:** Cognifyz Technologies (Cognifyz IT Solutions Pvt. Ltd.)  
**Internship Duration:** 17/11/2026 to 17/12/2026  
**Reference ID:** CTI/A1/C439738  

---

## Executive Overview
This repository contains the complete implementation of all three levels (Level 1, Level 2, and Level 3) of the Data Science Internship program at **Cognifyz Technologies**. The project involves comprehensive Exploratory Data Analysis (EDA), geospatial intelligence, descriptive statistics, feature engineering, and predictive machine learning modeling on a comprehensive restaurant dataset of 9,551 establishments across multiple countries.

---

## Project Structure & Deliverables

```
Cognifyz_Data_Science_Tasks/
│
├── Dataset .csv                            # Primary dataset (9,551 rows, 21 columns)
├── Level_1_Tasks.ipynb                     # Executed Jupyter Notebook for Level 1
├── Level_1_Tasks.html                      # Standalone HTML report for Level 1
├── Level_2_Tasks.ipynb                     # Executed Jupyter Notebook for Level 2
├── Level_2_Tasks.html                      # Standalone HTML report for Level 2
├── Level_3_Tasks.ipynb                     # Executed Jupyter Notebook for Level 3
├── Level_3_Tasks.html                      # Standalone HTML report for Level 3
├── README.md                               # Project documentation & executive summary
│
└── visualizations/                         # High-resolution charts & interactive maps
    ├── target_distribution.png
    ├── top_cuisines_cities.png
    ├── global_geospatial_scatter.png
    ├── interactive_restaurant_map.html    # Interactive Folium map with clustered markers
    ├── city_rating_boxplot.png
    ├── table_booking_delivery.png
    ├── ratings_table_booking.png
    ├── delivery_by_price_range.png
    ├── price_range_distribution.png
    ├── price_range_avg_rating.png
    ├── rating_color_comparison.png
    ├── feature_engineering_correlation.png
    ├── model_performance_comparison.png
    ├── feature_importance_rf.png
    ├── most_popular_cuisines_votes.png
    ├── highest_rated_cuisines.png
    ├── comprehensive_rating_distribution.png
    ├── city_and_cuisine_rating_comparison.png
    └── multivariate_relationships.png
```

---

## Detailed Task Breakdown & Key Findings

### Level 1: Data Exploration, Descriptive Analysis & Geospatial Intelligence
* **Task 1: Data Exploration and Preprocessing**
  * **Dataset Dimensions:** 9,551 records across 21 features.
  * **Missing Values:** Only 9 values (0.09%) missing in `Cuisines`, effectively imputed with `'Unknown'` to preserve sample fidelity.
  * **Target Distribution & Imbalance:** `Aggregate rating` ranges from 0.0 to 4.9. A significant zero-inflation exists where **2,148 restaurants (22.49%)** are unrated (`Rating text: 'Not rated'`). When isolating active ratings, the distribution follows a near-normal curve with a mean of **3.44** (median: 3.40).
* **Task 2: Descriptive Analysis**
  * **Geographic Concentration:** Over **90.6% (8,652 restaurants)** belong to Country Code 1 (India), heavily clustered in the National Capital Region (NCR): New Delhi (5,473), Gurgaon (1,118), and Noida (1,080).
  * **Top Cuisines:** **North Indian** (3,960 restaurants) and **Chinese** (2,735 restaurants) dominate the culinary landscape.
* **Task 3: Geospatial Analysis**
  * Plotted geographic coordinates globally and created an interactive clustered map (`interactive_restaurant_map.html`).
  * While raw coordinates show negligible linear correlation ($r \approx 0.00$), one-way **ANOVA tests** confirm that city location has a statistically significant effect on restaurant ratings ($F = 9.87, p < 0.0001$).

---

### Level 2: Service Capabilities, Price Range Analysis & Feature Engineering
* **Task 1: Table Booking and Online Delivery**
  * **Adoption Rates:** Only **12.12%** of restaurants offer table booking; **25.66%** offer online food delivery.
  * **Rating Premium:** Restaurants offering table booking achieve a **statistically significant higher rating** (**3.44** vs. **2.56**, $p < 10^{-15}$ via two-sample t-test).
  * **Delivery by Price Range:** Mid-tier restaurants leverage delivery most aggressively (Price Range 2: **41.31%**, Price Range 3: **29.19%**), whereas luxury establishments (Price Range 4) remain dine-in focused (9.04%).
* **Task 2: Price Range Analysis**
  * **Mode Price Range:** **Price Range 1 (Budget)** is most prevalent, comprising **46.53%** of all venues.
  * **Monotonic Rating Trend:** Higher price ranges systematically correspond to higher average ratings: Range 1 (2.00) $\rightarrow$ Range 2 (2.94) $\rightarrow$ Range 3 (3.68) $\rightarrow$ Range 4 (3.82).
  * **Highest-Rated Color:** **'Dark Green'** represents the highest customer satisfaction tier with an average rating of **4.66** (min: 4.5, max: 4.9).
* **Task 3: Feature Engineering**
  * Extracted text characteristics: `Restaurant_Name_Length`, `Address_Length`, and `Cuisine_Count`.
  * Encoded categorical features into numeric flags: `Has_Table_booking_Enc` and `Has_Online_delivery_Enc`.
  * Correlation analysis proved that `Price range` ($r = 0.438$), `Votes` ($r = 0.314$), and `Has Table booking` ($r = 0.190$) are the strongest individual predictors of customer satisfaction.

---

### Level 3: Predictive Modeling, Customer Preferences & Advanced Visualizations
* **Task 1: Predictive Modeling**
  * Built and compared three regression algorithms to predict `Aggregate rating` (80% train, 20% test split):
    | Model | MSE | RMSE | MAE | $R^2$ Score |
    | :--- | :---: | :---: | :---: | :---: |
    | **Linear Regression** | 1.6686 | 1.2918 | 1.0781 | 0.2669 |
    | **Decision Tree Regressor** | 0.1172 | 0.3423 | 0.2240 | 0.9485 |
    | **Random Forest Regressor** | **0.1064** | **0.3262** | **0.2127** | **0.9532** |
  * **Feature Importance:** Random Forest attributed over **90%** of decision importance to customer engagement (`Votes`), followed by `Price range` and `Average Cost for two`.
* **Task 2: Customer Preference Analysis**
  * **Popularity Leader (Votes):** **North Indian** generated over 580,000+ total votes, followed by **Chinese** (340,000+) and **Fast Food** (178,000+).
  * **Highest-Rated Cuisines:** High-end and experiential cuisines consistently secure higher ratings: **Continental** (3.72), **Italian** (3.68), and **Cafe** (3.62).
* **Task 3: Advanced Data Visualization**
  * Constructed bivariate and multivariate plots, violin charts, and log-scale scatter plots illustrating that customer engagement and pricing tier jointly govern restaurant performance.

---

## How to Run & Verify

1. **Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `Level_1_Tasks.ipynb`, `Level_2_Tasks.ipynb`, or `Level_3_Tasks.ipynb` and select *Run All*.
2. **Web Browser (Instant View):**
   Double-click any of the exported HTML files (`Level_1_Tasks.html`, `Level_2_Tasks.html`, `Level_3_Tasks.html`) to inspect all formatted code and rendered visuals directly in Chrome, Edge, or Firefox.
3. **Interactive Map:**
   Open `visualizations/interactive_restaurant_map.html` in any browser to interact with geospatial restaurant clusters.

---
*Created as part of the Cognifyz Technologies Data Science Internship Program.*
