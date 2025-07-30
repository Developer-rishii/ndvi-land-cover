# Summer Analytics First Hackathon

Overview

Welcome to First course hackathon of Summer Analytics 2025.
Hosted by Consulting & Analytics Club and GeeksforGeeks (GFG)
Classify land cover types using NDVI time-series data from satellite imagery and OpenStreetMap (OSM) labels. Your challenge is to build a Logistic Regression model that accurately predicts land cover classes despite noisy NDVI signals. Top performers win GFG Premium memberships, and all participants get exclusive discounts!

Start
Jun 6, 2025

Close
Jun 13, 2025

Description

Hackathon Problem Statement: NDVI-based Land Cover Classification
Key Concepts

NDVI (Normalized Difference Vegetation Index)
Measures vegetation health using satellite data:
Where:-

NIR = Near-Infrared reflectance
Red = Red reflectance
2. Data Challenges
Noise: The main challenge with the dataset is that both the imagery and the crowdsourced data contain noise (due to cloud cover in the images and inaccurate labeling/digitizing of polygons).

Missing Data: Certain NDVI values are missing because of cloud cover obstructing the satellite view.

Temporal Variations: NDVI values vary seasonally, requiring careful feature engineering to extract meaningful trends.

Important Note:

The training and public leaderboard test data may contain noisy observations, while the private leaderboard data is clean and free of noise. This design helps evaluate how well your model generalizes beyond noisy training conditions.

Dataset

Each row in the dataset contains:

class: Ground truth label of the land cover type — one of {Water, Impervious, Farm, Forest, Grass, Orchard}

ID:Unique identifier for the sample

27 NDVI Time Points: Columns labeled in the format YYYYMMDD_N (e.g., 20150720_N, 20150602_N) represent NDVI values collected on different dates. These values form a time series representing vegetation dynamics for each location.

