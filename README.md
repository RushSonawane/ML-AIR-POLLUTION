Air Pollution Death Rate Prediction Using Machine Learning
Project Overview
This project focuses on analyzing and predicting air pollution-related death rates using Machine Learning regression algorithms.
The dataset contains historical air pollution data for different countries from 1990 to 2017. It includes total air pollution-related deaths and individual pollution categories such as indoor air pollution, outdoor particulate matter, and outdoor ozone pollution.
The main objective is to explore relationships between different air pollution indicators and build regression models to predict death rates per 100,000 population.
Dataset Information
Dataset Name: Death Rates from Air Pollution
Total Records: 6,468
Total Features: 7
Time Period: 1990–2017
Data Type: Structured Numerical and Categorical Data
Dataset Features
Feature	Description
Entity	Country name
Code	Country code
Year	Observation year
Air pollution (total)	Total deaths per 100,000
Indoor air pollution	Deaths per 100,000
Outdoor particulate matter	Deaths per 100,000
Outdoor ozone pollution	Deaths per 100,000
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Statsmodels
Jupyter Notebook
Machine Learning Algorithms Implemented
The project explores multiple regression algorithms:
1.Linear Regression
2.Random Forest Regression
3.Gradient Boosting Regression
4.Support Vector Regression (SVR)
5.Ridge Regression
6.Lasso Regression
7.Elastic Net Regression
Project Workflow
1.Data Collection: Load the air pollution death-rate dataset.
2.Data Exploration: Examine dataset dimensions, columns, and statistical information.
3.Data Preprocessing: Check missing values and encode categorical columns.
4.Feature Selection: Select relevant independent and dependent variables.
5.Data Splitting: Split the dataset into training and testing sets.
6.Model Training: Train different regression algorithms.
7.Model Evaluation: Evaluate model performance using Mean Squared Error (MSE) and R-squared.
8.Model Comparison: Compare regression algorithms to understand their predictive performance.
Evaluation Metrics
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
R-squared (R²): Measures how much variation in the target variable is explained by the model.
Project Structure
Air-Pollution-Death-Rate-Prediction/
│
├── dataset/
│   └── death_ratesfrom_airpollution.csv.xls
│
├── notebooks/
│   └── Air_Pollution_Analysis.ipynb
│
├── README.md
│
└── requirements.txt
Key Learnings
Understanding and exploring real-world environmental datasets.
Performing data preprocessing and feature selection.
Implementing different regression algorithms.
Understanding model evaluation metrics.
Comparing traditional Machine Learning regression techniques.
Working with Python libraries for data analysis and predictive modelling.
Future Improvements
Perform detailed Exploratory Data Analysis (EDA).
Develop country-wise and year-wise prediction models.
Apply time-series forecasting to predict future trends.
Improve model performance through hyperparameter tuning.
Create interactive visualizations and dashboards.
Develop a user-friendly interface for making predictions.
Important Note
The dataset contains related pollution indicators. Some variables may have mathematical relationships with the total or component death rates. Therefore, feature selection and data leakage checks are necessary before interpreting model performance.
This project is intended for educational and analytical purposes. Predictions should not be interpreted as medical or public-health conclusions.
Author
Rushika Sonawane
Machine Learning | Python | Data Analysis
GitHub: https://www.linkedin.com/in/rushika-sonawane-0294ba25a?utm_source=share_via&utm_content=profile&utm_medium=member_android
LinkedIn: https://www.linkedin.com/in/rushika-sonawane-0294ba25a

Exploring the relationship between air pollution and public health through data analysis and machine learning.
