# ✈️ Predict_Fare_of_Airline_Tickets Using Machine Learning
## Objective: 
This project focuses on building a machine learning model to predict airline ticket prices based on historical flight data. It includes comprehensive data preprocessing, feature engineering, model building, hyperparameter tuning, and evaluation.

## Dataset Used:
- <a href="https://github.com/AishwaryaSatpute29/Predict_Fare_of_Airline_Tickets/blob/main/Data_Train.xlsx"> Training Dataset</a>
- <a href="https://github.com/AishwaryaSatpute29/Predict_Fare_of_Airline_Tickets/blob/main/Test_set.xlsx"> Testing Dataset</a>

## 📊 Tech Stack
- Python

- Pandas, NumPy

- Matplotlib, Seaborn, Plotly

- Scikit-learn

- RandomForestRegressor, DecisionTreeRegressor

- Hyperparameter tuning with RandomizedSearchCV

## 🧹 Data Preprocessing & Feature Engineering

- Converted time features to numerical format (hour/minute).

- Derived new features: Journey_day, Journey_month, Duration_hour, etc.

- One-hot encoded and label encoded categorical variables.

- Outlier detection and treatment using IQR method.

- Removed irrelevant or redundant features (Route, Additional_Info, etc.).

## 🔍 Exploratory Data Analysis
- Analyzed price vs total stops.

- Found relationships between flight duration and price.

- Airline-wise pricing trends visualized using box plots.

## 🤖 Model Building
- Models Trained:

   - Random Forest Regressor

   - Decision Tree Regressor

  Split: `75% Training` and `25% Testing`

## 📈 Model Evaluation
Metrics:

  - R² Score (Coefficient of Determination)
   
  - MAE (Mean Absolute Error)
   
  - MSE (Mean Squared Error)
   
  - RMSE (Root Mean Squared Error)
   
  - MAPE (Mean Absolute Percentage Error)  


