# House Price Prediction and Exploratory Analysis

This notebook analyzes and models the **House Prices dataset**, applying data cleaning, feature engineering, and regression modeling to predict housing prices. It walks through a full machine learning workflow using exploratory analysis and model evaluation.

## Dataset
The dataset includes typical housing attributes such as:
- LotArea, OverallQual, YearBuilt, GrLivArea, TotalBsmtSF, GarageCars, GarageArea, and SalePrice.  
Missing values are imputed based on context (median or mode), and categorical variables are label or one-hot encoded.

## Workflow Overview
1. **Data Cleaning:**  
   - Handle missing values for numerical and categorical features.  
   - Remove irrelevant or highly correlated variables.  
2. **Exploratory Data Analysis (EDA):**  
   - Distribution plots for key features.  
   - Correlation heatmap highlighting relationships with `SalePrice`.  
3. **Feature Engineering:**  
   - Log transformation of skewed variables.  
   - Creation of combined features (e.g., TotalSF = GrLivArea + TotalBsmtSF).  
4. **Modeling:**  
   - Train-test split (80:20).  
   - Models compared: Linear Regression, Ridge, Lasso, Random Forest, and XGBoost.  
   - Hyperparameter tuning via cross-validation.  
5. **Evaluation Metrics:**  
   - Root Mean Squared Error (RMSE)  
   - R² Score  
   - Cross-validation score comparison.

## Key Findings
- **GrLivArea**, **OverallQual**, and **TotalBsmtSF** are top predictors of price.  
- Regularized regression (Lasso/Ridge) performs better than plain linear regression.  
- Final RMSE: ~0.13 (after log transformation).

## Tools and Libraries
- Python  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn, xgboost  
