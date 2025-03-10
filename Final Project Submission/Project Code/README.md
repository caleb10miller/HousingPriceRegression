# Predicting Housing Prices using Machine Learning Models

## Authors
**Caleb Miller, Hashim Afzal**  
College of Computing & Informatics, Drexel University   

## Project Overview
This project leverages **machine learning models in PySpark** to predict housing prices in **King County, Washington**. The dataset includes **21,614 property sales records**, and models were trained to analyze **key factors influencing housing prices**.  

Through **Exploratory Data Analysis (EDA), feature engineering, and model evaluation**, we identified the best-performing model for price prediction, which was **XGBoost Regression (R² = 0.8964, RMSE = $135,699.62)**. The project highlights PySpark’s efficiency in handling large datasets and its potential application in real estate pricing.  

## Key Features
- **Data Preprocessing**: Outlier removal, handling skewed distributions, feature selection using **VIF analysis**.
- **Exploratory Data Analysis (EDA)**: Visualizing distributions, correlation analysis, and identifying influential property features.
- **Feature Engineering**: Created new features such as **house age and years since renovation** to improve model accuracy.
- **Machine Learning Models**:
  - **XGBoost Regression** (Best-performing model)
  - **Multi-Layer Perceptron (MLP) Regression**
  - **Linear Regression**
  - **Decision Tree Regression**
- **Model Evaluation**: Compared models using **R² scores and RMSE values**.

## Dataset
We used the **"House Sales in King County" dataset** from Kaggle, which includes:
- **Property Characteristics**: Bedrooms, bathrooms, square footage, lot size, floors, waterfront view, condition, and construction grade.
- **Geospatial Data**: Zip code, latitude, longitude.
- **Transaction Details**: Sale price and date.

Dataset Link: [King County House Sales Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

## Results
| **Model**               | **R² Score** | **RMSE ($)** |
|-------------------------|------------|--------------|
| **XGBoost**            | 0.8964     | 135,699.62   |
| **MLP Regression**     | 0.8247     | 191,888.17   |
| **Linear Regression**  | 0.8148     | 193,438.71   |
| **Decision Tree**      | 0.7732     | 140,705.00   |

### **Key Findings**
- **XGBoost outperformed all models**, demonstrating strong predictive power and ability to handle nonlinear relationships.
- **MLP Regression captured complex interactions** but was computationally expensive and required TensorFlow integration.
- **Linear Regression served as a strong baseline model**, generalizing well but struggling with extreme values.
- **Decision Tree Regression suffered from overfitting**, making it unsuitable for price prediction.

## Future Work
To improve model robustness and expand usability, we propose:
- **Expanding the dataset**: Incorporate **nationwide real estate data** for broader applicability.
- **Adding more features**: Include **crime rates, mortgage interest rates, and climate data**.
- **Enhancing computational efficiency**: Utilize **distributed computing techniques** to scale predictions across larger datasets.

## License
This project is licensed under the MIT License.

### Contact Information
For questions or collaborations, feel free to reach out:
Email: cm3962@drexel.edu & ha695@drexel.edu
