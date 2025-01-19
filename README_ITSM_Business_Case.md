
# ITSM Business Case: Machine Learning for ABC Tech

## Project Overview
This project explores the use of Machine Learning (ML) techniques to enhance the Incident Management process at ABC Tech. With over 22-25k annual IT incidents, ABC Tech aims to leverage ML to improve processes such as ticket prioritization, incident volume forecasting, ticket tagging, and predicting requests for change (RFC).

## Key Objectives

1. **Predict High-Priority Tickets**: Identify priority 1 & 2 tickets to enable preventive measures.
2. **Forecast Incident Volume**: Provide quarterly and annual forecasts for resource and technology planning.
3. **Auto-Tagging of Tickets**: Reduce reassignment delays by automatically tagging tickets with the correct priority and department.
4. **Predict RFCs and IT Asset Failures**: Identify potential failures and misconfigurations in IT assets.

## Data
- **Source**: MySQL database (read-only access).
- **Fields**: Includes CI category, subcategory, priority, impact, urgency, reassignments, related interactions, and more.
- **Volume**: 46,000 records from the years 2012-2014.

## Technologies and Libraries

### Programming Language:
- Python

### Libraries:
- Data Handling: `pandas`, `numpy`
- Database Connectivity: `mysql-connector`
- Visualization: `matplotlib`, `seaborn`
- Machine Learning: `scikit-learn`, `xgboost`, `imblearn`
- Time Series Analysis: `statsmodels`

### Installation
Install the required dependencies using pip:

```bash
pip install pandas numpy mysql-connector matplotlib seaborn scikit-learn xgboost statsmodels imbalanced-learn
```

## Workflow

### Data Preprocessing
1. Loaded data from MySQL.
2. Handled missing values and transformed categorical features.
3. Applied encoding techniques and feature scaling.

### Sample Code Snippet: Data Preprocessing
```python
# Import required libraries
import pandas as pd
import mysql.connector

# Connect to MySQL database
connection = mysql.connector.connect(
    host='18.136.157.135',
    user='dm_team',
    password='DM!$Team@27920!',
    database='project_itsm'
)

# Load data
query = "SELECT * FROM dataset_list"
data = pd.read_sql_query(query, connection)

# Display dataset info
print(data.info())
```

### Machine Learning Models
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Decision Tree**
- **Random Forest**
- **Artificial Neural Networks (ANN)**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**
- **Gradient Boosting**

### Time Series Models
- **ARIMA**: For incident volume forecasting.
- **SARIMAX**: Seasonal ARIMA with exogenous variables for improved forecasting.

### Visualizing Data
#### Incident Volume Over Time
```python
import matplotlib.pyplot as plt

data['open_time'] = pd.to_datetime(data['open_time'])
data.set_index('open_time', inplace=True)
incident_volume = data['No_of_Incident'].resample('M').sum()

plt.figure(figsize=(10, 6))
plt.plot(incident_volume, marker='o', linestyle='-')
plt.title('Monthly Incident Volume')
plt.xlabel('Month')
plt.ylabel('Incident Count')
plt.grid(True)
plt.show()
```
![Incident Volume Over Time](./images/incident_volume.png)

### Model Evaluation
Evaluated models using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.

### Results
- **Best Model for Ticket Prioritization**: ANN (95.24% accuracy, minimal false negatives).
- **Best Model for RFC Prediction**: Gradient Boosting (94.74% accuracy).
- **Incident Volume Forecasting**: SARIMAX yielded the best results for time series forecasting.

## Usage
1. Connect to the database using the provided credentials.
2. Run the script to preprocess data and train models.
3. Use the model outputs for predictions and insights.

## How to Run

1. Set up a Python environment.
2. Connect to the MySQL database:
    - Host: `18.136.157.135`
    - Port: `3306`
    - Username: `dm_team`
    - Password: `DM!$Team@27920!`
3. Execute the script:

```bash
python ITSM_bussinesscase.py
```

4. Review model outputs and visualizations for insights.

## Results and Insights
- High-priority tickets are effectively identified, reducing customer dissatisfaction.
- Quarterly and annual incident forecasts help in proactive resource allocation.
- Automated ticket tagging minimizes delays caused by manual errors.

## Future Enhancements
- Include real-time data integration.
- Expand use cases to include predictive maintenance.
- Explore advanced deep learning models for further improvements.

## Author
**Team ID**: PTID-CDS-JUL-23-1658

**Client**: ABC Tech

**Category**: ITSM - ML
