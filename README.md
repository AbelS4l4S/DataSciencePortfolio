# Churn Prediction Project

## Overview
This project focuses on predicting customer churn for a bank using machine learning techniques. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/syviaw/bankchurners) and contains various customer demographic, behavioral, and financial attributes. The target variable is `Attrition_Flag`, which indicates whether a customer is active or has churned.

## Project Objectives
- Build a machine learning model to predict customer churn.
- Gain insights into the factors influencing churn.
- Evaluate the performance of the model using appropriate metrics.

## Dataset Description
The dataset contains the following features:

### Target Variable
- **Attrition_Flag:**
  - "Attrited Customer" (churned)
  - "Existing Customer" (active)

### Features
- **CLIENTNUM:** Unique identification number for each customer (excluded from modeling).
- **Customer_Age:** Age of the customer in years.
- **Gender:** Gender of the customer.
- **Dependent_count:** Number of dependents reliant on the customer.
- **Education_Level:** Educational background of the customer.
- **Marital_Status:** Marital status of the customer.
- **Income_Category:** Annual income category.
- **Card_Category:** Type of credit card held by the customer.
- **Months_on_book:** Length of the customer's relationship with the bank in months.
- **Total_Relationship_Count:** Number of products/services used by the customer.
- **Months_Inactive_12_mon:** Months of inactivity in the last year.
- **Contacts_Count_12_mon:** Customer's contact frequency with the bank over the past year.
- **Credit_Limit:** Customer's credit limit.
- **Total_Revolving_Bal:** Outstanding revolving credit balance.
- **Avg_Open_To_Buy:** Available credit for purchases.
- **Total_Amt_Chng_Q4_Q1:** Change in transaction amount between Q4 and Q1.
- **Total_Trans_Amt:** Total transaction amount within a given period.
- **Total_Trans_Ct:** Total transaction count within a given period.
- **Total_Ct_Chng_Q4_Q1:** Change in transaction count between Q4 and Q1.
- **Avg_Utilization_Ratio:** Average credit utilization.

### Exclusions
- Columns related to a Naive Bayes Classifier output (not relevant).
- **CLIENTNUM** (identifier only).

## Methodology

### 1. Data Preparation
- Removed irrelevant features.
- Applied one-hot encoding for categorical variables.
- Scaled numerical features using `StandardScaler`.

### 2. Model Development
- Used `RandomForestClassifier` for initial modeling.
- Split data into training and testing sets (80/20).
- Evaluated performance using:
  - **Confusion Matrix**
  - **Classification Report**
  - **ROC-AUC Score**

### 3. Prediction
- Used the trained model to predict churn for new customers.

## Tools and Libraries
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn (for EDA)

## Results
- Classification metrics showed the model achieved competitive accuracy and recall.
- The model identified key features influencing churn, such as `Total_Trans_Amt`, `Months_Inactive_12_mon`, and `Avg_Utilization_Ratio`.

## Usage
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to train the model and make predictions.

## Example Prediction
```python
new_data = pd.DataFrame({
    'Customer_Age': [45],
    'Gender': ['Female'],
    'Dependent_count': [2],
    'Education_Level': ['Graduate'],
    'Marital_Status': ['Married'],
    'Income_Category': ['$80K - $120K'],
    'Card_Category': ['Platinum'],
    'Months_on_book': [36],
    'Total_Relationship_Count': [4],
    'Months_Inactive_12_mon': [1],
    'Contacts_Count_12_mon': [2],
    'Credit_Limit': [12000],
    'Total_Revolving_Bal': [1500],
    'Avg_Open_To_Buy': [10500],
    'Total_Amt_Chng_Q4_Q1': [1.2],
    'Total_Trans_Amt': [3000],
    'Total_Trans_Ct': [42],
    'Total_Ct_Chng_Q4_Q1': [0.9],
    'Avg_Utilization_Ratio': [0.12]
})

churn_prediction = pipeline.predict(new_data)
churn_probability = pipeline.predict_proba(new_data)[:, 1]

print("Churn Prediction (0=No Churn, 1=Churn):", churn_prediction)
print("Churn Probability:", churn_probability)
```

## Future Improvements
- Implement more advanced models like XGBoost or LightGBM.
- Conduct hyperparameter tuning for better performance.
- Analyze and address class imbalance using techniques like SMOTE.
- Deploy the model using a web framework (e.g., Flask or Django).

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, reach out to:
**abelsl1999@gmail.com**

