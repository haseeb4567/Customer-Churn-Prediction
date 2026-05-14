# Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn for a bank using Machine Learning. The main objective of the project is to identify customers who are likely to leave the bank based on customer information such as age, balance, geography, credit score, and account activity.

By analyzing customer behavior and training a classification model, the project helps businesses improve customer retention strategies and reduce customer loss.

---

# Dataset Information

The dataset used in this project contains customer-related information including:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary
* Exited (Target Variable)

### Target Variable

* **Exited = 1** → Customer left the bank
* **Exited = 0** → Customer stayed with the bank

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# Project Workflow

## 1. Importing Libraries

The required Python libraries for data analysis, visualization, preprocessing, and machine learning were imported.

## 2. Loading Dataset

The customer churn dataset was loaded using Pandas.

```python
pd.read_csv('Churn_Modelling.csv')
```

## 3. Exploratory Data Analysis (EDA)

Performed initial data exploration to understand the dataset.

### EDA Tasks Performed

* Checked dataset shape
* Viewed column names
* Inspected data types
* Generated statistical summary
* Checked missing values

---

# Data Preprocessing

Several preprocessing steps were performed to prepare the data for machine learning.

## Removed Unnecessary Columns

The following columns were removed because they do not significantly contribute to churn prediction:

* RowNumber
* CustomerId
* Surname

```python
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
```

## Encoding Categorical Variables

Categorical columns such as Geography and Gender were converted into numerical format using One-Hot Encoding.

```python
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
```

## Feature and Target Split

The dataset was divided into:

* Features (X)
* Target Variable (y)

```python
X = df.drop('Exited', axis=1)
y = df['Exited']
```

## Train-Test Split

The dataset was split into training and testing sets.

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

## Feature Scaling

Standardization was applied using StandardScaler.

```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

# Machine Learning Model

A Logistic Regression model was used for customer churn prediction.

```python
model = LogisticRegression(max_iter=100)
```

## Model Training

The model was trained using the training dataset.

```python
model.fit(X_train, y_train)
```

## Predictions

Predictions were made on the testing dataset.

```python
y_pred = model.predict(X_test)
```

---

# Model Evaluation

The model performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

```python
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
```

---

# Data Visualization

Different visualizations were created to better understand customer churn patterns.

## Visualizations Included

* Histogram plots
* Customer churn countplot
* Correlation heatmap

```python
df.hist(figsize=(10,10))
sns.countplot(x='Exited', data=df)
sns.heatmap(df.corr(), annot=True)
```

---

# Project Outcome

This project successfully predicts whether a customer is likely to leave the bank using machine learning techniques. The model helps in understanding customer behavior and supports businesses in improving customer retention strategies.

---

# Future Improvements

* Improve model accuracy using advanced algorithms
* Apply Hyperparameter Tuning
* Use Random Forest or XGBoost models
* Deploy the model using Flask or Streamlit
* Build an interactive dashboard

---

# How to Run the Project

## Step 1: Clone Repository

```bash
git clone <repository-link>
```

## Step 2: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Step 3: Run Jupyter Notebook

```bash
jupyter notebook
```

---

# Author

Haseeb Ahmed
