# Titanic-Classification
The Titanic Classification project involves building a predictive model to determine the likelihood of survival for passengers on the Titanic. This project uses historical data from the tragic sinking of the RMS Titanic in 1912. The goal is to predict which passengers survived based on various features such as age, gender, ticket class, and more.

Dataset
The dataset is publicly available and contains information about the passengers, including:

PassengerId
Survived (target variable: 0 = No, 1 = Yes)
Pclass (Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd)
Name
Sex
Age
SibSp (Number of siblings/spouses aboard)
Parch (Number of parents/children aboard)
Ticket
Fare
Cabin
Embarked (Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton)
Methodology
Data Preprocessing:

Handle missing values, particularly for Age and Embarked.
Convert categorical variables (e.g., Sex and Embarked) into numerical values using one-hot encoding.
Drop irrelevant features (e.g., Name, Ticket, Cabin) that do not contribute to the prediction.
Feature and Target Variables:

Separate the features (X) from the target variable (y), which is the 'Survived' column.
Model Training:

Split the dataset into training and testing sets to evaluate model performance.
Train a logistic regression model, a common choice for binary classification tasks.
Prediction and Evaluation:

Use the trained model to predict survival on the test set.
Evaluate the model's performance using accuracy score.
Code Implementation
The implementation involves:

Importing necessary libraries.
Loading and preprocessing the dataset.
Splitting the data into training and testing sets.
Training a logistic regression model.
Predicting and evaluating the model's performance.
