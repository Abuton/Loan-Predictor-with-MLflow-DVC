import mlflow
import pandas as pd

logged_model = 'runs:/b48b0a2cefda472c90b1ab0582ca7a52/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

data_path = "data/loan_data.csv"

data = pd.read_csv(data_path)

data['Gender']= data['Gender'].map({'Male':0, 'Female':1})
data['Married']= data['Married'].map({'No':0, 'Yes':1})
data['Loan_Status']= data['Loan_Status'].map({'N':0, 'Y':1})

print('Dropping Missing Values\n')
data = data.dropna()


# Predict on a Pandas DataFrame.
test_inference = loaded_model.predict(data[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']])

print(test_inference)