from __future__ import print_function
import os
import sys
import platform

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, enet_path

import mlflow
import mlflow.sklearn
import plot_utils

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

class Trainer(object):
    def __init__(self, experiment_name, data_path, run_origin="none"):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.run_origin = run_origin
        np.random.seed(2021)

        print("experiment_name:",self.experiment_name)
        print("run_origin:",run_origin)

        # Read the wine-quality csv file 
        print("data_path:",data_path)
        data = pd.read_csv(data_path)

    
        data['Gender']= data['Gender'].map({'Male':0, 'Female':1})
        data['Married']= data['Married'].map({'No':0, 'Yes':1})
        data['Loan_Status']= data['Loan_Status'].map({'N':0, 'Y':1})

        print('Dropping Missing Values\n')
        data = data.dropna()

        X = data[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
        y = data.Loan_Status
        
        # Split the data into training and test sets. (0.75, 0.25) split.
        self.train_x , self.test_x, self.train_y, self.test_y = train_test_split(X, y)
    
        # The predicted column is "Loan_Status" which is a binary either [0 or 1]
        self.current_file = os.path.basename(__file__)

        self.X = X.values
        self.y = y.values.ravel()

        # If using 'mlflow run' must use --experiment-id to set experiment since set_experiment() does not work
        if self.experiment_name != "none":
            mlflow.set_experiment(experiment_name)
            client = mlflow.tracking.MlflowClient()
            experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
            print("experiment_id:",experiment_id)

    def eval_metrics(self, actual, pred):
        acc_score = accuracy_score(actual, pred)
        recal_sc = recall_score(actual, pred)
        r2 = r2_score(actual, pred)
        return acc_score, recal_sc, r2
    
    def train(self, C, l1_ratio, max_iter):
        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            print("run_id:",run_id)
            experiment_id = run.info.experiment_id
            print("  experiment_id:",experiment_id)
            clf = LogisticRegression(C=C, l1_ratio=l1_ratio, max_iter=max_iter, random_state=42)
            clf.fit(self.train_x, self.train_y)
    
            predicted_qualities = clf.predict(self.test_x)
            (acc_score, recal_sc, r2) = self.eval_metrics(self.test_y, predicted_qualities)
    
            #print("Parameters:(alpha={}, l1_ratio={}):".format(alpha, l1_ratio))
            print("  Parameters:")
            print("    C:",C)
            print("    l1_ratio:",l1_ratio)
            print("    max_iter:",max_iter)

            print("  Metrics:")
            print("    Accuracy_Score:",acc_score)
            print("    Recall_Score:",recal_sc)
            print("    R2:",r2)
    
            mlflow.log_param("C", C)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_param("max_iter", max_iter)

    
            mlflow.log_metric("accuracy_score", acc_score)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("recall_score", recal_sc)
            
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("exp_id", experiment_id)
            mlflow.set_tag("exp_name", self.experiment_name)
            mlflow.set_tag("run_origin", self.run_origin)
            mlflow.set_tag("platform", platform.system())
    
            mlflow.sklearn.log_model(clf, "model")
    
            eps = 5e-3  # the smaller it is the longer is the path
            alphas_enet, coefs_enet, _ = enet_path(self.X, self.y, eps=eps, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=False)
            plot_file = "LogisticRegression-paths.png"
            plot_utils.plot_enet_descent_path(self.X, self.y, l1_ratio, alphas_enet, coefs_enet, plot_file)
            mlflow.log_artifact(plot_file)
    
        return (experiment_id,run_id)

if __name__ == "__main__":
    train = Trainer(experiment_name='loan-predictor', data_path='data/data.csv', run_origin='Local Run')

    train.train(C=0.5, l1_ratio=0.8, max_iter=300)
