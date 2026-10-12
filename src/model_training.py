import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection  import train_test_split
import os
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix,recall_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self):
        self.processed_data_path =  "artifacts/processed"
        self.model_path = "artifacts/models"
        os.makedirs(self.model_path, exist_ok=True)
        self.model = DecisionTreeClassifier(criterion="gini",max_depth=30,random_state=42)
        logger.info("Model Training Initialized")


    def load_data(self):
        try:
            X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            logger.info(" Data Loaded  Succesfully")
            return X_train, X_test, y_train, y_test

        
        except Exception as e:
            logger.error(f"Error While loading the Data{e}")
            raise CustomException("Error While loading the Data", e)

    def train_model(self, X_train, y_train):
        try:
            self.model.fit(X_train,y_train)
            joblib.dump(self.model, os.path.join(self.model_path, "model.pkl"))
            logger.info(" Model trained and  Loaded  Succesfully")
        
        except Exception as e:
            logger.error(f"Error While training the Model{e}")
            raise CustomException("Error While training the Model", e)

    def evaluate_model(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            accuracy= accuracy_score(y_test,y_pred)
            precision =  precision_score(y_test,y_pred, average="weighted")
            recall = recall_score(y_test,y_pred, average="weighted")
            f1 =  f1_score(y_test,y_pred, average="weighted")

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            logger.info(" Model Evaluated Succesfully")

            cm = confusion_matrix(y_test,y_pred)

            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot= True, cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")
            logger.info(" Display the COnfusion matrix Succesfully")
            confusion_matrix_path = os.path.join(self.model_path, 'confusion_matrix.png')
            plt.savefig(confusion_matrix_path)
            plt.close()


        except Exception as e:
            logger.error(f"Error While evaluating the Model{e}")
            raise CustomException("Error While evaluating the Model", e)


    def run(self):
         X_train, X_test, y_train, y_test = self.load_data()
         self.train_model(X_train, y_train)
         self.evaluate_model(X_test,y_test)


if __name__ =="__main__":
    trainer = ModelTraining()
    trainer.run()