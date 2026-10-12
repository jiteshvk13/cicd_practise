import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection  import train_test_split
import os
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.processed_data_path = "artifacts/processed"

        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info("Read data Succesfully")
        except Exception as e:
            logger.error(f"Error While reading the File {e}")
            raise CustomException("Failed to Load the Data", e)

    def handle_outliers(self,column):
        try:
            logger.info("handling Outlier")
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3-Q1
            Lower_value = Q1 - 1.5*IQR
            Upper_value = Q3 + 1.5*IQR
            sepal_median  =self.df[column].median()        

            for i in self.df[column]:
                if i<Lower_value or i>Upper_value:
                    self.df[column] = self.df[column].replace(i, sepal_median)

            logger.info("Handled Outlier Succesfully")

        except Exception as e:
            logger.error(f"Error While Handling the Outlier {e}")
            raise CustomException("Failed to Handle  the Outlier", e)


    def split_data(self):
        try:
            X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
            Y = self.df["Species"]
            X_train,X_test,y_train,y_test = train_test_split(X,Y , test_size=0.2 , random_state=42)

            logger.info("Data Splitted Succesfully")
            joblib.dump(X_train, os.path.join(self.processed_data_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.processed_data_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.processed_data_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("File Saved Succesfully")
        except Exception as e:
                    logger.error(f"Error While Splitting the Data {e}")
                    raise CustomException("Failed to Split  the Data", e)

    def run(self):
         self.load_data()
         self.handle_outliers("SepalWidthCm")
         self.split_data()


if __name__ =="__main__":
     data_processor = DataProcessing("artifacts/raw/data.csv")
     data_processor.run()