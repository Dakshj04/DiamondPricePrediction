import pandas as pd
import sys,os
from src.exception import CustomException
from src.logger import logging
import sys,os
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
         pass 

 
    def predict(self, features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
            # Add your processing code here
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)
            

class CustomData:
    def __init__(self, carat: float, cut: str, color: str, clarity: str, depth: float, table: float, x: float, y: float, z: float):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z]
            }
            logging.info("Dataframe gatehred")
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.info("Exception occurred in creating dataframe from custom data")
            raise CustomException(e, sys)            

   
