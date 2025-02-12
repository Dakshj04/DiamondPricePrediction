from sklearn.impute import SimpleImputer ## handle missing value
from sklearn.preprocessing import StandardScaler ## handle feature scaling
from sklearn.preprocessing import OrdinalEncoder ## ordinal encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
import numpy as np
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass
from src.utils import save_obj
@dataclass  
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def  get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated")
             # categorical and num col
            categorical_cols=['cut', 'color', 'clarity']
            numerical_cols=['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories=["Fair","Good","Very Good","Premium","Ideal"]
            color_categories=["D","E","F","G","H","I","J"]
            clarity_categories=["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

            logging.info("Data transformation pipeline initiated")

            num_pipeline=Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler())
                    ]
                )   
            cat_pipeline=Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy="most_frequent")),
                        ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                        ("scaler",StandardScaler())
                    ]
                )

            preprocessor=ColumnTransformer([
                    ("num_pipeline",num_pipeline,numerical_cols),
                    ("cat_pipeline",cat_pipeline,categorical_cols)
                ])
            
            logging.info("Data transformation completed")

            return preprocessor

        except Exception as e:
            logging.info("Exception Occured in data tranformation")
            raise CustomException(e,sys)
   
    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("Read train and test data completed ")
            logging.info(f"Train Dataframe head  : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe head  : \n{test_df.head().to_string()}")
            logging.info("Obtaining preprocessing object ")

            preprocessing_obj=self.get_data_transformation_object()

            target_column="price"
            drop_column=[target_column,"id"]

            ## dividing the dataset into independent and dependent dataset for train
            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)

            target_feature_train_df=train_df[target_column]

            ## dividing the dataset into independent and dependent dataset for test

            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)

            target_feature_test_df=test_df[target_column]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on  training and testing datasets ")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            logging.info("Exception occured at data transformation stage")
            raise CustomException(e, sys)