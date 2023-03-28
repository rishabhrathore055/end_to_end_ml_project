import sys
import os
from Dataclasses import Dataclasses

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer   
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor_obj.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("std_scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns standard complteted")

            logging.info("Categorical columns encoding completed")


        except Exception as error:
            raise CustomException(error, sys)