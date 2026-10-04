from distutils import dir_util

from src.constant.training_pipeline import SCHEMA_FILE_PATH

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from src.entity.config_entity import DataValidationConfig
from src.exception import SensorException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys)

    def drop_zero_std_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_columns = dataframe.select_dtypes(include=["number"]).columns
            zero_std_columns = [
                col
                for col in numeric_columns
                if dataframe[col].nunique() <= 1 or dataframe[col].std() == 0
            ]
            if zero_std_columns:
                logging.info(f"Dropping zero std columns: {zero_std_columns}")
                dataframe = dataframe.drop(columns=zero_std_columns)
            return dataframe
        except Exception as e:
            raise SensorException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            drop_columns = self._schema_config.get("drop_columns", [])
            # Also exclude zero-variance numeric columns
            numeric_columns = dataframe.select_dtypes(include=["number"]).columns
            zero_variance_columns = [
                col
                for col in numeric_columns
                if dataframe[col].nunique() <= 1 or dataframe[col].std() == 0
            ]
            all_drop_columns = set(drop_columns + zero_variance_columns)
            # Schema columns is a list of dicts, extract keys
            schema_columns = [list(d.keys())[0] for d in self._schema_config["columns"]]
            expected_columns = [
                col for col in schema_columns if col not in all_drop_columns
            ]
            number_of_columns = len(expected_columns)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise SensorException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            drop_columns = self._schema_config.get("drop_columns", [])
            # Also exclude zero-variance numeric columns
            numeric_columns = dataframe.select_dtypes(include=["number"]).columns
            zero_variance_columns = [
                col
                for col in numeric_columns
                if dataframe[col].nunique() <= 1 or dataframe[col].std() == 0
            ]
            all_drop_columns = set(drop_columns + zero_variance_columns)
            numerical_columns = [
                col
                for col in self._schema_config["numerical_columns"]
                if col not in all_drop_columns
            ]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []

            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)

            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")

            return numerical_column_present

        except Exception as e:
            raise SensorException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        column: {
                            "p_value": float(is_same_dist.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Compute zero-variance columns on original data (before dropping)
            numeric_columns = train_dataframe.select_dtypes(include=["number"]).columns
            zero_variance_columns = [
                col
                for col in numeric_columns
                if train_dataframe[col].nunique() <= 1 or train_dataframe[col].std() == 0
            ]
            if zero_variance_columns:
                logging.info(f"Zero variance columns detected: {zero_variance_columns}")

            # Drop zero-variance columns
            if zero_variance_columns:
                train_dataframe = train_dataframe.drop(columns=zero_variance_columns)
                test_dataframe = test_dataframe.drop(columns=zero_variance_columns)

            drop_columns = self._schema_config.get("drop_columns", [])
            all_drop_columns = set(drop_columns + zero_variance_columns)
            schema_columns = [list(d.keys())[0] for d in self._schema_config["columns"]]
            expected_columns = [col for col in schema_columns if col not in all_drop_columns]
            number_of_columns = len(expected_columns)

            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Train data frame has columns: {len(train_dataframe.columns)}")
            logging.info(f"Test data frame has columns: {len(test_dataframe.columns)}")

            if len(train_dataframe.columns) != number_of_columns:
                error_message = (
                    f"{error_message}Train dataframe does not contain all columns.\n"
                )
            if len(test_dataframe.columns) != number_of_columns:
                error_message = (
                    f"{error_message}Test dataframe does not contain all columns.\n"
                )

            # Validate numerical columns
            numerical_columns = [
                col
                for col in self._schema_config["numerical_columns"]
                if col not in all_drop_columns
            ]
            dataframe_columns = train_dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []

            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)

            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")

            if not numerical_column_present:
                error_message = f"{error_message}Train dataframe does not contain all numerical columns.\n"

            dataframe_columns = test_dataframe.columns
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)

            if not numerical_column_present:
                error_message = f"{error_message}Test dataframe does not contain all numerical columns.\n"

            if len(error_message) > 0:
                raise Exception(error_message)

            status = self.detect_dataset_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )

            # Save validated dataframes to valid_data_dir
            valid_train_file_path = self.data_validation_config.valid_train_file_path
            valid_test_file_path = self.data_validation_config.valid_test_file_path

            os.makedirs(os.path.dirname(valid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(valid_test_file_path), exist_ok=True)

            train_dataframe.to_csv(valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)
        





