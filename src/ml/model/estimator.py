from src.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os
import sys
import numpy as np
import pandas as pd
from src.exception import SensorException

class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))




#Write a code to train model and check the accuracy.

class SensorModel:

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise SensorException(e, sys)

    def _sanitize_input(self, x):
        try:
            if isinstance(x, pd.DataFrame):
                sanitized_df = x.copy()
                for col in sanitized_df.columns:
                    if pd.api.types.is_object_dtype(sanitized_df[col]) or pd.api.types.is_string_dtype(sanitized_df[col]):
                        sanitized_df[col] = sanitized_df[col].replace(r"^\s*$", np.nan, regex=True)
                        sanitized_df[col] = pd.to_numeric(sanitized_df[col], errors="coerce")
                return sanitized_df

            if isinstance(x, pd.Series):
                sanitized_series = x.copy()
                sanitized_series = sanitized_series.replace(r"^\s*$", np.nan, regex=True)
                return pd.to_numeric(sanitized_series, errors="coerce")

            array = np.asarray(x, dtype=object)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            return pd.DataFrame(array)
        except Exception as e:
            raise SensorException(e, sys)

    def predict(self, x):
        try:
            sanitized_x = self._sanitize_input(x)
            x_transform = self.preprocessor.transform(sanitized_x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise SensorException(e, sys)
    

class ModelResolver:

    def __init__(self, model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise SensorException(e, sys)

    def get_best_model_path(self) -> str:
        try:
            if not os.path.exists(self.model_dir):
                raise Exception(f"Model directory {self.model_dir} does not exist")

            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                raise Exception("No model timestamps found")

            valid_timestamps = []
            for ts in timestamps:
                try:
                    valid_timestamps.append(int(ts))
                except ValueError:
                    continue

            if not valid_timestamps:
                raise Exception("No valid timestamp directories found")

            latest_timestamp = max(valid_timestamps)
            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}", MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise SensorException(e, sys)

    def is_model_exists(self) -> bool:
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False

            valid_timestamps = []
            for ts in timestamps:
                try:
                    valid_timestamps.append(int(ts))
                except ValueError:
                    continue

            if not valid_timestamps:
                return False

            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise SensorException(e, sys)


