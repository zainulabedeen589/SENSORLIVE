import pandas as pd
from src.exception import SensorException
import os
import sys
from src.logger import logging
from src.pipeline.training_pipeline import TrainPipeline
from src.utils.main_utils import read_yaml_file
from src.constant.training_pipeline import SAVED_MODEL_DIR
from fastapi import FastAPI, File, UploadFile
from src.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response, StreamingResponse
from src.ml.model.estimator import ModelResolver, TargetValueMapping
from src.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import io

env_file_path = os.path.join(os.getcwd(), "env.yaml")


def set_env_variable(env_file_path):
    if os.getenv("MONGO_DB_URL", None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ["MONGO_DB_URL"] = env_config["MONGO_DB_URL"]


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


def prepare_input_data(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Prepare input data for prediction by ensuring feature alignment."""
    # Drop target column if present
    if "class" in df.columns:
        df = df.drop(columns=["class"])
    
    # Get expected feature names from preprocessor
    expected_features = preprocessor.feature_names_in_
    
    # Ensure we only have expected features
    missing_features = [f for f in expected_features if f not in df.columns]
    extra_features = [f for f in df.columns if f not in expected_features]
    
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    if extra_features:
        df = df[expected_features]
    
    return df


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        
        # Prepare input data with correct features
        df = prepare_input_data(df, model.preprocessor)
        
        y_pred = model.predict(df)
        df["predicted_column"] = y_pred
        df["predicted_column"].replace(TargetValueMapping().reverse_mapping(), inplace=True)

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as e:
        raise SensorException(e, sys)


def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)