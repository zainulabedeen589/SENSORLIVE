# SensorLive - Run Guide

Complete guide to run the Sensor Fault Detection project locally and in production.

---

## Prerequisites

- Python 3.8+
- MongoDB (local or Atlas)
- AWS Account (for S3 model storage - optional)
- Git

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-repo/SensorLive.git
cd SensorLive

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file in project root:

```env
MONGO_DB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGO_DATABASE_NAME=ineuron
MONGO_COLLECTION_NAME=sensor
```

Or use `env.yaml` for local MongoDB:
```yaml
MONGO_DB_URL: mongodb://localhost:27017/
```

### 3. Load Data to MongoDB (One-time)

```bash
# If you have the training CSV
python get_data.py
```

This loads the dataset into MongoDB collection `sensor` in database `ineuron`.

---

## Running the Training Pipeline

### Option A: Direct Python (Recommended)

```bash
python -c "
from src.pipeline.training_pipeline import TrainPipeline
tp = TrainPipeline()
tp.run_pipeline()
"
```

### Option B: Via Main Module

```bash
python main.py
# Then visit: http://localhost:8080/train
```

### Option C: Using FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
# Train: GET http://localhost:8080/train
```

**Expected Output:**
```
Training pipeline completed successfully!
```

Artifacts will be in:
- `artifact/` - Pipeline artifacts per timestamp
- `saved_models/` - Trained model with timestamp

---

## Running Predictions

### 1. FastAPI Endpoint

```bash
# Start server
uvicorn main:app --host 0.0.0.0 --port 8080

# Predict (POST with CSV file)
curl -X POST "http://localhost:8080/predict" \
  -H "accept: text/csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data.csv" \
  --output predictions.csv
```

### 2. Python Direct

```bash
python -c "
import pandas as pd
from src.ml.model.estimator import ModelResolver, TargetValueMapping
from src.utils.main_utils import load_object
from src.constant.training_pipeline import SAVED_MODEL_DIR

model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
model = load_object(file_path=model_resolver.get_best_model_path())

# Load your data (must have same features as training)
df = pd.read_csv('your_test_data.csv')

# Predict
y_pred = model.predict(df)
df['predicted_column'] = y_pred
df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)
df.to_csv('predictions.csv', index=False)
"
```

---

## Running Streamlit Dashboard

```bash
streamlit run app.py
```

**Features:**
- Dashboard with metrics
- Dataset explorer
- Batch prediction with CSV upload
- Training pipeline trigger
- Analytics & visualizations

**Access:** http://localhost:8501

---

## Project Structure

```
SensorLive/
├── app.py                    # Streamlit dashboard
├── main.py                   # FastAPI application
├── get_data.py               # Load CSV to MongoDB
├── requirements.txt          # Dependencies
├── env.yaml                  # Local env config
├── .env                      # Environment variables
├── config/
│   └── schema.yaml           # Data schema
├── src/
│   ├── pipeline/
│   │   └── training_pipeline.py  # Main pipeline orchestrator
│   ├── components/
│   │   ├── data_ingestion.py     # MongoDB → CSV split
│   │   ├── data_validation.py    # Schema & drift check
│   │   ├── data_transformation.py # Preprocessing + SMOTE
│   │   ├── model_trainer.py      # XGBoost training
│   │   ├── model_evaluation.py   # Model comparison
│   │   └── model_pusher.py       # Save to S3/local
│   ├── ml/
│   │   ├── model/estimator.py    # Model wrapper + resolver
│   │   └── metric/classification_metric.py
│   ├── data_access/sensor_data.py # MongoDB operations
│   ├── configuration/mongo_db_connection.py
│   ├── entity/                   # Config & artifact dataclasses
│   ├── constant/                 # Constants & schema
│   ├── utils/main_utils.py       # YAML, pickle, numpy I/O
│   ├── logger.py                 # Logging config
│   └── exception.py              # Custom exceptions
└── artifact/                     # Generated pipeline artifacts
```

---

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **Data Ingestion** | MongoDB → train/test CSV | `artifact/.../data_ingestion/ingested/` |
| **Data Validation** | Schema check, drift detection, zero-variance drop | `artifact/.../data_validation/validated/` |
| **Data Transformation** | Impute → RobustScale → SMOTE | `artifact/.../data_transformation/transformed/` |
| **Model Trainer** | XGBoost with metrics | `artifact/.../model_trainer/trained_model/model.pkl` |
| **Model Evaluation** | Compare with best model | Accept/Reject decision |
| **Model Pusher** | Copy to `saved_models/` | `saved_models/<timestamp>/model.pkl` |

---

## Configuration

### Key Constants (`src/constant/training_pipeline/__init__.py`)

```python
TARGET_COLUMN = "class"
TRAIN_TEST_SPLIT_RATIO = 0.2
EXPECTED_ACCURACY = 0.6
OVERFITTING_THRESHOLD = 0.05
MODEL_EVALUATION_THRESHOLD = 0.02
```

### Schema (`config/schema.yaml`)
- `columns`: All expected columns with types
- `numerical_columns`: List of numerical features
- `drop_columns`: Columns to exclude

---

## Docker Deployment

```dockerfile
# Build
docker build -t sensorlive .

# Run
docker run -d -p 8080:8080 \
  -e MONGO_DB_URL="your_mongo_url" \
  -e AWS_ACCESS_KEY_ID="xxx" \
  -e AWS_SECRET_ACCESS_KEY="xxx" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  sensorlive
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Swagger UI redirect |
| GET | `/docs` | API documentation |
| GET | `/train` | Trigger training pipeline |
| POST | `/predict` | Batch prediction (CSV upload) |

---

## Troubleshooting

### MongoDB Connection Failed
```bash
# Check .env has correct URL
cat .env

# Test connection
python -c "from src.configuration.mongo_db_connection import MongoDBClient; print(MongoDBClient().database.list_collection_names())"
```

### Model Not Found
```bash
# Run training first
python -c "from src.pipeline.training_pipeline import TrainPipeline; TrainPipeline().run_pipeline()"

# Check saved_models
ls -la saved_models/
```

### Feature Mismatch in Prediction
- Ensure input CSV has same columns as training data
- Zero-variance columns are auto-dropped
- Use `prepare_input_data()` from `main.py` for alignment

### Streamlit Warnings
The `ScriptRunContext` warnings are normal when importing Streamlit outside `streamlit run`.

---

## Monitoring

- **Logs**: `logs/YYYY_MM_DD_HH_MM_SS.log`
- **Drift Reports**: `artifact/.../data_validation/drift_report/report.yaml`
- **Model Metrics**: Training artifacts include F1, Precision, Recall

---

## License

Internal use - SensorLive AI Platform