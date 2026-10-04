from pathlib import Path
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# SENSORLIVE PROJECT IMPORTS
# ============================================================

from src.pipeline.training_pipeline import TrainPipeline
from src.ml.model.estimator import ModelResolver, TargetValueMapping
from src.utils.main_utils import load_object, read_yaml_file
from src.constant.training_pipeline import SAVED_MODEL_DIR


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SensorLive AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE_PATH = ROOT_DIR / "env.yaml"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {
    "page": "Dashboard",
    "prediction_df": None,
    "uploaded_df": None,
    "prediction_result": None,
    "last_prediction_file": None,
    "prediction_mode": "CSV Batch Prediction",
}

for key, value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# ENVIRONMENT
# ============================================================

def set_env_variable():

    """
    Load MONGO_DB_URL from env.yaml if it is not already present.
    """

    try:

        if os.getenv("MONGO_DB_URL") is None:

            if not ENV_FILE_PATH.exists():
                return False

            env_config = read_yaml_file(
                str(ENV_FILE_PATH)
            )

            mongo_url = env_config.get(
                "MONGO_DB_URL"
            )

            if mongo_url:
                os.environ["MONGO_DB_URL"] = mongo_url

        return True

    except Exception:

        return False


ENV_READY = set_env_variable()


# ============================================================
# MODEL HELPERS
# ============================================================

def get_model_resolver():

    return ModelResolver(
        model_dir=SAVED_MODEL_DIR
    )


def is_model_available():

    try:

        resolver = get_model_resolver()

        return resolver.is_model_exists()

    except Exception:

        return False


def get_model_path():

    try:

        resolver = get_model_resolver()

        if not resolver.is_model_exists():
            return None

        return resolver.get_best_model_path()

    except Exception:

        return None


@st.cache_resource(show_spinner=False)
def load_model(model_path):

    """
    Cache loaded model so it isn't loaded
    on every Streamlit rerun.
    """

    return load_object(
        file_path=model_path
    )


def get_expected_features(model):

    """
    Extract expected feature names from
    trained model preprocessor.
    """

    try:

        preprocessor = model.preprocessor

        features = getattr(
            preprocessor,
            "feature_names_in_",
            None
        )

        if features is not None:
            return list(features)

        # ----------------------------------------------------
        # Fallback: try model itself
        # ----------------------------------------------------

        features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if features is not None:
            return list(features)

        return []

    except Exception:

        return []


# ============================================================
# FEATURE TYPE HELPERS
# ============================================================

def get_training_feature_types(model):

    """
    Try to determine original feature types from
    the fitted preprocessor.

    Returns:

        {
            "feature": "numeric",
            "feature2": "categorical"
        }
    """

    feature_types = {}

    try:

        preprocessor = model.preprocessor

        transformers = getattr(
            preprocessor,
            "transformers_",
            []
        )

        for transformer_name, transformer, columns in transformers:

            if transformer_name == "remainder":
                continue

            if columns is None:
                continue

            if isinstance(columns, str):
                columns = [columns]

            transformer_text = str(
                transformer
            ).lower()

            for column in columns:

                if (
                    "onehot" in transformer_text
                    or "ordinal" in transformer_text
                    or "encoder" in transformer_text
                    or "categorical" in transformer_text
                ):

                    feature_types[column] = "categorical"

                else:

                    feature_types[column] = "numeric"

    except Exception:

        pass

    return feature_types


def get_numeric_default(feature):

    """
    Return a safe default numeric value.
    """

    name = str(feature).lower()

    if "age" in name:
        return 30.0

    if "year" in name:
        return 2024.0

    if "percent" in name:
        return 0.0

    if "rate" in name:
        return 0.0

    if "score" in name:
        return 0.0

    return 0.0


def get_categorical_options(model, feature):

    """
    Try to extract known categorical values from
    fitted preprocessing transformers.
    """

    options = []

    try:

        preprocessor = model.preprocessor

        transformers = getattr(
            preprocessor,
            "transformers_",
            []
        )

        for transformer_name, transformer, columns in transformers:

            if columns is None:
                continue

            if isinstance(columns, str):
                columns = [columns]

            if feature not in columns:
                continue

            # ------------------------------------------------
            # OneHotEncoder
            # ------------------------------------------------

            categories = getattr(
                transformer,
                "categories_",
                None
            )

            if categories is not None:

                index = list(columns).index(
                    feature
                )

                if index < len(categories):

                    options = [
                        str(value)
                        for value in categories[index]
                    ]

                    break

            # ------------------------------------------------
            # OrdinalEncoder
            # ------------------------------------------------

            categories = getattr(
                transformer,
                "categories_",
                None
            )

            if categories is not None:

                index = list(columns).index(
                    feature
                )

                if index < len(categories):

                    options = [
                        str(value)
                        for value in categories[index]
                    ]

                    break

    except Exception:

        pass

    return options


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_input_data(df, model):

    """
    Same feature-alignment logic as backend.

    1. Remove target column.
    2. Validate required features.
    3. Ignore extra columns.
    4. Reorder columns exactly like training.
    """

    working_df = df.copy()

    # --------------------------------------------------------
    # Remove target
    # --------------------------------------------------------

    if "class" in working_df.columns:

        working_df = working_df.drop(
            columns=["class"]
        )

    expected_features = get_expected_features(
        model
    )

    # --------------------------------------------------------
    # If feature names aren't available
    # --------------------------------------------------------

    if not expected_features:

        return (
            working_df,
            [],
            []
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    missing_features = [

        feature

        for feature in expected_features

        if feature not in working_df.columns

    ]

    # --------------------------------------------------------
    # Extra
    # --------------------------------------------------------

    extra_features = [

        feature

        for feature in working_df.columns

        if feature not in expected_features

    ]

    # --------------------------------------------------------
    # Stop if missing
    # --------------------------------------------------------

    if missing_features:

        raise ValueError(
            "Missing required features:\n\n"
            +
            "\n".join(
                f"• {feature}"
                for feature in missing_features
            )
        )

    # --------------------------------------------------------
    # Exact feature order
    # --------------------------------------------------------

    working_df = working_df[
        expected_features
    ]

    return (
        working_df,
        missing_features,
        extra_features
    )


# ============================================================
# TARGET MAPPING
# ============================================================

def apply_target_mapping(result_df):

    """
    Convert encoded prediction values back
    to original target labels.
    """

    try:

        reverse_mapping = (
            TargetValueMapping()
            .reverse_mapping()
        )

        result_df[
            "predicted_column"
        ] = (
            result_df[
                "predicted_column"
            ].replace(
                reverse_mapping
            )
        )

    except Exception:

        pass

    return result_df


# ============================================================
# PREDICTION
# ============================================================

def generate_predictions(input_df):

    """
    Complete local inference pipeline.

    No FastAPI.
    No HTTP.
    Direct model execution.
    """

    model_path = get_model_path()

    if model_path is None:

        raise FileNotFoundError(
            "No trained model is available."
        )

    model = load_model(
        str(model_path)
    )

    prepared_df, missing_features, extra_features = (
        prepare_input_data(
            input_df,
            model
        )
    )

    predictions = model.predict(
        prepared_df
    )

    result_df = prepared_df.copy()

    result_df[
        "predicted_column"
    ] = predictions

    result_df = apply_target_mapping(
        result_df
    )

    return (
        result_df,
        model,
        extra_features
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(59, 130, 246, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(139, 92, 246, 0.10),
                transparent 30%
            ),
            #070a12;

        color: #f8fafc;
    }

    .main {
        background: transparent;
    }

    section[data-testid="stMain"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #090d18 0%,
                #070a12 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .brand span {
        color: #60a5fa;
    }

    .brand-subtitle {
        color: #7f8ba3;
        font-size: 12px;
        margin-bottom: 25px;
    }

    .sidebar-section {
        color: #59657c;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 28px;
        margin-bottom: 8px;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(20, 31, 55, 0.96),
                rgba(10, 15, 29, 0.96)
            );

        border:
            1px solid rgba(96,165,250,0.15);

        border-radius: 24px;

        padding: 38px;

        margin-bottom: 28px;

        box-shadow:
            0 25px 80px rgba(0,0,0,0.30);
    }

    .hero-badge {
        display: inline-block;

        padding: 7px 13px;

        border-radius: 100px;

        background:
            rgba(34,197,94,0.10);

        border:
            1px solid rgba(34,197,94,0.20);

        color: #86efac;

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 1px;
    }

    .hero-title {
        font-size: 46px;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -2px;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #94a3b8;
        max-width: 720px;
        font-size: 16px;
        line-height: 1.7;
    }


    /* =====================================================
       SECTION HEADERS
       ===================================================== */

    .section-title {
        font-size: 22px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 18px;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    div[data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(17,25,43,0.95),
                rgba(10,15,27,0.95)
            );

        border:
            1px solid rgba(148,163,184,0.10);

        border-radius: 18px;

        padding: 20px;

        min-height: 125px;

        box-shadow:
            0 15px 40px rgba(0,0,0,0.15);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 800;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {

        border-radius: 12px;

        border:
            1px solid rgba(96,165,250,0.20);

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #4f46e5
            );

        color: white;

        font-weight: 700;

        min-height: 44px;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {

        border-color:
            rgba(147,197,253,0.6);

        transform:
            translateY(-1px);

        box-shadow:
            0 10px 30px rgba(37,99,235,0.25);
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {

        background:
            rgba(15,23,42,0.60);

        border-radius: 18px;
    }

    [data-testid="stFileUploaderDropzone"] {

        border:
            1px dashed rgba(96,165,250,0.30);

        border-radius: 16px;

        background:
            rgba(15,23,42,0.45);
    }


    /* =====================================================
       DATAFRAMES
       ===================================================== */

    [data-testid="stDataFrame"] {

        border-radius: 15px;

        overflow: hidden;

        border:
            1px solid rgba(148,163,184,0.10);
    }


    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {

        color: #64748b;

        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {

        color: #60a5fa;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    div[data-testid="stAlert"] {

        border-radius: 14px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {

        text-align: center;

        color: #475569;

        font-size: 11px;

        padding-top: 40px;

        padding-bottom: 10px;
    }

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            ⚡ Sensor<span>Live</span>
        </div>

        <div class="brand-subtitle">
            AI Prediction & MLOps Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    if is_model_available():

        st.success(
            "●  MODEL ONLINE"
        )

    else:

        st.warning(
            "●  MODEL NOT AVAILABLE"
        )

    st.markdown(
        '<div class="sidebar-section">Navigation</div>',
        unsafe_allow_html=True
    )

    pages = [
        "Dashboard",
        "Prediction",
        "Training",
        "Dataset",
        "Analytics",
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(
            st.session_state.page
        ),
        label_visibility="collapsed",
    )

    st.session_state.page = selected_page

    st.markdown(
        '<div class="sidebar-section">Model Engine</div>',
        unsafe_allow_html=True
    )

    model_path = get_model_path()

    if model_path:

        st.success(
            "✓ MODEL AVAILABLE"
        )

        model_name = Path(
            model_path
        ).name

        st.caption(
            model_name
        )

    else:

        st.error(
            "✕ MODEL UNAVAILABLE"
        )

    st.divider()

    st.caption(
        "SensorLive AI"
    )

    st.caption(
        "ML • MLOps • Batch Inference"
    )


# ============================================================
# COMMON MODEL INFORMATION
# ============================================================

MODEL_AVAILABLE = is_model_available()

MODEL_PATH = get_model_path()


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":


    # ============================================================
# HERO SECTION
# ============================================================

    st.html("""
    <style>

    .hero {
        position: relative;
        overflow: hidden;

        padding: 42px 46px;
        margin: 0 0 30px 0;

        border-radius: 24px;

        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(59, 130, 246, 0.22),
                transparent 30%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(139, 92, 246, 0.14),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #111827 0%,
                #0b1220 100%
            );

        border: 1px solid rgba(255, 255, 255, 0.08);

        box-shadow:
            0 25px 80px rgba(0, 0, 0, 0.35);

        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .hero::after {
        content: "";

        position: absolute;

        width: 280px;
        height: 280px;

        right: -110px;
        bottom: -130px;

        border-radius: 50%;

        background: rgba(59, 130, 246, 0.10);

        filter: blur(25px);

        pointer-events: none;
    }

    .hero-badge {
        position: relative;
        z-index: 2;

        display: inline-block;

        padding: 8px 14px;
        margin-bottom: 18px;

        border-radius: 999px;

        background: rgba(59, 130, 246, 0.12);

        border: 1px solid rgba(59, 130, 246, 0.30);

        color: #60a5fa;

        font-size: 11px;
        font-weight: 800;

        letter-spacing: 1.2px;

        text-transform: uppercase;
    }

    .hero-title {
        position: relative;
        z-index: 2;

        margin: 0 0 18px 0;

        color: #ffffff;

        font-size: 64px;

        font-weight: 900;

        line-height: 1;

        letter-spacing: -3px;
    }

    .hero-description {
        position: relative;
        z-index: 2;

        max-width: 720px;

        color: #94a3b8;

        font-size: 17px;

        line-height: 1.7;
    }

    </style>

    <div class="hero">

        <div class="hero-badge">
            ● BATCH + MANUAL INFERENCE ENGINE
        </div>

        <div class="hero-title">
            SensorLive AI
        </div>

        <div class="hero-description">
            Intelligent machine-learning platform for sensor
            data processing, model training, batch prediction,
            manual prediction and predictive analytics.
        </div>

    </div>
    """)

    
    
    
    
    
        
    # st.markdown(
    #     """
    #     <div class="hero">

    #         <div class="hero-badge">
    #             ● BATCH + MANUAL INFERENCE ENGINE
    #         </div>

    #         <div class="hero-title">
    #             SensorLive AI
    #         </div>

    #         <div class="hero-description">
    #             Intelligent machine-learning platform for sensor
    #             data processing, model training, batch prediction,
    #             manual prediction and predictive analytics.
    #         </div>

    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    
    # st.markdown(
    #     """
    #     <div class="hero">

    #         <div class="hero-badge">
    #             ● BATCH + MANUAL INFERENCE ENGINE
    #         </div>

    #         <div class="hero-title">
    #             SensorLive AI
    #         </div>

    #         <div class="hero-description">
    #             Intelligent machine-learning platform for sensor
    #             data processing, model training, batch prediction,
    #             manual prediction and predictive analytics.
    #         </div>

    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Current platform status</div>',
        unsafe_allow_html=True
    )

    model_status = (
        "READY"
        if MODEL_AVAILABLE
        else "OFFLINE"
    )

    prediction_count = 0

    if (
        st.session_state.prediction_result
        is not None
    ):

        prediction_count = len(
            st.session_state.prediction_result
        )

    cols = st.columns(4)

    with cols[0]:

        st.metric(
            "MODEL STATUS",
            model_status
        )

    with cols[1]:

        rows = 0

        if (
            st.session_state.uploaded_df
            is not None
        ):

            rows = len(
                st.session_state.uploaded_df
            )

        st.metric(
            "DATASET ROWS",
            f"{rows:,}"
        )

    with cols[2]:

        features = 0

        if (
            st.session_state.uploaded_df
            is not None
        ):

            features = len(
                st.session_state.uploaded_df.columns
            )

        st.metric(
            "FEATURES",
            features
        )

    with cols[3]:

        st.metric(
            "PREDICTIONS",
            f"{prediction_count:,}"
        )

    st.markdown(
        '<div class="section-title">ML Lifecycle</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'End-to-end machine learning workflow'
        '</div>',
        unsafe_allow_html=True
    )

    lifecycle = [
        ("01", "DATA", "CSV / Manual input"),
        ("02", "QUALITY", "Validation"),
        ("03", "FEATURES", "Feature alignment"),
        ("04", "MODEL", "Trained estimator"),
        ("05", "PREDICT", "Inference"),
        ("06", "ANALYZE", "Prediction analytics"),
    ]

    lifecycle_cols = st.columns(6)

    for col, item in zip(
        lifecycle_cols,
        lifecycle
    ):

        number, title, description = item

        with col:

            st.info(
                f"**{number}**\n\n"
                f"**{title}**\n\n"
                f"{description}"
            )

    st.markdown(
        '<div class="section-title">Quick Start</div>',
        unsafe_allow_html=True
    )

    quick_cols = st.columns(3)

    with quick_cols[0]:

        if st.button(
            "🚀  Run Prediction",
            use_container_width=True
        ):

            st.session_state.page = "Prediction"

            st.rerun()

    with quick_cols[1]:

        if st.button(
            "🧠  Train Model",
            use_container_width=True
        ):

            st.session_state.page = "Training"

            st.rerun()

    with quick_cols[2]:

        if st.button(
            "📊  Explore Dataset",
            use_container_width=True
        ):

            st.session_state.page = "Dataset"

            st.rerun()


# ============================================================
# PREDICTION
# ============================================================

elif st.session_state.page == "Prediction":

    st.title(
        "Prediction Workbench"
    )

    st.caption(
        "Generate predictions either from a CSV file "
        "or by entering sensor values manually."
    )

    if not MODEL_AVAILABLE:

        st.error(
            "No trained model is available. "
            "Train the model first."
        )

        if st.button(
            "Go to Training"
        ):

            st.session_state.page = "Training"

            st.rerun()

        st.stop()

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "🧠 Model Information",
        expanded=True
    ):

        info_cols = st.columns(3)

        with info_cols[0]:

            st.write(
                "**Status**"
            )

            st.success(
                "READY"
            )

        with info_cols[1]:

            st.write(
                "**Model File**"
            )

            st.code(
                Path(MODEL_PATH).name
            )

        with info_cols[2]:

            st.write(
                "**Model Directory**"
            )

            st.caption(
                str(
                    Path(MODEL_PATH).parent
                )
            )

    # ========================================================
    # PREDICTION MODE
    # ========================================================

    st.markdown(
        "### Select Prediction Method"
    )

    prediction_mode = st.radio(
        "Prediction Method",
        [
            "CSV Batch Prediction",
            "Manual Input Prediction",
        ],
        horizontal=True,
        key="prediction_mode_selector",
        label_visibility="collapsed"
    )

    st.session_state.prediction_mode = (
        prediction_mode
    )

    # ========================================================
    # CSV BATCH PREDICTION
    # ========================================================

    if prediction_mode == "CSV Batch Prediction":

        st.markdown(
            "### Upload Prediction Dataset"
        )

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help=(
                "CSV must contain all features required "
                "by the trained model."
            ),
            key="prediction_uploader"
        )

        if uploaded_file is None:

            st.info(
                "Upload a CSV file to start batch prediction."
            )

        else:

            # ------------------------------------------------
            # Read CSV
            # ------------------------------------------------

            try:

                df = pd.read_csv(
                    uploaded_file
                )

                st.session_state.uploaded_df = df

            except Exception as e:

                st.error(
                    f"Unable to read CSV: {e}"
                )

                st.stop()

            # ------------------------------------------------
            # Dataset Summary
            # ------------------------------------------------

            st.markdown(
                "### Dataset Validation"
            )

            summary_cols = st.columns(4)

            with summary_cols[0]:

                st.metric(
                    "Rows",
                    f"{len(df):,}"
                )

            with summary_cols[1]:

                st.metric(
                    "Columns",
                    len(df.columns)
                )

            with summary_cols[2]:

                st.metric(
                    "Missing Values",
                    int(
                        df.isna()
                        .sum()
                        .sum()
                    )
                )

            with summary_cols[3]:

                st.metric(
                    "Duplicate Rows",
                    int(
                        df.duplicated()
                        .sum()
                    )
                )

            # ------------------------------------------------
            # Model Compatibility
            # ------------------------------------------------

            try:

                model = load_model(
                    str(MODEL_PATH)
                )

                expected_features = (
                    get_expected_features(
                        model
                    )
                )

                actual_features = list(
                    df.columns
                )

                missing_features = [

                    feature

                    for feature
                    in expected_features

                    if feature
                    not in actual_features
                ]

                extra_features = [

                    feature

                    for feature
                    in actual_features

                    if feature
                    not in expected_features

                    and feature != "class"
                ]

            except Exception as e:

                st.error(
                    f"Unable to inspect model: {e}"
                )

                st.stop()

            # ------------------------------------------------
            # Feature Compatibility
            # ------------------------------------------------

            if expected_features:

                if missing_features:

                    st.error(
                        f"Missing "
                        f"{len(missing_features)} "
                        f"required feature(s)."
                    )

                    with st.expander(
                        "Show missing features"
                    ):

                        for feature in missing_features:

                            st.write(
                                f"• `{feature}`"
                            )

                    st.stop()

                else:

                    st.success(
                        "✓ Feature compatibility check passed — "
                        f"{len(expected_features)} features detected."
                    )

                if extra_features:

                    with st.expander(
                        f"ℹ {len(extra_features)} "
                        "extra column(s) will be ignored"
                    ):

                        for feature in extra_features:

                            st.write(
                                f"• `{feature}`"
                            )

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            tab1, tab2, tab3 = st.tabs(
                [
                    "Preview",
                    "Schema",
                    "Statistics"
                ]
            )

            with tab1:

                st.dataframe(
                    df.head(100),
                    use_container_width=True,
                    height=380
                )

            with tab2:

                schema_df = pd.DataFrame(
                    {
                        "Feature": df.columns,

                        "Data Type": [
                            str(dtype)
                            for dtype in df.dtypes
                        ],

                        "Missing": [
                            int(
                                df[column]
                                .isna()
                                .sum()
                            )
                            for column
                            in df.columns
                        ],

                        "Unique": [
                            int(
                                df[column]
                                .nunique()
                            )
                            for column
                            in df.columns
                        ],
                    }
                )

                st.dataframe(
                    schema_df,
                    use_container_width=True,
                    height=380
                )

            with tab3:

                st.dataframe(
                    df.describe(
                        include="all"
                    ).T,
                    use_container_width=True,
                    height=380
                )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            st.markdown(
                "### Run Batch Inference"
            )

            predict_col1, predict_col2 = st.columns(
                [3, 1]
            )

            with predict_col1:

                st.write(
                    "The pipeline will validate the features, "
                    "align the columns and run the trained model."
                )

            with predict_col2:

                run_prediction = st.button(
                    "⚡ Generate Predictions",
                    type="primary",
                    use_container_width=True,
                    key="run_csv_prediction"
                )

            if run_prediction:

                try:

                    with st.spinner(
                        "Running model inference..."
                    ):

                        (
                            result_df,
                            model,
                            extra_columns
                        ) = generate_predictions(
                            df
                        )

                    st.session_state.prediction_result = (
                        result_df
                    )

                    csv_bytes = (
                        result_df
                        .to_csv(
                            index=False
                        )
                        .encode("utf-8")
                    )

                    st.session_state.last_prediction_file = (
                        csv_bytes
                    )

                    st.success(
                        "Prediction completed successfully — "
                        f"{len(result_df):,} rows processed."
                    )

                except Exception:

                    st.error(
                        "Prediction failed."
                    )

                    with st.expander(
                        "Technical error details"
                    ):

                        st.code(
                            traceback.format_exc()
                        )

    # ========================================================
    # MANUAL INPUT PREDICTION
    # ========================================================

    else:

        st.markdown(
            "### Manual Sensor Input"
        )

        st.caption(
            "Enter the sensor values below. "
            "The application will create a single-row dataframe "
            "and send it through the same trained model."
        )

        # ----------------------------------------------------
        # Load Model
        # ----------------------------------------------------

        try:

            model = load_model(
                str(MODEL_PATH)
            )

            expected_features = (
                get_expected_features(
                    model
                )
            )

            feature_types = (
                get_training_feature_types(
                    model
                )
            )

        except Exception as e:

            st.error(
                f"Unable to load model: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # Check features
        # ----------------------------------------------------

        if not expected_features:

            st.error(
                "The trained model does not expose "
                "feature names. Manual input cannot be "
                "generated automatically."
            )

            st.info(
                "In this case, the model preprocessor needs "
                "`feature_names_in_` or the application must "
                "be configured with the training feature list."
            )

            st.stop()

        st.success(
            f"✓ {len(expected_features)} model features detected."
        )

        # ----------------------------------------------------
        # Manual Input Form
        # ----------------------------------------------------

        manual_input = {}

        with st.form(
            "manual_prediction_form"
        ):

            # Use 2 columns for cleaner UI.
            input_cols = st.columns(2)

            for index, feature in enumerate(
                expected_features
            ):

                with input_cols[index % 2]:

                    detected_type = (
                        feature_types.get(
                            feature
                        )
                    )

                    # ========================================
                    # CATEGORICAL
                    # ========================================

                    if detected_type == "categorical":

                        options = (
                            get_categorical_options(
                                model,
                                feature
                            )
                        )

                        if options:

                            manual_input[
                                feature
                            ] = st.selectbox(
                                label=feature,
                                options=options,
                                key=(
                                    f"manual_"
                                    f"{feature}"
                                )
                            )

                        else:

                            manual_input[
                                feature
                            ] = st.text_input(
                                label=feature,
                                key=(
                                    f"manual_"
                                    f"{feature}"
                                )
                            )

                    # ========================================
                    # NUMERIC
                    # ========================================

                    else:

                        manual_input[
                            feature
                        ] = st.number_input(
                            label=feature,
                            value=float(
                                get_numeric_default(
                                    feature
                                )
                            ),
                            key=(
                                f"manual_"
                                f"{feature}"
                            )
                        )

            st.markdown(
                "---"
            )

            submit_manual = st.form_submit_button(
                "⚡ Predict Sensor Status",
                type="primary",
                use_container_width=True
            )

        # ----------------------------------------------------
        # Execute Manual Prediction
        # ----------------------------------------------------

        if submit_manual:

            try:

                # --------------------------------------------
                # Create single-row dataframe
                # --------------------------------------------

                manual_df = pd.DataFrame(
                    [manual_input]
                )

                # --------------------------------------------
                # Force exact feature order
                # --------------------------------------------

                manual_df = manual_df[
                    expected_features
                ]

                # --------------------------------------------
                # Run same prediction pipeline
                # --------------------------------------------

                with st.spinner(
                    "Running AI prediction..."
                ):

                    (
                        result_df,
                        model,
                        extra_columns
                    ) = generate_predictions(
                        manual_df
                    )

                # --------------------------------------------
                # Store result
                # --------------------------------------------

                st.session_state.prediction_result = (
                    result_df
                )

                csv_bytes = (
                    result_df
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.session_state.last_prediction_file = (
                    csv_bytes
                )

                # --------------------------------------------
                # Display result
                # --------------------------------------------

                st.success(
                    "Prediction completed successfully."
                )

                st.markdown(
                    "### AI Prediction Result"
                )

                prediction_value = (
                    result_df[
                        "predicted_column"
                    ].iloc[0]
                )

                result_col1, result_col2 = st.columns(
                    [2, 3]
                )

                with result_col1:

                    st.metric(
                        "Predicted Class",
                        str(
                            prediction_value
                        )
                    )

                with result_col2:

                    st.info(
                        "The prediction above was generated "
                        "using the currently deployed trained model."
                    )

                st.markdown(
                    "### Input + Prediction"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                st.download_button(
                    label="⬇ Download Prediction CSV",
                    data=csv_bytes,
                    file_name="manual_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception:

                st.error(
                    "Manual prediction failed."
                )

                with st.expander(
                    "Technical error details"
                ):

                    st.code(
                        traceback.format_exc()
                    )

    # ========================================================
    # LAST RESULT
    # ========================================================

    if (
        st.session_state.prediction_result
        is not None
    ):

        # Don't duplicate the result immediately after
        # a manual prediction form.
        if prediction_mode == "CSV Batch Prediction":

            result_df = (
                st.session_state.prediction_result
            )

            st.markdown(
                "### Prediction Results"
            )

            result_cols = st.columns(3)

            with result_cols[0]:

                st.metric(
                    "Rows Processed",
                    f"{len(result_df):,}"
                )

            with result_cols[1]:

                unique_predictions = (
                    result_df[
                        "predicted_column"
                    ].nunique()
                )

                st.metric(
                    "Prediction Classes",
                    unique_predictions
                )

            with result_cols[2]:

                st.metric(
                    "Output Columns",
                    len(result_df.columns)
                )

            st.dataframe(
                result_df.head(100),
                use_container_width=True,
                height=400
            )

            st.download_button(
                label="⬇ Download predictions.csv",
                data=(
                    st.session_state
                    .last_prediction_file
                ),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_batch_predictions"
            )


# ============================================================
# TRAINING
# ============================================================

elif st.session_state.page == "Training":

    st.title(
        "Training Center"
    )

    st.caption(
        "Train a new SensorLive model using the existing "
        "training pipeline."
    )

    st.warning(
        "Training can take time and may consume significant "
        "CPU / memory depending on your pipeline."
    )

    st.markdown(
        "### Current Model"
    )

    if MODEL_AVAILABLE:

        model_cols = st.columns(3)

        with model_cols[0]:

            st.success(
                "MODEL AVAILABLE"
            )

        with model_cols[1]:

            st.write(
                "**Current Model**"
            )

            st.code(
                Path(MODEL_PATH).name
            )

        with model_cols[2]:

            try:

                model_size = (
                    Path(MODEL_PATH)
                    .stat()
                    .st_size
                    / 1024
                    / 1024
                )

                st.write(
                    "**Model Size**"
                )

                st.write(
                    f"{model_size:.2f} MB"
                )

            except Exception:

                st.write(
                    "Unknown"
                )

    else:

        st.info(
            "No trained model currently available."
        )

    st.markdown(
        "### Start Training"
    )

    confirm = st.checkbox(
        "I understand that starting training will execute "
        "the complete training pipeline."
    )

    if st.button(
        "🧠 Start Training",
        type="primary",
        disabled=not confirm,
        use_container_width=True
    ):

        try:

            with st.status(
                "Training SensorLive model...",
                expanded=True
            ) as status:

                st.write(
                    "Initializing training pipeline..."
                )

                train_pipeline = TrainPipeline()

                if train_pipeline.is_pipeline_running:

                    status.update(
                        label="Training already running",
                        state="error"
                    )

                    st.error(
                        "Training pipeline is already running."
                    )

                else:

                    st.write(
                        "Executing training pipeline..."
                    )

                    train_pipeline.run_pipeline()

                    status.update(
                        label="Training completed successfully",
                        state="complete"
                    )

                    st.success(
                        "Training completed successfully."
                    )

                    st.cache_resource.clear()

                    st.rerun()

        except Exception:

            st.error(
                "Training failed."
            )

            with st.expander(
                "Technical error details"
            ):

                st.code(
                    traceback.format_exc()
                )


# ============================================================
# DATASET
# ============================================================

elif st.session_state.page == "Dataset":

    st.title(
        "Dataset Explorer"
    )

    st.caption(
        "Inspect, validate and understand your sensor dataset."
    )

    uploaded_dataset = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"],
        key="dataset_explorer"
    )

    if uploaded_dataset is not None:

        try:

            dataset = pd.read_csv(
                uploaded_dataset
            )

            st.session_state.uploaded_df = (
                dataset
            )

        except Exception as e:

            st.error(
                f"Could not load dataset: {e}"
            )

            st.stop()

    else:

        dataset = (
            st.session_state.uploaded_df
        )

    if dataset is None:

        st.info(
            "Upload a CSV dataset to begin exploration."
        )

        st.stop()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    cols = st.columns(5)

    with cols[0]:

        st.metric(
            "Rows",
            f"{dataset.shape[0]:,}"
        )

    with cols[1]:

        st.metric(
            "Columns",
            dataset.shape[1]
        )

    with cols[2]:

        st.metric(
            "Missing",
            int(
                dataset.isna()
                .sum()
                .sum()
            )
        )

    with cols[3]:

        st.metric(
            "Duplicates",
            int(
                dataset.duplicated()
                .sum()
            )
        )

    with cols[4]:

        st.metric(
            "Memory",
            f"{dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "Data Preview",
            "Schema",
            "Missing Values",
            "Statistics"
        ]
    )

    with tabs[0]:

        st.dataframe(
            dataset,
            use_container_width=True,
            height=520
        )

    with tabs[1]:

        schema = pd.DataFrame(
            {
                "Feature": dataset.columns,

                "Type": [
                    str(x)
                    for x in dataset.dtypes
                ],

                "Non-Null": [
                    int(
                        dataset[col]
                        .notna()
                        .sum()
                    )
                    for col
                    in dataset.columns
                ],

                "Unique": [
                    int(
                        dataset[col]
                        .nunique()
                    )
                    for col
                    in dataset.columns
                ],
            }
        )

        st.dataframe(
            schema,
            use_container_width=True,
            height=500
        )

    with tabs[2]:

        missing = (
            dataset.isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        missing = missing[
            missing > 0
        ]

        if len(missing) == 0:

            st.success(
                "✓ No missing values detected."
            )

        else:

            missing_df = pd.DataFrame(
                {
                    "Feature": missing.index,

                    "Missing Values": (
                        missing.values
                    ),

                    "Missing %": (
                        missing.values
                        / len(dataset)
                        * 100
                    ).round(2)
                }
            )

            st.dataframe(
                missing_df,
                use_container_width=True
            )

            fig = px.bar(
                missing_df,
                x="Feature",
                y="Missing %",
                title="Missing Value Distribution"
            )

            fig.update_layout(
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with tabs[3]:

        st.dataframe(
            dataset.describe(
                include="all"
            ).T,
            use_container_width=True,
            height=500
        )


# ============================================================
# ANALYTICS
# ============================================================

elif st.session_state.page == "Analytics":

    st.title(
        "Prediction Analytics"
    )

    st.caption(
        "Explore the latest batch or manual prediction results."
    )

    result_df = (
        st.session_state.prediction_result
    )

    if result_df is None:

        st.info(
            "No prediction results available yet."
        )

        if st.button(
            "Go to Prediction"
        ):

            st.session_state.page = "Prediction"

            st.rerun()

        st.stop()

    if (
        "predicted_column"
        not in result_df.columns
    ):

        st.warning(
            "Prediction column is not available."
        )

        st.stop()

    predictions = (
        result_df[
            "predicted_column"
        ]
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    cols = st.columns(4)

    with cols[0]:

        st.metric(
            "Total Predictions",
            f"{len(predictions):,}"
        )

    with cols[1]:

        st.metric(
            "Unique Classes",
            predictions.nunique()
        )

    with cols[2]:

        mode_values = (
            predictions.mode()
        )

        most_frequent = (
            str(
                mode_values.iloc[0]
            )
            if not mode_values.empty
            else "N/A"
        )

        st.metric(
            "Most Frequent",
            most_frequent
        )

    with cols[3]:

        st.metric(
            "Missing Predictions",
            int(
                predictions.isna()
                .sum()
            )
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    st.markdown(
        "### Prediction Distribution"
    )

    distribution = (
        predictions
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "Prediction",
        "Count"
    ]

    chart_cols = st.columns(2)

    with chart_cols[0]:

        fig_bar = px.bar(
            distribution,
            x="Prediction",
            y="Count",
            title="Prediction Counts",
            text="Count"
        )

        fig_bar.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    with chart_cols[1]:

        fig_pie = px.pie(
            distribution,
            names="Prediction",
            values="Count",
            title="Prediction Composition",
            hole=0.55
        )

        fig_pie.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Numeric distributions
    # --------------------------------------------------------

    numeric_columns = (
        result_df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if numeric_columns:

        st.markdown(
            "### Feature Distribution"
        )

        selected_feature = st.selectbox(
            "Select numeric feature",
            numeric_columns
        )

        fig_hist = px.histogram(
            result_df,
            x=selected_feature,
            nbins=40,
            title=(
                f"Distribution — "
                f"{selected_feature}"
            )
        )

        fig_hist.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SensorLive AI • Machine Learning • MLOps •
        Batch + Manual Inference
    </div>
    """,
    unsafe_allow_html=True
)