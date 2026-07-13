import joblib
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - fallback for environments without xgboost
    XGBRegressor = None

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

DATASET_PATH = Path(__file__).resolve().parent / "House Price.csv"
MODEL_PATH = Path(__file__).resolve().parent / "house_price_model.joblib"

st.markdown(
    """
    <style>
    .main {padding-top: 0.5rem;}
    .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
    .stApp {background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);}
    .hero {
        background: linear-gradient(120deg, #0f172a, #2563eb);
        padding: 1.4rem 1.5rem;
        border-radius: 18px;
        color: white;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.22);
        margin-bottom: 1rem;
    }
    .card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
    }
    .form-shell {
        background: white;
        border-radius: 20px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }
    .form-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .form-help {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def build_model():
    if not DATASET_PATH.exists():
        st.error(f"Dataset not found at {DATASET_PATH}")
        st.stop()

    df = pd.read_csv(DATASET_PATH)
    df = df.copy()

    df["CITY"] = df["ADDRESS"].fillna("").str.split(",").str[-1].str.strip()
    df["CITY"] = df["CITY"].replace("", "Unknown")

    required_columns = [
        "POSTED_BY",
        "UNDER_CONSTRUCTION",
        "RERA",
        "BHK_NO.",
        "BHK_OR_RK",
        "SQUARE_FT",
        "READY_TO_MOVE",
        "RESALE",
        "LONGITUDE",
        "LATITUDE",
        "CITY",
        "TARGET(PRICE_IN_LACS)",
    ]
    df = df[required_columns].dropna()

    q1 = df["TARGET(PRICE_IN_LACS)"].quantile(0.25)
    q3 = df["TARGET(PRICE_IN_LACS)"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df["TARGET(PRICE_IN_LACS)"] >= lower_bound) & (df["TARGET(PRICE_IN_LACS)"] <= upper_bound)]

    X = df.drop(columns=["TARGET(PRICE_IN_LACS)"])
    y = df["TARGET(PRICE_IN_LACS)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    categorical_features = ["POSTED_BY", "BHK_OR_RK", "CITY"]
    numerical_features = [
        "UNDER_CONSTRUCTION",
        "RERA",
        "BHK_NO.",
        "SQUARE_FT",
        "READY_TO_MOVE",
        "RESALE",
        "LONGITUDE",
        "LATITUDE",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical_features),
        ]
    )

    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
        )
        model_name = "XGBoost"
    else:
        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "Random Forest"

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    test_pred = pipeline.predict(X_test)
    metrics = {
        "model": model_name,
        "r2": r2_score(y_test, test_pred),
        "mae": mean_absolute_error(y_test, test_pred),
        "rmse": mean_squared_error(y_test, test_pred) ** 0.5,
        "rows": len(df),
    }
    return pipeline, metrics, df


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            pipeline = joblib.load(MODEL_PATH)
            return pipeline, None, None
        except Exception:
            pass

    return build_model()[0], build_model()[1], build_model()[2]


pipeline, metrics, df = load_model()
if metrics is None:
    metrics = {"model": "Loaded model", "r2": "n/a", "mae": "n/a", "rmse": "n/a", "rows": len(df) if df is not None else 0}

st.markdown(
    """
    <div class='hero'>
        <h2>🏠 House Price Predictor</h2>
        <p>Estimate a property’s price from size, location, and listing details using a trained regression model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<div class='form-shell'>
    <div class='form-title'>🏡 Property Details</div>
    <div class='form-help'>Fill in the details below to estimate the market price in lakhs.</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        posted_by = st.selectbox("Posted by", ["Owner", "Dealer", "Builder"])
        under_construction = st.selectbox("Under construction", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        rera = st.selectbox("RERA approved", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        bhk_no = st.slider("BHK count", min_value=1, max_value=6, value=2)
        bhk_or_rk = st.selectbox("Type", ["BHK", "RK"])
    with col2:
        square_ft = st.number_input("Square feet", min_value=300.0, max_value=10000.0, value=1300.0, step=50.0)
        ready_to_move = st.selectbox("Ready to move", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        resale = st.selectbox("Resale", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        latitude = st.number_input("Latitude", min_value=8.0, max_value=35.0, value=12.97, step=0.01)
        longitude = st.number_input("Longitude", min_value=72.0, max_value=92.0, value=77.60, step=0.01)
    address = st.text_input("Address or locality", value="Bangalore")
    submitted = st.form_submit_button("Predict Price", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    city = address.strip().split(",")[-1].strip() if address.strip() else "Unknown"
    if not city:
        city = "Unknown"

    input_df = pd.DataFrame(
        [{
            "POSTED_BY": posted_by,
            "UNDER_CONSTRUCTION": under_construction,
            "RERA": rera,
            "BHK_NO.": bhk_no,
            "BHK_OR_RK": bhk_or_rk,
            "SQUARE_FT": square_ft,
            "READY_TO_MOVE": ready_to_move,
            "RESALE": resale,
            "LONGITUDE": longitude,
            "LATITUDE": latitude,
            "CITY": city,
        }]
    )

    try:
        prediction = float(pipeline.predict(input_df)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.markdown(
        f"""
        <div class='card' style='margin-top: 1rem;'>
            <h3>💰 Estimated Price</h3>
            <h1>₹{prediction:,.2f} Lakhs</h1>
            <p>This estimate is based on the trained house-price model and the details you supplied.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

