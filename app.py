import streamlit as st
import pandas as pd
import joblib
from scipy.io import arff


# ============================================================
# LOAD DATASET
# ============================================================

data, meta = arff.loadarff(
    "data/Rice_Cammeo_Osmancik.arff"
)

df = pd.DataFrame(data)

df["Class"] = df["Class"].str.decode("utf-8")


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/rice_classifier.joblib"
)

encoder = joblib.load(
    "models/label_encoder.joblib"
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Perimeter",
    "Major_Axis_Length",
    "Area",
    "Convex_Area",
    "Eccentricity",
    "Minor_Axis_Length"
]


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Rice Variety Classifier",
    page_icon="🌾",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌾 Rice Variety Classification")

st.write(
    "Predict whether a rice grain belongs to "
    "**Cammeo** or **Osmancik** using physical measurements."
)

st.info(
    "The model uses six selected physical features "
    "and Logistic Regression."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Information")

st.sidebar.write(
    "**Algorithm:** Logistic Regression"
)

st.sidebar.write(
    "**Features:** 6"
)

st.sidebar.write(
    "**C:** 10"
)

st.sidebar.write(
    "**Solver:** liblinear"
)

st.sidebar.write(
    "**Test Accuracy:** 91.60%"
)

st.sidebar.write(
    "**Test F1 Score:** 91.58%"
)


# ============================================================
# INPUT METHOD
# ============================================================

st.header("Select Input Method")

input_method = st.radio(
    "Choose how you want to provide the rice measurements:",
    [
        "Use a real dataset sample",
        "Enter measurements manually"
    ]
)


# ============================================================
# REAL DATASET SAMPLE
# ============================================================

if input_method == "Use a real dataset sample":

    st.subheader("🌾 Real Dataset Sample")

    sample_number = st.number_input(
        "Select sample number",
        min_value=1,
        max_value=len(df),
        value=1,
        step=1
    )

    sample = df.iloc[sample_number - 1]

    st.write("### Input Measurements")

    sample_input = pd.DataFrame(
        {
            feature: [sample[feature]]
            for feature in features
        }
    )

    st.dataframe(
        sample_input,
        use_container_width=True
    )

    actual_class = sample["Class"]

    st.write(
        f"Actual dataset class: **{actual_class}**"
    )

    if st.button("🔍 Predict Selected Sample"):

        prediction = model.predict(
            sample_input
        )

        predicted_class = encoder.inverse_transform(
            prediction
        )[0]

        probabilities = model.predict_proba(
            sample_input
        )[0]

        confidence = max(probabilities) * 100

        st.success(
            f"🌾 Predicted Rice Variety: **{predicted_class}**"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        probability_df = pd.DataFrame(
            {
                "Rice Variety": encoder.classes_,
                "Probability": probabilities * 100
            }
        )

        st.subheader("Prediction Probabilities")

        st.bar_chart(
            probability_df.set_index(
                "Rice Variety"
            )
        )


# ============================================================
# MANUAL INPUT
# ============================================================

else:

    st.subheader("✏️ Enter Rice Measurements")

    st.caption(
        "Use values within the ranges of the original dataset."
    )

    # Get realistic ranges from dataset

    perimeter_min = float(df["Perimeter"].min())
    perimeter_max = float(df["Perimeter"].max())

    major_axis_min = float(
        df["Major_Axis_Length"].min()
    )

    major_axis_max = float(
        df["Major_Axis_Length"].max()
    )

    area_min = float(df["Area"].min())
    area_max = float(df["Area"].max())

    convex_min = float(
        df["Convex_Area"].min()
    )

    convex_max = float(
        df["Convex_Area"].max()
    )

    eccentricity_min = float(
        df["Eccentricity"].min()
    )

    eccentricity_max = float(
        df["Eccentricity"].max()
    )

    minor_axis_min = float(
        df["Minor_Axis_Length"].min()
    )

    minor_axis_max = float(
        df["Minor_Axis_Length"].max()
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    perimeter = st.number_input(
        "Perimeter",
        min_value=perimeter_min,
        max_value=perimeter_max,
        value=float(df["Perimeter"].median())
    )

    major_axis = st.number_input(
        "Major Axis Length",
        min_value=major_axis_min,
        max_value=major_axis_max,
        value=float(df["Major_Axis_Length"].median())
    )

    area = st.number_input(
        "Area",
        min_value=area_min,
        max_value=area_max,
        value=float(df["Area"].median())
    )

    convex_area = st.number_input(
        "Convex Area",
        min_value=convex_min,
        max_value=convex_max,
        value=float(df["Convex_Area"].median())
    )

    eccentricity = st.number_input(
        "Eccentricity",
        min_value=eccentricity_min,
        max_value=eccentricity_max,
        value=float(df["Eccentricity"].median())
    )

    minor_axis = st.number_input(
        "Minor Axis Length",
        min_value=minor_axis_min,
        max_value=minor_axis_max,
        value=float(df["Minor_Axis_Length"].median())
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button("🔍 Predict Rice Variety"):

        input_data = pd.DataFrame(
            {
                "Perimeter": [perimeter],
                "Major_Axis_Length": [major_axis],
                "Area": [area],
                "Convex_Area": [convex_area],
                "Eccentricity": [eccentricity],
                "Minor_Axis_Length": [minor_axis]
            }
        )

        prediction = model.predict(
            input_data
        )

        predicted_class = encoder.inverse_transform(
            prediction
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        confidence = max(probabilities) * 100


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success(
            f"🌾 Predicted Rice Variety: **{predicted_class}**"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Rice Variety": encoder.classes_,
                "Probability": probabilities * 100
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Rice Variety"
            )
        )