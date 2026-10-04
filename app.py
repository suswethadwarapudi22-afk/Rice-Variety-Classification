import streamlit as st
import pandas as pd
import joblib
from scipy.io import arff


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rice Variety Classification",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/Rice_Cammeo_Osmancik.arff"
MODEL_PATH = "models/rice_classifier.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

FEATURES = [
    "Perimeter",
    "Major_Axis_Length",
    "Area",
    "Convex_Area",
    "Eccentricity",
    "Minor_Axis_Length"
]


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    data, meta = arff.loadarff(DATA_PATH)

    df = pd.DataFrame(data)

    # Convert byte strings to normal strings
    if df["Class"].dtype == object:
        df["Class"] = df["Class"].apply(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes)
            else x
        )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    df = load_dataset()

    model, encoder = load_model()

except Exception as e:

    st.error("❌ Unable to load the application files.")

    st.code(str(e))

    st.info(
        "Make sure the following files exist in your GitHub repository:\n\n"
        "data/Rice_Cammeo_Osmancik.arff\n"
        "models/rice_classifier.joblib\n"
        "models/label_encoder.joblib"
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🌾 Rice Variety Classification")

st.markdown(
    """
    ### Machine Learning Based Rice Variety Prediction

    This application predicts whether a rice grain belongs to
    **Cammeo** or **Osmancik** using physical measurements of the grain.
    """
)

st.info(
    "The final model uses six selected physical features "
    "and Logistic Regression."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Model Information")

st.sidebar.markdown(
    """
    **Algorithm:** Logistic Regression

    **Number of Features:** 6

    **C:** 10

    **Solver:** liblinear

    **Test Accuracy:** 91.60%

    **Test F1 Score:** 91.58%
    """
)

st.sidebar.divider()

st.sidebar.subheader("Selected Features")

for feature in FEATURES:
    st.sidebar.write(f"• {feature}")

st.sidebar.divider()

st.sidebar.write(
    f"📚 Dataset samples: **{len(df)}**"
)

st.sidebar.write(
    "🌾 Classes: **Cammeo, Osmancik**"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("📚 Dataset Information"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Samples",
            len(df)
        )

    with col2:
        st.metric(
            "Features",
            7
        )

    with col3:
        st.metric(
            "Classes",
            df["Class"].nunique()
        )

    st.write("### Class Distribution")

    class_counts = df["Class"].value_counts()

    st.dataframe(
        class_counts.rename("Number of Samples"),
        use_container_width=True
    )


# ============================================================
# INPUT METHOD
# ============================================================

st.header("🔍 Rice Variety Prediction")

input_method = st.radio(
    "Choose an input method:",
    [
        "Use a real dataset sample",
        "Enter measurements manually"
    ],
    horizontal=True
)


# ============================================================
# REAL DATASET SAMPLE
# ============================================================

if input_method == "Use a real dataset sample":

    st.subheader("🌾 Predict a Real Dataset Sample")

    sample_number = st.number_input(
        "Select sample number",
        min_value=1,
        max_value=len(df),
        value=1,
        step=1
    )

    sample = df.iloc[int(sample_number) - 1]

    st.write("### Input Measurements")

    sample_input = pd.DataFrame(
        {
            feature: [float(sample[feature])]
            for feature in FEATURES
        }
    )

    st.dataframe(
        sample_input,
        use_container_width=True
    )

    actual_class = sample["Class"]

    st.write(
        f"**Actual Dataset Class:** `{actual_class}`"
    )

    if st.button(
        "🔍 Predict Selected Sample",
        type="primary"
    ):

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

        st.divider()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if predicted_class == actual_class:

            st.success(
                f"✅ Correct Prediction: **{predicted_class}**"
            )

        else:

            st.error(
                f"❌ Incorrect Prediction: **{predicted_class}**"
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Actual Class",
                actual_class
            )

        with col2:

            st.metric(
                "Predicted Class",
                predicted_class
            )

        with col3:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader("📊 Prediction Probabilities")

        probability_df = pd.DataFrame(
            {
                "Rice Variety": encoder.classes_,
                "Probability (%)": probabilities * 100
            }
        )

        probability_df["Probability (%)"] = (
            probability_df["Probability (%)"]
            .round(2)
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

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
        "Enter values within the range of the original dataset."
    )

    # ========================================================
    # GET DATASET RANGES
    # ========================================================

    perimeter_min = float(
        df["Perimeter"].min()
    )

    perimeter_max = float(
        df["Perimeter"].max()
    )

    perimeter_median = float(
        df["Perimeter"].median()
    )


    major_axis_min = float(
        df["Major_Axis_Length"].min()
    )

    major_axis_max = float(
        df["Major_Axis_Length"].max()
    )

    major_axis_median = float(
        df["Major_Axis_Length"].median()
    )


    area_min = float(
        df["Area"].min()
    )

    area_max = float(
        df["Area"].max()
    )

    area_median = float(
        df["Area"].median()
    )


    convex_area_min = float(
        df["Convex_Area"].min()
    )

    convex_area_max = float(
        df["Convex_Area"].max()
    )

    convex_area_median = float(
        df["Convex_Area"].median()
    )


    eccentricity_min = float(
        df["Eccentricity"].min()
    )

    eccentricity_max = float(
        df["Eccentricity"].max()
    )

    eccentricity_median = float(
        df["Eccentricity"].median()
    )


    minor_axis_min = float(
        df["Minor_Axis_Length"].min()
    )

    minor_axis_max = float(
        df["Minor_Axis_Length"].max()
    )

    minor_axis_median = float(
        df["Minor_Axis_Length"].median()
    )


    # ========================================================
    # INPUT COLUMNS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        perimeter = st.number_input(
            "Perimeter",
            min_value=perimeter_min,
            max_value=perimeter_max,
            value=perimeter_median,
            step=0.01
        )

        major_axis = st.number_input(
            "Major Axis Length",
            min_value=major_axis_min,
            max_value=major_axis_max,
            value=major_axis_median,
            step=0.01
        )

        area = st.number_input(
            "Area",
            min_value=area_min,
            max_value=area_max,
            value=area_median,
            step=1.0
        )


    with col2:

        convex_area = st.number_input(
            "Convex Area",
            min_value=convex_area_min,
            max_value=convex_area_max,
            value=convex_area_median,
            step=1.0
        )

        eccentricity = st.number_input(
            "Eccentricity",
            min_value=eccentricity_min,
            max_value=eccentricity_max,
            value=eccentricity_median,
            step=0.0001,
            format="%.4f"
        )

        minor_axis = st.number_input(
            "Minor Axis Length",
            min_value=minor_axis_min,
            max_value=minor_axis_max,
            value=minor_axis_median,
            step=0.01
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if st.button(
        "🔍 Predict Rice Variety",
        type="primary"
    ):

        input_data = pd.DataFrame(
            {
                "Perimeter": [perimeter],

                "Major_Axis_Length": [
                    major_axis
                ],

                "Area": [area],

                "Convex_Area": [
                    convex_area
                ],

                "Eccentricity": [
                    eccentricity
                ],

                "Minor_Axis_Length": [
                    minor_axis
                ]
            }
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )

        predicted_class = encoder.inverse_transform(
            prediction
        )[0]


        probabilities = model.predict_proba(
            input_data
        )[0]


        confidence = (
            max(probabilities) * 100
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.success(
            f"🌾 Predicted Rice Variety: "
            f"**{predicted_class}**"
        )


        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Probabilities"
        )


        probability_df = pd.DataFrame(
            {
                "Rice Variety": encoder.classes_,

                "Probability (%)": (
                    probabilities * 100
                )
            }
        )


        probability_df[
            "Probability (%)"
        ] = probability_df[
            "Probability (%)"
        ].round(2)


        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


        st.bar_chart(
            probability_df.set_index(
                "Rice Variety"
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Rice Variety Classification using Lightweight Machine Learning "
    "| Logistic Regression"
)