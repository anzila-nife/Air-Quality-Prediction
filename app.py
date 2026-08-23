import streamlit as st
import pandas as pd
import joblib
from datetime import date, time, datetime
import lzma
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IoT Air Quality Prediction",
    page_icon="🌫️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with lzma.open(
        "air_quality_extra_trees.pkl.xz",
        "rb"
    ) as f:

        package = joblib.load(f)

    return package


package = load_model()

model = package["model"]
features = package["features"]
targets = package["targets"]


# ============================================================
# INPUT TRAINING RANGES
# ============================================================

input_ranges = {
    "CO_Sensor": (647.0, 2040.0),
    "NMHC_Sensor": (383.0, 2214.0),
    "NOx_Sensor": (322.0, 2683.0),
    "NO2_Sensor": (551.0, 2775.0),
    "O3_Sensor": (221.0, 2523.0),

    "Temperature": (-1.9, 44.6),
    "Relative_Humidity": (9.2, 88.7),
    "Absolute_Humidity": (0.1847, 2.231)
}


# ============================================================
# TARGET INFORMATION
# ============================================================

target_info = {

    "CO_Concentration": {
        "unit": "mg/m³",
        "min": 0.1,
        "max": 11.9,
        "normal": 2.6,
        "elevated": 3.9
    },

    "Benzene_Concentration": {
        "unit": "µg/m³",
        "min": 0.1,
        "max": 63.7,
        "normal": 13.6,
        "elevated": 20.1
    },

    "NOx_Concentration": {
        "unit": "ppb",
        "min": 2.0,
        "max": 1479.0,
        "normal": 284.0,
        "elevated": 497.4
    },

    "NO2_Concentration": {
        "unit": "µg/m³",
        "min": 2.0,
        "max": 340.0,
        "normal": 133.0,
        "elevated": 171.0
    }
}


# ============================================================
# STATUS FUNCTION
# ============================================================

def get_status(value, normal_limit, elevated_limit):

    if value <= normal_limit:

        return "🟢 Normal"

    elif value <= elevated_limit:

        return "🟡 Elevated"

    else:

        return "🔴 High"


# ============================================================
# TITLE
# ============================================================

st.title("🌫️ IoT-Based Air Quality Prediction System")

st.write(
    "Machine-learning-based estimation of CO, Benzene, "
    "NOx and NO2 concentrations using IoT sensor and "
    "environmental measurements."
)

st.info(
    "The model uses an Extra Trees Regressor trained on "
    "historical UCI Air Quality data."
)

st.divider()


# ============================================================
# SENSOR INPUTS
# ============================================================

st.header("📡 Sensor & Environmental Inputs")

st.caption(
    "Enter values within the ranges observed during model training."
)


col1, col2 = st.columns(2)


# ============================================================
# SENSOR COLUMN 1
# ============================================================

with col1:

    st.subheader("Gas Sensor Measurements")

    co_min, co_max = input_ranges["CO_Sensor"]

    co_sensor = st.number_input(
        "CO Sensor",
        min_value=co_min,
        max_value=co_max,
        value=1098.0,
        format="%.3f"
    )

    st.caption(
        f"Sensor response | Training range: "
        f"{co_min:.0f} – {co_max:.0f}"
    )


    nmhc_min, nmhc_max = input_ranges["NMHC_Sensor"]

    nmhc_sensor = st.number_input(
        "NMHC Sensor",
        min_value=nmhc_min,
        max_value=nmhc_max,
        value=938.0,
        format="%.3f"
    )

    st.caption(
        f"Sensor response | Training range: "
        f"{nmhc_min:.0f} – {nmhc_max:.0f}"
    )


    nox_min, nox_max = input_ranges["NOx_Sensor"]

    nox_sensor = st.number_input(
        "NOx Sensor",
        min_value=nox_min,
        max_value=nox_max,
        value=834.0,
        format="%.3f"
    )

    st.caption(
        f"Sensor response | Training range: "
        f"{nox_min:.0f} – {nox_max:.0f}"
    )


    no2_min, no2_max = input_ranges["NO2_Sensor"]

    no2_sensor = st.number_input(
        "NO2 Sensor",
        min_value=no2_min,
        max_value=no2_max,
        value=1456.0,
        format="%.3f"
    )

    st.caption(
        f"Sensor response | Training range: "
        f"{no2_min:.0f} – {no2_max:.0f}"
    )


    o3_min, o3_max = input_ranges["O3_Sensor"]

    o3_sensor = st.number_input(
        "O3 Sensor",
        min_value=o3_min,
        max_value=o3_max,
        value=1021.0,
        format="%.3f"
    )

    st.caption(
        f"Sensor response | Training range: "
        f"{o3_min:.0f} – {o3_max:.0f}"
    )


# ============================================================
# ENVIRONMENT COLUMN
# ============================================================

with col2:

    st.subheader("Environmental Measurements")

    temp_min, temp_max = input_ranges["Temperature"]

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=temp_min,
        max_value=temp_max,
        value=18.3,
        format="%.3f"
    )

    st.caption(
        f"Training range: {temp_min} – {temp_max} °C"
    )


    rh_min, rh_max = input_ranges["Relative_Humidity"]

    relative_humidity = st.number_input(
        "Relative Humidity (%)",
        min_value=rh_min,
        max_value=rh_max,
        value=49.2,
        format="%.3f"
    )

    st.caption(
        f"Training range: {rh_min} – {rh_max} %"
    )


    ah_min, ah_max = input_ranges["Absolute_Humidity"]

    absolute_humidity = st.number_input(
        "Absolute Humidity",
        min_value=ah_min,
        max_value=ah_max,
        value=1.024,
        format="%.4f"
    )

    st.caption(
        f"Training range: {ah_min} – {ah_max}"
    )


# ============================================================
# DATE AND TIME
# ============================================================

st.divider()

st.header("📅 Date & Time")

st.warning(
    "This model was trained using historical observations "
    "from 2004–2005. Therefore, use a date within the "
    "model's training period for this demonstration."
)


date_col, time_col = st.columns(2)


with date_col:

    selected_date = st.date_input(
        "Date",
        value=date(2005, 1, 1),
        min_value=date(2004, 1, 1),
        max_value=date(2005, 12, 31)
    )


with time_col:

    selected_time = st.time_input(
        "Time",
        value=time(12, 0)
    )


# ============================================================
# DERIVE TIME FEATURES
# ============================================================

selected_datetime = datetime.combine(
    selected_date,
    selected_time
)

year = selected_datetime.year
month = selected_datetime.month
day = selected_datetime.day
hour = selected_datetime.hour

# Monday = 0, Sunday = 6
weekday_number = selected_datetime.weekday()


# ============================================================
# SHOW DERIVED FEATURES
# ============================================================

with st.expander("View model time features"):

    time_features_df = pd.DataFrame({
        "Feature": [
            "Year",
            "Month",
            "Day",
            "Hour",
            "Weekday_Number"
        ],

        "Value": [
            year,
            month,
            day,
            hour,
            weekday_number
        ]
    })

    st.dataframe(
        time_features_df,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Air Quality",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    input_data = {

        "CO_Sensor": co_sensor,

        "NMHC_Sensor": nmhc_sensor,

        "NOx_Sensor": nox_sensor,

        "NO2_Sensor": no2_sensor,

        "O3_Sensor": o3_sensor,

        "Temperature": temperature,

        "Relative_Humidity": relative_humidity,

        "Absolute_Humidity": absolute_humidity,

        "Year": year,

        "Month": month,

        "Day": day,

        "Hour": hour,

        "Weekday_Number": weekday_number
    }


    input_df = pd.DataFrame(
        [input_data]
    )


    # --------------------------------------------------------
    # ENSURE EXACT FEATURE ORDER
    # --------------------------------------------------------

    input_df = input_df[
        features
    ]


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_df
    )[0]


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.divider()

    st.header("📊 Predicted Air Quality")


    result_columns = st.columns(4)


    for i, target in enumerate(targets):

        value = float(prediction[i])

        info = target_info[target]

        status = get_status(
            value,
            info["normal"],
            info["elevated"]
        )


        with result_columns[i]:

            st.metric(
                target.replace(
                    "_Concentration",
                    ""
                ),
                f"{value:.3f} {info['unit']}"
            )

            st.write(status)

            st.caption(
                f"Normal: ≤ {info['normal']} {info['unit']}"
                    )

            st.caption(
                    f"Elevated: {info['normal']} – "
                    f"{info['elevated']} {info['unit']}"
                    )

            st.caption(
                f"High: > {info['elevated']} {info['unit']}"
                )

            st.caption(
                f"Training range: {info['min']} – "
                f"{info['max']} {info['unit']}"
                )


    # ========================================================
    # OVERALL STATUS
    # ========================================================

    statuses = []

    for i, target in enumerate(targets):

        value = float(prediction[i])

        info = target_info[target]

        if value > info["elevated"]:

            statuses.append("High")

        elif value > info["normal"]:

            statuses.append("Elevated")

        else:

            statuses.append("Normal")


    if "High" in statuses:

        overall_status = "🔴 HIGH"

        message = (
            "One or more predicted pollutant "
            "concentrations are high relative to "
            "the training-data distribution."
        )

        st.error(
            f"Overall indication: {overall_status}\n\n"
            f"{message}"
        )


    elif "Elevated" in statuses:

        overall_status = "🟡 ELEVATED"

        message = (
            "One or more predicted pollutant "
            "concentrations are elevated relative "
            "to the training-data distribution."
        )

        st.warning(
            f"Overall indication: {overall_status}\n\n"
            f"{message}"
        )


    else:

        overall_status = "🟢 NORMAL"

        message = (
            "All predicted pollutant concentrations "
            "are within the normal range defined "
            "from the training-data distribution."
        )

        st.success(
            f"Overall indication: {overall_status}\n\n"
            f"{message}"
        )


    # ========================================================
    # MOST CONCERNING POLLUTANT
    # ========================================================

    severity_score = []

    for i, target in enumerate(targets):

        value = float(prediction[i])

        info = target_info[target]

        if value > info["elevated"]:

            severity = 2

        elif value > info["normal"]:

            severity = 1

        else:

            severity = 0

        severity_score.append(
            (severity, target, value)
        )


    severity_score.sort(
        reverse=True
    )


    highest_severity = severity_score[0]


    if highest_severity[0] > 0:

        pollutant_name = (
            highest_severity[1]
            .replace("_Concentration", "")
        )

        st.info(
            f"Primary area of concern: **{pollutant_name}**"
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ Status classification is based on the "
        "distribution of the model's training data "
        "(75th and 90th percentile thresholds). "
        "It is intended for project demonstration "
        "and is not an official CPCB AQI calculation "
        "or a health assessment."
    )