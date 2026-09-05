# IoT-Based Air Quality Prediction System

## Project Overview

This project is an IoT-based Air Quality Prediction System that uses machine learning to predict air pollutant concentrations from sensor and environmental measurements.

The application is developed using Python and Streamlit and uses an Extra Trees Regressor model for prediction.

## GitHub Repository

GitHub Repository:
https://github.com/khdanzi/Air-Quality-Prediction

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Extra Trees Regressor

## Machine Learning Model

The project uses an Extra Trees Regressor trained on air-quality sensor data.

### Input Features

The model uses sensor and environmental parameters such as:

- CO Sensor
- NMHC Sensor
- NOx Sensor
- NO2 Sensor
- O3 Sensor
- Temperature
- Relative Humidity
- Absolute Humidity

### Predicted Parameters

The application predicts:

- CO Concentration
- Benzene Concentration
- NOx Concentration
- NO2 Concentration

## System Prerequisites

Before running the application, install:

- Python 3.10 or later
- pip
- Git
- A modern web browser

## Project Files

```text
Air-Quality-Prediction/
├── app.py
├── app copy.py
├── requirements.txt
├── air_quality_extra_trees.pkl.xz
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/khdanzi/Air-Quality-Prediction.git
cd Air-Quality-Prediction
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Web Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After starting the application, open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

## Test Inputs

The application can be tested by entering sensor and environmental values in the Streamlit interface.

Example test input:

```text
CO Sensor: 1200
NMHC Sensor: 1000
NOx Sensor: 800
NO2 Sensor: 1000
O3 Sensor: 900
Temperature: 25
Relative Humidity: 55
Absolute Humidity: 1.0
```

These values are sample inputs for testing the application and are not intended to represent official air-quality measurements.

## Application Output

After submitting the input values, the application provides predicted pollutant concentrations and an indication of the corresponding pollution level.

The system provides predictions for:

- CO
- Benzene
- NOx
- NO2

## Credentials

No external credentials or API keys are required to run the application locally.

## Important Note

This project is intended for educational and demonstration purposes. The model predictions should not be treated as a replacement for certified air-quality monitoring equipment or official regulatory measurements.

## Future Improvements

Possible future improvements include:

- Real-time IoT sensor integration
- Live air-quality monitoring
- Historical prediction charts
- Additional machine-learning models
- Cloud deployment
- Database integration
- Automated high-pollution alerts

## License

This project is intended for educational and academic purposes.
