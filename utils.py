from audioop import avg
import pandas as pd
import numpy as np
from tensorflow import keras
from pmdarima.arima import auto_arima
from pathlib import Path


def load_model(model_path):
    """Function to load a model.
    Args-
        model_path: String denoting model path
    Returns-
        model: Pre-trained model
    """
    model = keras.models.load_model(model_path)
    return model


def check_csv(file_name):
    """Function to check if the uploaded file is a csv file or not
    Args-
        file_name: String denoting file type
    Returns-
        is_csv: Boolean value denoting whether the file is csv or not
    """
    is_csv = Path(file_name).suffix == ".csv"
    return is_csv


def train_arima(file):
    """Function to train arima model.
    Args-
        file: CSV file the model is to be trained on
    Returns-
        model: Trained ARIMA model
    """
    df = pd.read_csv(file)
    model = auto_arima(df.daily_usage, seasonal=False)
    return model


def generate_arima_prediction(arima_model):
    """Function to generate predictions using arima model.
    Args-
        file: CSV file the model is to be trained on
    Returns-
        model: Trained ARIMA model
    """
    predictions = arima_model.predict(n_periods=7).tolist()
    return predictions

def get_user_type(data_list):
    FREE_USER = 0  # 0 to 15
    SMALL_USER = 1  # 16 to 50
    MID_TIER_USER = 2  # 51 to 100
    ENTERPRISE_USER = 3  # 100 and above
    avg_consumption = np.average(data_list)
    if avg_consumption >= 100:
        return ENTERPRISE_USER
    elif avg_consumption >= 51 and avg_consumption < 100:
        return MID_TIER_USER
    elif avg_consumption >= 16 and avg_consumption < 50:
        return SMALL_USER
    else:
        return FREE_USER