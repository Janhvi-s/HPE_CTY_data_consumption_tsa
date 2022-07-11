import pandas as pd
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