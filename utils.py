from tensorflow import keras

def load_model(model_path):
    """Function to load a model.
    Args-
        model_path: String denoting model path
    Returns-
        model: Pre-trained model
    """
    model = keras.models.load_model(model_path)
    return model