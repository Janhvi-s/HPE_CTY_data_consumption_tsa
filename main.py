# Importing necessary modules
import os
from h11 import Data
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Importing util function
from models import RequestBody
from utils import get_user_type, load_model, train_arima, generate_arima_prediction, check_csv

# Declaring our FastAPI instance
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defining path operation for root endpoint
@app.get('/')
async def main():
	return {'message': 'Server Health: OK'}

# Endpoint for pretrained models
@app.post('/predict')
async def predict(data : RequestBody, model: str = "MLP"):
    n_output_days = 7
    data = data.input_data
    
    if model == "MLP":
        path = os.path.join(os.getcwd(), "MLP")
    elif model == "LSTM":
        path = os.path.join(os.getcwd(), "LSTM")
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    model = load_model(path)

    for i in range(n_output_days):
        output = model.predict([data[i:]]).item(0)
        data.append(output)
    result = dict()
    result["forecast"] = data
    result["user_type"] = get_user_type(data)
    
    return {'output': result}


# Endpoint for train-and-predict models
@app.post('/predict/upload')
def predict_from_file(file: UploadFile, model: str = "ARIMA"):
    if not check_csv(file.filename):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    n_output_days = 7
    
    if model == "ARIMA":
        arima_model = train_arima(file.file)
        data = generate_arima_prediction(arima_model)
    elif model == "PROPHET":
        # Hardcoded output value. Model not implemented yet
        data = [1, 2, 3, 4, 5, 6, 7]
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    return {'output': data}