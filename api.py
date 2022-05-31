# Importing necessary modules
import uvicorn
from fastapi import FastAPI

from models import RequestBody
from utils import load_model

# Declaring our FastAPI instance
app = FastAPI()

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'Server Health: OK'}

@app.post('/predict')
def predict(data : RequestBody):
    n_output_days = 7
    data = data.input_data
    model = load_model(r'D:\HPE_CTY\api\MLP')
    for i in range(n_output_days):
        output = model.predict([data[i:]]).item(0)
        data.append(output)
    return {'output': data}