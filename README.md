# Predicting Cloud Storage Consumption using Time Series Analysis
---

In this end-to-end project developed under the HPE-CTY program, we have developed an ML-based web application that uses different machine learning architectures to predict a user's future cloud storage consumption based on their past usage history. 

These predictions can be leveraged for tasks such as predicting users' monthly cloud service consumption billings.

## Steps to Run The Server Locally
---

1. Create a Python virtual environment:
```py
python -m venv venv
```
2. Activate the virtual environment:
```powershell
& <path_to_project_directory>/venv/Scripts/Activate.ps1
```
Eg.-  `& d:/HPE_CTY/api/venv/Scripts/Activate.ps1`

3. Install the required project dependencies:
```py
pip install -r requirements.txt
```
4. Running the server:
```py
uvicorn main:app --reload
```
5. Go to the API docs:
```
Once the server is up and running, you can head over to http://127.0.0.1:8000/docs to access the API docs and test out the endpoints using fast-API's swagger documentation interface.