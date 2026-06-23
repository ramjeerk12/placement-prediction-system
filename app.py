from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

class Student(BaseModel):
    cgpa: float
    iq: float

@app.get("/")
def home():
    return {"message": "Placement Predictor API Running"}

@app.post("/predict")
def predict(data: Student):

    x = np.array([[data.cgpa, data.iq]])

    x = scaler.transform(x)

    result = model.predict(x)

    return {"placement": int(result[0])}