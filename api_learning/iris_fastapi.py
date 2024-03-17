import pickle
from datetime import datetime

from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from sklearn import datasets
import csv

app = FastAPI()

classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)


# Function to save user input to CSV file
def save_to_csv(data: dict):
    filename = "data/database.csv"
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        # Check if the file is empty and write header if needed
        if file.tell() == 0:
            writer.writeheader()

        # Write user input and timestamp to CSV
        writer.writerow(data)


@app.post("/iris_v1/")
async def iris_inference_v1(
    sepal_length: float, sepal_width: float, petal_length: float, petal_width: float,
    background_tasks: BackgroundTasks
):
    """Version 1 of the iris inference endpoint."""
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Prepare data to be saved
    data_to_save = {
        "time": timestamp,
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
        "prediction": prediction
    }

    # Add the background task to save data to CSV
    background_tasks.add_task(save_to_csv, data_to_save)

    return {"prediction": classes[prediction], "prediction_int": prediction}


with open("prediction_database.csv", "w") as file:
    file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")


def add_to_database(
    now: str,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    prediction: int,
):
    """Simple function to add prediction to database."""
    with open("prediction_database.csv", "a") as file:
        file.write(f"{now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n")


@app.post("/iris_v2/")
async def iris_inference_v2(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
):
    """Version 2 of the iris inference endpoint."""
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()

    now = str(datetime.now())
    background_tasks.add_task(
        add_to_database,
        now,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        prediction,
    )

    return {"prediction": classes[prediction], "prediction_int": prediction}


@app.get("/iris_monitoring/", response_class=HTMLResponse)
async def iris_monitoring():
    """Simple get request method that returns a monitoring report."""
    iris_frame = datasets.load_iris(as_frame=True).frame

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    data_drift_report.run(
        current_data=iris_frame.iloc[:60],
        reference_data=iris_frame.iloc[60:],
        column_mapping=None,
    )
    data_drift_report.save_html("monitoring.html")

    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)


# uvicorn --reload --port 8000 api_learning.iris_fastapi:app

"""
curl -X 'POST' \
    'http://127.0.0.1:8000/iris_v1/?sepal_length=1.0&sepal_width=1.0&petal_length=1.0&petal_width=1.0' \
    -H 'accept: application/json' \
    -d ''

"""
