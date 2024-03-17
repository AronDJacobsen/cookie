from fastapi import FastAPI
from http import HTTPStatus

app = FastAPI()

# strating, get methods
@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# for specifying the type of the parameter
from enum import Enum
class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}
"""
curl -X 'GET' \
  'http://127.0.0.1:8000/restric_items/alexnet' \
  -H 'accept: application/json'
"""

# for query parameters, i.e. optional parameters

@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}

"""
curl -X 'GET' \
  'http://localhost:8000/query_items?item_id=42' \
  -H 'accept: application/json'

"""

# for storing data in the body of the request
database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


# for different inputs
from typing import Optional
#from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re

class EmailDomain(str, Enum):
    gmail = "gmail"
    hotmail = "hotmail"

class EmailRequest(BaseModel):
    email: str
    domain_match: EmailDomain

class EmailResponse(BaseModel):
    email: str
    domain_match: str
    is_email_valid: bool
    is_domain_match: bool

@app.post("/check_email/", response_model=EmailResponse)
def check_email(request: EmailRequest):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Check if the email matches the regex pattern
    is_email_valid = re.fullmatch(regex, request.email) is not None

    # Check if the domain matches the specified domain in the request
    is_domain_match = request.email.endswith(f"@{request.domain_match.value}")

    response = EmailResponse(
        email=request.email,
        domain_match=request.domain_match.value,
        is_email_valid=is_email_valid,
        is_domain_match=is_domain_match
    )

    return response

"""
curl -X 'POST' \
  'http://localhost:8000/check_email/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "mlops@gmail.com",
    "domain_match": "gmail"
  }'
"""

# file input
from fastapi import UploadFile, File
import cv2
from fastapi.responses import FileResponse


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
    # Save the uploaded file to the local filesystem
    with open('./api_learning/image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)

    # Read the saved image using OpenCV
    img = cv2.imread("./api_learning/image.jpg")

    # Resize the image using the specified dimensions (default: 28x28)
    res = cv2.resize(img, (h, w))

    # Save the resized image
    cv2.imwrite('./api_learning/image_resize.jpg', res)

    # Return the resized image as a FileResponse
    return FileResponse('./api_learning/image_resize.jpg', media_type='image/jpeg', filename='image_resize.jpg')



# uvicorn --reload --port 8000 api_learning.main:app
