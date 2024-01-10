# Base image
FROM python:3.11-slim

# Install system dependencies for building software
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Set the working directory
WORKDIR /

RUN pip install --no-cache-dir --upgrade pip
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .
# Install the package without dependencies
RUN pip install --no-deps --no-cache-dir .

# Set the entry point for your application
ENTRYPOINT ["python", "-u", "cookie/predict_model.py"]
