FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /content

# Install dependencies google-cloud-bigquery[pandas]==2.34.3
RUN pip install joblib pandas google-cloud-aiplatform google-cloud-storage google-cloud-bigquery google-genai google-cloud-bigquery[pandas]==2.34.3

# Copy the correct training script and config
COPY gru4rec_train.py /content/gru4rec_train.py
COPY gru4rec_config.json /content/gru4rec_config.json
COPY GRU4Rec_PyTorch_Official /content/GRU4Rec_PyTorch_Official

ENTRYPOINT ["python", "gru4rec_train.py"]