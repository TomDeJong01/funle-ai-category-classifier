FROM python:3.9

WORKDIR /app
COPY . .
VOLUME /app/ml_models
RUN python -m pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]
#CMD ["param1"]