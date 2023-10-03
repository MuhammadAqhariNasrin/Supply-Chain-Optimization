# 
FROM python:3.9

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

# Install FastAPI
RUN pip install fastapi


EXPOSE 8000

CMD ["uvicorn", "app3:app", "--host", "0.0.0.0", "--port", "8000"]
