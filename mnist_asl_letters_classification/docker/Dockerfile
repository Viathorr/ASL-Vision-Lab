FROM python:3.10

WORKDIR /app

RUN pip install --upgrade pip

COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-build-isolation -r /app/requirements.txt 

COPY ../ /app/ 

# Install local package from setup.py
RUN pip install --no-cache-dir --no-build-isolation . 

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]