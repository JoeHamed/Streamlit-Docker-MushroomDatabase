FROM python:3.11

WORKDIR /code

COPY /app/requirements.txt /code/requirements.txt

COPY ./data/mushrooms.csv /code/data/mushrooms.csv

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 9000

CMD ["streamlit", "run", "app/app.py", "--server.address", "0.0.0.0", "--server.port", "9000"]
