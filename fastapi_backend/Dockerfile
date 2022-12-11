FROM python:3.8-slim-buster

COPY ./ ./
#COPY ./requirements.txt /requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r ./requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]