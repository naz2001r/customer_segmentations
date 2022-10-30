# build stage
FROM python:3.8-slim-buster AS build

# Create the virtual environment.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r ./requirements.txt

# Final stage:
FROM python:3.8-slim-buster

# Copy the virtual environment from the first stage.
COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]