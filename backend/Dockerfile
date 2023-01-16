# docker build -t myimage .
# docker run -p 8000:8000 myimage
# http://localhost:8000 use this

#FROM python:3.9-slim-buster giving error
FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
# this is for the import to work
ENV PYTHONPATH "${PYTHONPATH}:/code/app" 

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]






