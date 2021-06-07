#FROM tensorflow/tensorflow:latest 
FROM python:3.8

RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app

RUN apt-get update -y && apt-get update \
  && apt-get install -y --no-install-recommends curl gcc g++ gnupg unixodbc-dev libssl-dev python3-dev python3-venv python3-pip python3-setuptools libzbar-dev libc-dev unixodbc-dev unixodbc libpq-dev libsasl2-dev gcc python-dev sasl2-bin libsasl2-2 libsasl2-dev libsasl2-modules

#RUN pip3 install setuptools-scm && pip3 install --upgrade setuptools
#RUN apt-get update && apt-get install -y libssl-dev

#COPY requirements.txt .
COPY requirements-fraxses.txt .
COPY python_fraxses_wrapper-0.4.0-py3-none-any.whl .

#RUN pip install -r requirements.txt
RUN pip install -r requirements-fraxses.txt
RUN pip install python_fraxses_wrapper-0.4.0-py3-none-any.whl

COPY app/ /usr/src/app

ENTRYPOINT ["python", "/usr/src/app/app.py"]
