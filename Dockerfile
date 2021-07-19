FROM rayproject/ray-ml:latest-cpu

# USER $RAY_UID

RUN sudo mkdir -p /usr/src/app

# RUN chown -R $RAY_UID /home/ray/
RUN sudo chmod 777 /home/ray

WORKDIR /usr/src/app

RUN sudo apt-get update -y && sudo apt-get update \
  && sudo apt-get install -y --no-install-recommends curl gnupg unixodbc-dev libssl-dev python3-dev python3-venv python3-pip python3-setuptools libzbar-dev libc-dev unixodbc-dev unixodbc libpq-dev libsasl2-dev gcc python-dev sasl2-bin libsasl2-2 libsasl2-dev libsasl2-modules

COPY requirements-fraxses.txt .
COPY python_fraxses_wrapper-0.4.0-py3-none-any.whl .

RUN pip install -r requirements-fraxses.txt
RUN pip install python_fraxses_wrapper-0.4.0-py3-none-any.whl

COPY app/ /usr/src/app

ENTRYPOINT ["python", "/usr/src/app/app.py"]
