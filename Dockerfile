FROM python:3.6

ENV LANG C.UTF-8

RUN pip install --upgrade pip virtualenv

RUN virtualenv /venv
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH
RUN mkdir -p /app
WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
ADD . /app
RUN python setup.py
