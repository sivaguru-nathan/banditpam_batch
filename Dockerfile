FROM python:3.8
COPY ./requirements.txt /banditpam/
RUN pip install --upgrade pip
RUN pip install -r /banditpam/requirements.txt
ADD . /banditpam
