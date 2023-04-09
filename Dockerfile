FROM python

RUN apt update
RUN apt install build-essential
RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade cython
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED 1
COPY . .
EXPOSE 5000
CMD python PythonApplication3.py