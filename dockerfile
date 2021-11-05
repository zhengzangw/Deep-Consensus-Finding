FROM continuumio/miniconda3:4.10.3
RUN conda config --set always_yes yes --set changeps1 no
RUN conda install python=3.8
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app
