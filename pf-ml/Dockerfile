FROM python:3.8.3

WORKDIR /app

COPY . .

RUN ["ls", "-a"]

RUN ["cd"]
# RUN apt-get update && \
#     apt-get install -y \
#     python3 python3-numpy python3-nose python3-pandas python-h5py \
#     python python-numpy python-nose python-pandas python3-h5py \
#     pep8 python-pip python3-pip python-wheel \
#     python-sphinx && \
#     pip install --upgrade setuptools

# RUN ["pip", "install", "-r", "requirements.txt"]

# RUN ["python", "./src/stanzaSpanish.py"]

# EXPOSE 8000

# CMD ["gunicorn", "-w 4", "-b", "0.0.0.0:8000", "main:app"]