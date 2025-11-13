# Containerfile
FROM nvcr.io/nvidia/pytorch:25.06-py3
RUN apt-get update && apt-get install -y git vim && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt
CMD ["bash"]