FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

#Could use for NAS in future: (will be hardcoded for now)
#ENV HF_HOME=/runpod-volume


# RUN copied from https://github.com/runpod-workers/worker-infinity-embedding
# install python and other packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip



WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]