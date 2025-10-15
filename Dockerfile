FROM python:3.12

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]