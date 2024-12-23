FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /mle_course/01_containerization

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    git wget unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

RUN useradd -ms /bin/bash devuser && chown -R devuser:devuser .
USER devuser

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]