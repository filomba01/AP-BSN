# Start with a base Ubuntu image with CUDA and cuDNN pre-installed
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install python3.10 and pip
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
  build-essential ca-certificates python3.10 python3.10-dev python3.10-distutils git vim wget cmake python3-pip
RUN ln -sv /usr/bin/python3.10 /usr/bin/python
RUN ln -svf /usr/bin/python3.10 /usr/bin/python3

# Install system dependencies
RUN apt update && apt install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    python3.10-venv

# Set the working directory
WORKDIR /exp

# Create a new virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

CMD ["bash"]
WORKDIR /exp


# Create and set permissions for system directories
RUN mkdir -p /.cache /.local /.config /.jupyter && \
    chmod -R 777 /.cache && \
    chmod -R 777 /.local && \
    chmod -R 777 /.config && \
    chmod -R 777 /.jupyter

# Copy project files into the container
COPY . /exp/

# install local packages
RUN pip install -e .