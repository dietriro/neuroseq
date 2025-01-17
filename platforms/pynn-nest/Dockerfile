FROM ubuntu:22.04

## LABEL about the custom image
LABEL maintainer="robin.dietrich@tum.de"
LABEL version="0.1"
LABEL description="Dockerized PyNN-NEST installation and simulation for 'neuroseq' package."

ARG PATH_CODE=/home/code
ARG BRANCH_CODE
ARG BRANCH_DATA

## Update Ubuntu Software repository
RUN apt update && apt install make git -y

## Install Conda from docker
COPY --from=continuumio/miniconda3 /opt/conda /opt/conda

# Add conda to Path
ENV PATH="/opt/conda/bin:$PATH"
# Add neuroseq package to Python Path
ENV PYTHONPATH="/home/code/neuroseq:$PYTHONPATH"

# Configure conda
RUN set -ex && \
    conda config --set always_yes yes --set changeps1 no && \
    conda info -a && \
    conda config --add channels conda-forge

# ToDo: Remove this for publication
## Copy code from local repository
RUN mkdir ${PATH_CODE}
WORKDIR ${PATH_CODE}
RUN git clone --depth 1 --branch ${BRANCH_CODE} https://github.com/dietriro/neuroseq.git
WORKDIR ${PATH_CODE}/neuroseq
RUN git submodule set-url data/evaluation https://github.com/dietriro/neuroseq-evaluation.git
RUN git submodule update --init
WORKDIR ${PATH_CODE}/neuroseq/data/evaluation
RUN git checkout ${BRANCH_DATA}

## Configure conda workspace
# Create a conda environment (with boost-cpp and gxx being optional, only necessary for e.g. a bare ubuntu docker)
RUN conda env create -f /home/code/neuroseq/platforms/pynn-nest/environment.yaml
# Source new environment
SHELL ["conda", "run", "-n", "neuroseq", "/bin/bash", "-c"]
# Add environment activation to bashrc
RUN echo "source activate neuroseq" >> ~/.bashrc

# Install neuron model
WORKDIR ${PATH_CODE}/neuroseq/models/mc
RUN python ${PATH_CODE}/neuroseq/scripts/install_mc_neuron.py

WORKDIR ${PATH_CODE}/neuroseq