# Neuromorphic Sequential Learning with Spiking Neural Networks

This repository contains code for training a spiking neural network (SNN) on sequential data as well as replaying already learned sequences for e.g. planning purposes.


## 1 Installation

### 1.1 Installation (NEST Simulation)

Please make sure that the 'make' package is installed in your system.

1. Download and prepare repository

```bash
# Clone repository into new folder
git clone 
# Make script executable
chmod +x conda.sh
# Install anaconda (optionally providing a custom prefix '-p /opt/conda')
./conda.sh -b -p /opt/conda
```


2. Install conda

```bash
# Download latest anaconda version
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O conda.sh
# Make script executable
chmod +x conda.sh
# Install anaconda (optionally providing a custom prefix '-p /opt/conda')
./conda.sh -b -p /opt/conda
```


3. Setup conda environment

```bash
# Create a conda environment (with boost-cpp and gxx being optional, only necessary for e.g. a bare ubuntu docker)
conda create --name shtm-pynn -c conda-forge nest-simulator=3.6.0 boost-cpp gxx_linux-64
# Source new environment
conda activate shtm-pynn
# Add shtm-pynn package to Python Path
export PYTHONPATH=/path/to/repository:$PYTHONPATH
# Alternatively, install the package
cd /path/to/repository
pip install .
```

4. Install custom neuron

```bash
# Make sure to source conda environment
conda activate shtm-pynn
# Make sure that the repository is available on the Python Path
export PYTHONPATH=/path/to/repository:$PYTHONPATH
# Install neuron into current nest installation
python test/install_mc_neuron.py
```

5. (Optional) Install custom PyNN-Nest version to remove "initial values" messages.

```bash
# Make sure to source conda environment
conda activate shtm-pynn
# Download branch fixes
git clone -b fixes git@github.com:dietriro/PyNN.git
# Change directory into new repository
cd PyNN
# Install custom PyNN package
pip install .
```


## To-Do's

- [ ] Add instructions for BSS-2 installation/use
- [ ] Add additional folders to installation (setup.py): config, data
