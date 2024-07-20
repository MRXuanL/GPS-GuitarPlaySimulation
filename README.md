# GPS-GuitarPlaySimulation

A guitar play simulation system capable of reading tablature and, after training, guiding robotic hands to play the guitar.
Our system works best with Linux OS because JAX has some bugs on Windows OS. For example, when training on Windows OS, it may not use the GPU for training. Implementing the DroQ algorithm using PyTorch could potentially solve this issue.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/mrxuanl/GPS-GuitarPlaySimulation.git
    ```
2. Navigate to the project directory:
    ```sh
    cd GPS-GuitarPlaySimulation
    ```
3. Install miniconda3:
    ```sh
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh
    ```
4.  Install dependencies：
    ```sh
    conda install python=3.11.5  
    conda install portaudio  
    pip install pyaudio==0.2.12  
    pip install gymnasium  
    pip install sbx-rl==0.12.0  
    pip install tensorflow-probability==0.23.0  
    pip install dm_control  
    pip install mujoco_utils  
    pip install tensorboard  
    conda install jax -c conda-forge  
    conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia  
    ```

