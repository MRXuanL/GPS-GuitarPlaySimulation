# GPS-GuitarPlaySimulation

A guitar play simulation system capable of reading tablature and, after training, guiding robotic hands to play the guitar.
Our system works best with Linux OS because JAX has some bugs on Windows OS. For example, when training on Windows OS, it may not use the GPU for training. Implementing the DroQ algorithm using PyTorch could potentially solve this issue.
[[Project Page]](https://mrxuanl.github.io/publications/guitarplayer.html)
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
    conda install portaudio=19.6.0  
    pip install pyaudio==0.2.12  
    pip install gymnasium==0.29.1  
    pip install sbx-rl==0.12.0  
    pip install matplotlib==3.8.4
    pip install tensorflow-probability==0.23.0  
    pip install dm_control==1.0.18  
    pip install mujoco_utils==0.0.6  
    pip install tensorboard==2.16.2  
    pip install mujoco==3.1.4
    conda install jax -c conda-forge  
    conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
    pip install dm_env_wrappers==0.0.13
    ```

## Usage

1. Find the `GPS-GuitarPlaySimulation/guitarplay/train_guitar.py` file.
2. Find the `main` function. If you want to train the model, you can use the `train` function for training and disable the `test` function.
3. You can modify the `args` array to change the parameters for training.
4. You can also change the value of `arg.table` to select another song for training.
5. Song IDs are listed in `GPS-GuitarPlaySimulation/tasklist.txt`.
6. Finally, run the command to train the model:
   ```sh
    python train_guitar.py
   ```
7. After training, you can use the `test` function to test the model and disable the `train` function. Note that you should keep the args in the `train` and `test` functions the same!
8. After completing the testing, you can find the video of your trained song in the `~/dm_env_wrappers` directory.


## Add a new song
If you know how to use Guitar Pro software, you can get a tablature from Guitar Pro software in MusicXml format. Then, using [MusicXmltoTarget](https://github.com/MRXuanL/MusicXmltoTarget), translate the MusicXml file to our target format.

This will generate a new `out.txt` file. Please copy it to our project, replacing the original `out.txt` file.

Then you need to train/test any song to update the `tasklist.txt` file for training. Please find the new song ID from the `tasklist.txt` for training.

## Check the result log
After training, you will find a new folder named result, which contains the best model and F1 score, recall, and precision metrics. You can run the command to see the metrics on the website:
```sh
tensorboard --logdir result
```

## Questions
Please feel free to ask me any questions about this project!


## BibTeX
```sh

@article{10.1111:cgf.15166,
journal = {Computer Graphics Forum},
title = {{Learning to Play Guitar with Robotic Hands}},
author = {Luo, Chaoyi and Tang, Pengbin and Ma, Yuqi and Huang, Dongjin},
year = {2024},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.15166}
}
        
```







