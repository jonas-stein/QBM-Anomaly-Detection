# Anomaly Detection with Quantum Boltzmann Machines using Quantum Annealing

This project provides a way to do Anomaly Detection in an unsupervised manner 
with a semi-restricted Quantum Boltzmann Machine (QBM) and a Restricted Boltzmann Machine (RBM), respectively.
It contains the code used as well as the results presented in the manuscript titled "Exploring Unsupervised Anomaly Detection with Quantum Boltzmann Machines in Fraud Detection" by Jonas Stein, Daniëlle Schuman, Magdalena Benkard, Thomas Holger, Wanja Sajko, Micheal Kölle, Jonas Nüßlein, Leo Sünkel, Oliver Salomon and Claudia Linnhoff-Popien,
which is being submitted to the [IEEE Quantum Week 2023](https://qce.quantum.ieee.org/2023) as a short paper. Find also the [preprint on arXiv](https://arxiv.org/pdf/2306.04998.pdf).
As we make it easy to compare the classic and the quantum approach, this code should be usable to reproduce the results.
A RBM is a type of neural network and we expect the quantum version to eventually take less training time as well as produce better results.
<br><br>
The implementation is done in Python and consists of a classic as well as a quantum approach.
An effort has been made to keep it efficient for local execution. This effort mainly consists
of using numpy and vectorized operations as much as possible since this is a lot faster than 
standard python.
<br><br>
In the following you can find some information on how to set this up and how to use it.

## Setup of Python

Since Python as well as versions of modules and their dependence on each other can sometimes be tricky, we provide files to allow for easy setup.
These files are meant to be used with a virtual python environment and pip. The following commands can be executed in a terminal. Usually 
Python IDEs also provide a way to create virtual environments via a setup file.

```bash
# create a virtual python environment on Desktop in directory "pyenv" 
# give it the prompt AD (Anomaly Detection) to verify if it is activated

cd Desktop
mkdir pyenv
python -m venv --prompt AD pyenv
source pyenv/bin/activate

# it should be actived now and show you the AD prompt
# so now change to the repo's pip_files directory via "cd" 
# do this with the terminal that uses the virtual environment
# then setup the environment via the pip file for your python version

python -m pip install -r pyXX.txt
```

![Setup process of venv](/screenshots/setup.png)

Supported python versions are 3.8 (`py38.txt`), 3.9 (`py39.txt`) and 3.10 (`py310.txt`).

## Project Structure

The project should already contain all the files you need to run everything on your local machine.
Unless otherwise specified the project will use the Simulated Annealing (SA) algorithm as a simulator for the quantum version.

### Generation of Datasets

`generator.py` can be found in the `src` folder.
<br>
This is a script with a command line interface which allows you to generate multiple randomized datasets containing
spherical clusters and outliers. You can specify multiple options like the amount of clusters or the amount of
outliers. To get an overview of all options:

```bash
generator.py -h
```

Datasets will be saved in a folder called `datasets` relative to the path where you called the script from.
Filenames should be descriptive enough to identify datasets, it adheres to flags and values used for creation.
The data itself is saved as numpy array in binary format.

The dataset used to create the experimental results presented in the manuscript is `l_o7_c5_d3_p200_v1.npy`, which is already contained in the dataset folder.

### Visualization of Datasets

`visualizer.py` can be found in the `src` folder.
<br>
This is a script which visualizes the datasets created by `generator.py`.
It takes one or multiple paths to directories or to certain datasets.
The command line interface allows you to specify the used format, for example `.pdf` or `.png`.
To get a list of available formats:

```bash
visualizer.py -h
```

Graphics will be saved to the `graphics` directory which is created relative to the datasets path.

### Restricted Boltzman Machine and Quantum Boltzmann Machine

There are two files `classical_main.py` and `quantum_main.py`. They will run either the classic or
the quantum approach. Some values can be changed to reconfigure the implementations, this is rather
straightforward and the code is documented to be self-explanatory.

## Moving to actual Quantum Hardware

You need two accounts. The first is for [Dwave-Leap](https://github.com/user/repo/blob/branch/other_file.md) and is needed to get 
the embedding graph for the quantum hardware, based on which your machine will locally compute an embedding for the annealer.
After registering there you can get a token which allows you to request these resources.
Our implementation expects you to create a python file called `dwave\_token.py` in folder `src/secrets` that looks like this:

```python
TOKEN = "[YOUR_TOKEN]"
```

The code will then use this token.
<br>
The second one is for [UQO](https://github.com/QAR-Lab/uqoclient), a python framework which simplifies execution of code on Quantum Hardware.
It is developed by the QAR-Lab at LMU and can be installed with pip. It is installed by default if you used the provided install file for pip.
Navigate to `src/secrets` in your terminal and execute the `credentials.py` file there. This will generate all the necessary files for you.
With this step the last thing to do is take up contact with the QAR-Lab to ask for an access token and an endpoint. When you got this, copy the file `src/secrets/config_template.json`,
fill in the missing information and rename it to `config.json`.
<br>
You are now ready to run on real quantum hardware. Go to `quantum_main.py` scroll down and change the solver to the desired hardware.
A list of available machines can be found in the comments.
