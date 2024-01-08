# dumbRL

## Introduction

dumbRL is an evolving collection of reinforcement learning (RL) environments, aimed at providing a playground for experimentation and learning. The project structure is designed to be intuitive and expandable, accommodating new RL environments and experiments.

## Project Structure

The project is organized into several key directories:

- `environments`: This directory contains individual RL environments. Each environment is a separate module, facilitating easy addition and modification of environments.
- `experiments`: Here, you can find scripts to run experiments using the environments. These scripts demonstrate how to interact with and utilize the environments.
- `utils`: A utility module for common functions and classes shared across the project.


## Project installation

1. For package management, we use [Poetry](https://python-poetry.org/), please refer to the [installation](https://python-poetry.org/docs/#installing-with-pipx) guide to install it on your machine. After installing poetry, run the following command to install the project dependencies:
   ```bash
   poetry install
   poetry shell # To activate the virtual environment
   ```
2. To run the experiments, currently each environment is ran is a module.
   ```bash
    poetry run python -m dumbrl.experiments.run_press_the_light_up_button # if you don't want to activate the virtual environment, you can run the following command
    python -m dumbrl.experiments.run_press_the_light_up_button # if you have activated the virtual environment
   ```

