# Active Learning for Discovering Complex Phase Diagrams with Gaussian Processes

## Introduction
This project is an implementation of the paper "Active Learning for Discovering Complex Phase Diagrams with Gaussian Processes". This algorithm uses Gaussian Processes to model the phase diagram and uses active learning to select the most informative points to sample. A graphical user interface is included for visualizing the phase diagram in 2D and 3D.

## Getting Started
### Prerequisites
This program has been tested using Python 3.11. Any modern operating system should be supported.

### Installation

#### 1. Clone the repository
First, you need to clone the project repository from GitHub to your local machine. Open a terminal (Command Prompt or PowerShell on Windows, Terminal on macOS and Linux), and run the following command:

```
https://github.com/Maccy-Z/Active_Learning_Phase_Diagram.git
```
#### 2. Create a Python virtual environment

After cloning the repository, navigate into the project directory:

```
cd Active_Learning_Phase_Diagram
```
Next, create a Python virtual environment by running:
```
python -m venv venv
```
This command creates a virtual environment named `venv` in your project directory, which helps isolate project dependencies from other Python projects on your system.
#### 3. Activate the virtual environment

Before installing dependencies or running the project, you need to activate the virtual environment. Depending on your operating system, use one of the following commands:

- **Windows:**
```
- \venv\Scripts\activate
```
- **macOS and Linux:**
```
source venv/bin/activate
```
#### 4. Install project dependencies

With the virtual environment activated, install the project dependencies by running:
```
pip install -r requirements.txt
```
### Docker
Alternatively, you can use Docker to build project (Linux recommended). First, make sure you have Docker installed on your machine. Then, build the Docker image by running:
```
docker build -t active_learning_phase_diagram .
```

## Usage
The main scripts are in /src/. The configuration for experiments can be changed in /src/config.py. 
### Configuration 
Before running the main program, the configuration must be correctly set.

Number of dimensions of parameter space: 
```
N_dim: int = {2 or 3}
```
In the **post_init** section, set the range of the parameter space to explore as a tuple of tuples, where each tuple represents the range of a parameter. Number of dimensions here must match **N_dim** above. For example, for a 2D parameter space with the first parameter ranging from 0 to 1 and the second parameter ranging from 0 to 2, set:
```
self.extent = ((0, 1), (0, 2))
```
For a 3D parameter space with the first parameter ranging from 0 to 1, the second parameter ranging from 0 to 2, and the third parameter ranging from 0 to 3, set:
```
self.extent = ((0, 1), (0, 2), (0, 3))
```
Total number of phases in the phase diagram. Note too many phases will give slow sampling. 
```
N_phases: int = ...
```
### Optional parameters
Number of points to search over. The search space will be a **N_dist** x **N_dist** (x **N_dist**) grid. Typically, set **N_dist**=**N_eval**. Too many points will give slow sampling. **N_display** is the number of points displayed in the GUI, higher than number of search points for good quality graphics. 
```
N_dist: int = ...
```

The parameters **skip_point** and **skip_phase** are used to skip sampling points that have high certantity to speed up sampling. **sample_dist** is size of region to compute distance function over, a larger distance is more accurate but slower.
**gaus_herm_n** is the number of points to use in the Gauss-Hermite quadrature. **N_optim_steps** is the number of optimization steps to for parameter optimization. **N_CPUs** is the number of CPU cores to use for parallel processing. 
**obs_prob** is the default probability for observations. A higher value means more certain observations. Note this is the default global value. It can also be set on a per-observation basis. 
## Running the program
To run the graphical user interface for 2D phase diagrams, run
```
python src/run_gui2D.py
```
For the 3D phase diagrams, run
```
python src/run_gui3D.py
```
Note the config **N_dim** must be set to 2 or 3 respectively to run the GUI. For other dimensional parameter spaces, the GUI is not possible and the program must be run from the command line.

When running the GUI, an window wil open prompting for first/exisitng observations. Enter the initial observations (at least 1) in comma seperated format. 

The main GUI will then open after a pause. Suggestions will be shown on the plot, and the exact suggested coordinates will also be shown. After conducting an observation at the suggested point, enter the measured phase (as an integer) and observation certantity (can be blank to use default). 
Note a different coordinate can be entered if you do not want to use the suggested point. Note the program will freeze during the computation. This is normal. 

## Reproducing Results
The folder /src/synthetic_results contains the baseline implementations. The folder /src/analysis contains the code used to analyse the results. 

