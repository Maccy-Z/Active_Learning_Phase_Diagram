# Graphical User Interface for active learning for sampling phase diagrams. 

## Usage Instructions
1) Run the dockerfile to create a container with the necessary dependencies. 
2) From the docker container, run with python run_gui.py. Note, you may need to run "xhost + " to allow the GUI to be displayed. 

The configuration can be edited in /code/config.py. There, the sampling domain, number of phases, resolution and Gaussian Process parameters can be set. 