<!-- <div style="text-align:center"><img src="docs/assets/Tax.png" /></div> -->

# MixER

This repository describes the implementation of MixER. Install the package and modify the main script to train on datasets in `examples/`. 

`pip install -e .`

MixER is built around 5 extensible modules: 
- a DataLoader: to store the dataset
- a Learner: a model and loss function
- A Trainer: to train
- a VisualTester: to test and visualize the results
