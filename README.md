
# Research Code: Agent-Based Model of the Banking Network
This repository contains the research code for an agent-based model (ABM) of the banking network. The model simulates the interactions between banks in an interbank network, focusing on the dynamics of reserve repo transactions and the resulting exposures. The primary goal is to analyze the stability and systemic risk within the banking network under various scenarios.

## Key Components

### 1. ClassDynamics
The `ClassDynamics` is the core of the simulation, responsible for initializing the network and graphics modules. It simulates the network over time and stores historical data, including network-level metrics and bank-specific metrics. The historical data is stored in DataFrames and numpy arrays for further analysis.

### 2. ClassNetwork
The `ClassNetwork` manages the creation of bank instances and the simulation of network steps. It maintains a DataFrame with exposure information for each bank at each time step, facilitating the analysis of interbank relationships and systemic risk.

### 3. ClassBank
The `ClassBank` represents individual banks within the network. Each bank has access to the network and other banks, allowing it to record and manage its reserve repo transactions. This class is crucial for simulating the behavior and interactions of banks in the network.

### 4. ClassGraphics
The `ClassGraphics` handles the visualization of the network's state over time. It uses the historical data stored in various DataFrames and arrays to plot the trajectories of network metrics and bank-specific metrics, providing insights into the network's evolution.

### 5. emp_metrics Module
This module defines all the metrics used for both simulation and empirical analysis. These metrics are essential for evaluating the performance and stability of the banking network under different conditions.

### 6. emp_preprocessing Module
The `emp_preprocessing` module contains functions to transform transaction data into exposure data and then into binary adjacency matrices. These transformations are necessary for analyzing the network structure and dynamics.

### 7. Notebook Single_run
The repository includes a Jupyter notebooks for running a simulation of the model.

### 10. Functions Module
This module contains various utility functions used throughout the project. These functions support the main simulation and analysis tasks, ensuring efficient and effective code execution.
