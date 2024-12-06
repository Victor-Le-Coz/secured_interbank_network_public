
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

### 7. emp_fake_data Module
This module provides fake MMSR transaction data, exposure views, and FINREP databases for testing the empirical analysis tools. It ensures that the analysis methods are robust and reliable.

### 8. network_statistics Module
The `network_statistics` module defines a class for computing network statistics based on the original article by Blasques (2018). These statistics help in understanding the structural properties and systemic risk of the network.

### 9. Notebooks
The repository includes several Jupyter notebooks for running empirical measures on the MMSR database, testing, analysis, and development work. These notebooks facilitate the exploration and validation of the model and its components.

### 10. Functions Module
This module contains various utility functions used throughout the project. These functions support the main simulation and analysis tasks, ensuring efficient and effective code execution.

This research code provides a comprehensive framework for studying the dynamics and stability of interbank networks using an agent-based modeling approach. It enables researchers to simulate different scenarios, analyze the resulting data, and gain insights into the factors influencing systemic risk in the banking sector.

