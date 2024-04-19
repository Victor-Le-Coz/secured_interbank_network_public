# secured-inter-bank-network
Here is a description of the structure of the project

## module dynamics
Definition of ClassDynamics.
This class is initialised using and instance Network of ClassNetwork.
This class initialises also an instance of ClassGraphics.
This class simulate a Network and stores all the historical information about this Network.
Most metrics are computed using the their definition in the module emp_metrics and stored in df_network_trajectory, a DataFrame with the historical information at network level.
The metrics specific to one bank are store in df_bank_trajectory.
The information about each individual bank across time is not stored in general, except the transaction information in df_reverse_repo and their associated exposure view (store in adj matrix) in arr_rev_repo_exp_adj (a nb_steps * nb_banks * nb_banks numpy array).
After aggregation in rolling windows, the binary adjencies are stored in dic_arr_binary_adj.
An other exception is dic_in_degree, dic_out_degree, dic_degree, which stores, for each agg_period, an array of the degree per bank across time.

## module network (subclass of dynamics)
Definition of ClassNetwork.
This class is initialises nb_banks instances of ClassBank.
It defines how to simulate the network for one step.
It maintains the time with Network.step.
It maintains df_banks, a DataFrame with all the exposure information about each bank at a given time step.

## module bank (subclass of network)
Definition of the ClassBank.
This class has access to Network and to all the other instances of ClassBank in the Network.
This class maintains at local level the history of the reserve repo transaction of this Bank lended to the other intances of ClassBank.

## module Graphics (subclass of dynamics)
Definition of the ClassGraphics.
This class plots the trajectory of the states of the network across time using df_network_trajectory, df_bank_trajectory, arr_rev_repo_exp_adj, dic_arr_binary_adj, and dic_degree.

## notebook emp_analysis
Notebook to run all empirical measures on the MMSR data base.

# module emp_metrics
Definition of all the metrics used for both the simulation and the empirical measures.

# module emp_preprocessing
Functions to transform transactions into exposures and exposures into binary adj matrices.

# module emp_fake_date
Definition of a fake MMSR transactions, MMSR exposure view, and FINREP data bases to test the empirical analysis tools

## notebook tools
For tests, analysis, dev work.

## module funcitons
Useful functions used in the project

## module network_statistics
Definition of a Class analytics for the computation of the network statistics in the original article from Blasques (2018)


