from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import os


def plot_loans_mro(metrics, path):
    plt.figure()
    length = len(metrics["Securities Usable"])
    plt.plot(np.arange(length), metrics["Loans"])
    plt.plot(np.arange(length), metrics["MROs"])
    plt.legend(["Loans", "MROs"])
    plt.xlabel("Steps")
    plt.ylabel("Total Network Amount")
    plt.title("Total Amount of Loans/MROs in Network")
    plt.savefig(os.path.join(path, "loans_mros.png"))
    plt.close()


def plot_collateral(metrics, path):
    plt.figure()
    length = len(metrics["Securities Usable"])
    plt.plot(np.arange(length), metrics["Securities Usable"])
    plt.plot(np.arange(length), metrics["Securities Encumbered"])
    plt.plot(np.arange(length), metrics["Securities Collateral"])
    plt.plot(np.arange(length), metrics["Securities Reused"])
    plt.legend(
        [
            "Securities Usable",
            "Securities Encumbered",
            "Securities Collateral",
            "Securities Reused",
        ]
    )
    plt.xlabel("Steps")
    plt.ylabel("Total Network Amount")
    plt.title("Total Amount of Collateral in Network")
    plt.savefig(os.path.join(path, "collateral.png"))
    plt.close()


def plot_network(adj, path, step):
    # build a network from an adjency matrix
    bank_network = nx.from_numpy_matrix(
        adj, parallel_edges=False, create_using=nx.DiGraph
    )

    # define the weight list from the weight information
    weights = [
        bank_network[node1][node2]["weight"]
        for node1, node2 in bank_network.edges()
    ]

    # define the position of the nodes
    pos = nx.spring_layout(bank_network)

    # draw the network
    plt.figure(1, figsize=(15, 15))
    nx.draw_networkx(bank_network, pos, width=weights, with_labels=True)

    # show the plot
    plt.title("Interbank network at the final step")
    plt.savefig(os.path.join(path, "step_{}_network.png".format(step)))
    plt.close()


class Graphics:
    """
    This class allows the representation or either the sample paths or the network state at a given time step.

    :param: network: the network to be plotted
    """

    def __init__(self, network):
        """
        Initialize an instance of the ClassGraphics.
        """
        self.network = network

    def loan_size_path(self):  # to be updated !
        """
        Plot the time serrie of the average loan size.
        """
        # create a vector of the average loan size
        average_loan_size = np.zeros(self.dynamics.T)
        for t in np.arange(self.dynamics.T):
            average_loan_size[t] = np.mean(
                self.dynamics.sample_path[t].wl
            )  # mean across 2 directions i and j

        # plot the average loan size
        plt.xlabel("time")
        plt.ylabel("average loan size")
        plt.plot(np.arange(self.dynamics.T), average_loan_size)
        plt.show()

    def search_cost_path(self):  # to be updated !
        """
        Plot the time serrie of the average search cost.
        """
        # create a vector of the average search cost
        average_search_cost = np.zeros(self.dynamics.T)
        for t in np.arange(self.dynamics.T):
            average_search_cost[t] = np.mean(
                self.dynamics.sample_path[t].s
            )  # mean across 2 directions i and j

        # plot the average search cost
        plt.xlabel("time")
        plt.ylabel("average search cost")
        plt.plot(np.arange(self.dynamics.T), average_search_cost)
        plt.show()

    def plot_network(self):
        """
        Plot the network state at the last time step.
        """
        # create an adjency matrix of the links from the sample path dictionary at time t
        adj = self.network.adj_matrix

        # build a network from an adjency matrix
        bank_network = nx.from_numpy_matrix(
            adj, parallel_edges=False, create_using=nx.DiGraph
        )

        # define the weight list from the weight information
        weights = [
            bank_network[node1][node2]["weight"]
            for node1, node2 in bank_network.edges()
        ]

        # define the position of the nodes
        pos = nx.spring_layout(bank_network)

        # draw the network
        plt.figure(1, figsize=(15, 15))
        nx.draw_networkx(bank_network, pos, width=weights, with_labels=True)

        # show the plot
        plt.title("Interbank network at the final step")
        plt.show()
