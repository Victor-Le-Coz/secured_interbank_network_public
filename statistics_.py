import numpy as np
import numpy.ma as ma
from copy import deepcopy
from scipy.stats import norm
from scipy.stats.mstats import skew
import pandas as pd
from workalendar.europe import EuropeanCentralBank
from statsmodels.tsa.stattools import acf


"""
The data from which the auxiliary statistics are obtained consist of three N × N × T arrays with elements $l_{i,jt} $, $y_{i,j,t}$, and $r_{i,j,t}$. The arrays for $y_{i,j,t}$, and $r_{i,j,t}$ contain missing values if and only if l i,j,t = 0.

The following code allows the building of such historical network object, named ClassHistory. It is fundamentally a sequence of ClassNetwork object.

These 2 classes are define in the below cell.

Few remarks:
1. all $y_{i,j,t}$, and $r_{i,j,t}$ where no transaction occured are modelled by numpy.ma "masked" values.
2. in case several transactions occured on the same date between the same counterparties, the notional of these transaction are added and modelled by a single transaction.
3. only one of the rates of these transactions is used ( in second time, a metric could be used here, like the mean or the median).

"""

# Definition of the class ClassNetwork
class ClassNetwork:
    """
    Object allowing the recording of the whole information of the observed network at a given period, to be used as an instance variable
    by the class classHistory.
    """

    def __init__(self, y, r, l):
        """
        Initialize an instance of the Class ClassMatrices.
        param: y: loan size matrix
        param: r: rates matrix
        param: l: adjency matrix
        param: end_maintenance: dummy variable (boolean) to identify the end of maintenance period
        """
        self.y = y
        self.r = r
        self.l = l
        self.end_maintenance = None


# Definition of the class ClassAnalytics
class ClassAnalytics:
    """
    This class provides the methods to compute the auxilary statistics regarding an observed historical 
    path of the network.
    Its allows the definition of each of the time series of the statistics of the network. These functions are then called by the run method of the class ClassAnalytics.
    """

    def __init__(self, observed_path):
        """
        Initialize an instance of the ClassAnalytics.
        :param: list of the positions of the end of maintenance periods, builded from the observed_path information
        """

        # initialisation of the input observed_path
        self.observed_path = observed_path

        # initialisation of the output results variables
        self.statistics = None
        self.variances = None
        self.statistics_names = None
        self.results = None

        # initialisation of the end_maintenance list based on the
        # maintenance period information stored in the observed_path dict
        # of the network states
        self.end_maintenance = []
        for t in np.arange(len(self.observed_path)):
            if self.observed_path[t].end_maintenance is True:
                self.end_maintenance.append(t)

    def estimate_stat(self, time_serie, block_size):
        """
        Estimate the mean, variance and auto-correlation of a given time_serrie
        Computes the variance of the estimation, trhough the breack of the time serie into n parts
        of size block_size.
        :param: time_serie, a numpy array, something of lengh shorter if the statistic involves cross periods
        :param: block_size: number of time periods of a block on which to compute the statistic to get 
        an estimation of its variance
        :return: a dictionary of the numpy arrays with the list of the statistics and variances respectively
        """
        # take out the information in the time serries that were computed on a end_maintenance period date
        time_serie = np.delete(time_serie, self.end_maintenance)

        # estimate the statistics
        mean = np.mean(time_serie)
        variance = np.var(time_serie)
        autocorrelation = acf(time_serie, nlags=1, fft=True)[1]

        #  split the time serie into blocks of block size
        n = time_serie.size // block_size  # number of blocks
        rest = time_serie.size % block_size  # extra time periods to be deleted
        reduced_time_serie = np.delete(
            time_serie, np.arange(-rest, 0)
        )  # reduce the time serrie
        sub_time_serie = np.split(reduced_time_serie, n)  # build the sub samples

        # estimate the variance of these statistics
        block_means = np.zeros(n)
        block_variances = np.zeros(n)
        block_autocorrelations = np.zeros(n)
        for i in np.arange(n):
            block_means[i] = np.mean(sub_time_serie[i])
            block_variances[i] = np.var(sub_time_serie[i])
            block_autocorrelations[i] = acf(sub_time_serie[i], nlags=1, fft=True)[1]
        var_mean = np.var(block_means)
        var_variance = np.var(block_variances)
        var_autocorrelation = np.var(block_autocorrelations)

        # store the results in a dictionary of 2 numpy arrays: one for statistics, one for their variances
        results = {
            "stat": np.array([mean, variance, autocorrelation]),
            "var": np.array([var_mean, var_variance, var_autocorrelation]),
        }

        return results

    def estimate_rho(self, time_serie_1, time_serie_2, block_size):
        """
        Estimate the correlation between 2 time serries
        Computes the variance of the estimation, trhough the breack of the time serie into n parts
        of size block_size.
        :param: time_serie_1 and time_serie_2: the 2 time serie for which a correlation must be computed
        :param: block_size: number of time periods of a block on which to compute the statistic to get 
        an estimation of its variance
        :return: a dictionary of the numpy arrays with the list of the statistics and variances respectively
        """
        # take out the information in the time serries that were computed on a end_maintenance period date
        time_serie_1 = np.delete(time_serie_1, self.end_maintenance)

        # take out the information in the time serries that were computed on a end_maintenance period date
        time_serie_2 = np.delete(time_serie_2, self.end_maintenance)

        # estimate the correlation
        rho = np.corrcoef(time_serie_1, time_serie_2)[0, 1]

        #  split the time serie into blocks of block size
        n = time_serie_1.size // block_size  # number of blocks
        rest = time_serie_1.size % block_size  # extra time periods to be deleted
        reduced_time_serie_1 = np.delete(
            time_serie_1, np.arange(-rest, 0)
        )  # reduce the time serries
        reduced_time_serie_2 = np.delete(
            time_serie_2, np.arange(-rest, 0)
        )  # reduce the time serries
        sub_time_serie_1 = np.split(
            reduced_time_serie_1, n
        )  # build the sub samples for time serie 1
        sub_time_serie_2 = np.split(
            reduced_time_serie_2, n
        )  # build the sub samples for time serie 2

        # estimate the variance of this correlation
        block_rhos = np.zeros(n)
        for i in np.arange(n):
            block_rhos[i] = np.corrcoef(sub_time_serie_1[i], sub_time_serie_2[i])[0, 1]

        var_rho = np.var(block_rhos)

        # store the results in a dictionary of 2 numpy arrays: one for statistics, one for their variances
        results = {"stat": rho, "var": var_rho}

        return results

    def run(self, block_size, T_rolling_window):
        """
        Compute all the statistics and their associated variance and store them in numpy arrays.     
        :param: history: an instance of the ClassShistory for which analytics must be computed.
        :param: block_size: number of time periods of a block on which to compute the statistic 
        to get an estimation of its variance
        :return: update the 2 numpy arrays of the estimations of the statistics and their variances
        """

        # store statistics and variances into numpy arrays
        self.statistics = np.block(
            [
                self.estimate_stat(compute_density(self.observed_path), block_size)[
                    "stat"
                ][
                    np.array([0, 2])
                ],  # only mean and auto-corr
                self.estimate_stat(compute_reciprocity(self.observed_path), block_size)[
                    "stat"
                ][
                    0
                ],  # choose only mean
                self.estimate_stat(compute_stability(self.observed_path), block_size)[
                    "stat"
                ][
                    0
                ],  # choose only mean
                self.estimate_rho(
                    np.delete(
                        compute_density(self.observed_path), [0]
                    ),  # to align sizes
                    compute_stability(self.observed_path),
                    block_size,
                )["stat"],
                self.estimate_stat(
                    compute_out_degree(self.observed_path)[0],  # select the avg
                    block_size,
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_out_degree(self.observed_path)[1],  # select the std
                    block_size,
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_out_degree(self.observed_path)[2],  # select the skew
                    block_size,
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_in_degree(self.observed_path)[0],  # select the avg
                    block_size,
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_in_degree(self.observed_path)[2],  # select the skew
                    block_size,
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(compute_clustering(self.observed_path), block_size)[
                    "stat"
                ][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_trading_relationship(self.observed_path, T_rolling_window)[
                        0
                    ],  # link corr
                    block_size,
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_trading_relationship(self.observed_path, T_rolling_window)[
                        1
                    ],  # rate corr
                    block_size,
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_log_volume(self.observed_path)[0], block_size  # average
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_log_volume(self.observed_path)[1], block_size  # std
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_log_volume(self.observed_path)[2], block_size  # skew
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_rate(self.observed_path)[0], block_size  # average
                )["stat"][
                    np.array([0, 2])
                ],  # mean and auto-corr
                self.estimate_stat(
                    compute_rate(self.observed_path)[1], block_size  # std
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_rate(self.observed_path)[2], block_size  # skew
                )["stat"][
                    0
                ],  # choose only mean
                self.estimate_rho(
                    compute_density(self.observed_path),
                    compute_rate(self.observed_path)[0],
                    block_size,
                )["stat"],
                self.estimate_stat(
                    compute_volume(self.observed_path)[0], block_size  # average
                )["stat"][
                    2
                ],  # choose only auto-corr
            ]
        )

        self.variances = np.block(
            [
                self.estimate_stat(compute_density(self.observed_path), block_size)[
                    "var"
                ][
                    np.array([0, 2])
                ],  # choose only mean & auto-corr
                self.estimate_stat(compute_reciprocity(self.observed_path), block_size)[
                    "var"
                ][
                    0
                ],  # choose only mean
                self.estimate_stat(compute_stability(self.observed_path), block_size)[
                    "var"
                ][
                    0
                ],  # choose only mean
                self.estimate_rho(
                    np.delete(
                        compute_density(self.observed_path), [0]
                    ),  # to align sizes
                    compute_stability(self.observed_path),
                    block_size,
                )["var"],
                self.estimate_stat(
                    compute_out_degree(self.observed_path)[0],  # select the avg
                    block_size,
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_out_degree(self.observed_path)[1],  # select the std
                    block_size,
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_out_degree(self.observed_path)[2],  # select the skew
                    block_size,
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_in_degree(self.observed_path)[0],  # select the avg
                    block_size,
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_in_degree(self.observed_path)[2],  # select the skew
                    block_size,
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(compute_clustering(self.observed_path), block_size)[
                    "var"
                ][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_trading_relationship(self.observed_path, T_rolling_window)[
                        0
                    ],  # link corr
                    block_size,
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_trading_relationship(self.observed_path, T_rolling_window)[
                        1
                    ],  # rate corr
                    block_size,
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_log_volume(self.observed_path)[0], block_size  # average
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_log_volume(self.observed_path)[1], block_size  # std
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_log_volume(self.observed_path)[2], block_size  # skew
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_rate(self.observed_path)[0], block_size  # average
                )["var"][
                    np.array([0, 2])
                ],  # mean & auto-corr
                self.estimate_stat(
                    compute_rate(self.observed_path)[1], block_size  # std
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_stat(
                    compute_rate(self.observed_path)[2], block_size  # skew
                )["var"][
                    0
                ],  # choose only mean
                self.estimate_rho(
                    compute_density(self.observed_path),
                    compute_rate(self.observed_path)[0],
                    block_size,
                )["var"],
                self.estimate_stat(
                    compute_volume(self.observed_path)[0], block_size  # average
                )["var"][
                    2
                ],  # choose only auto-corr
            ]
        )

        # to help the reading, the list of the names of the statistics is stored in an additional array
        # this array is then concatenated with the numpy arrays of the stats and variances
        self.statistics_names = np.array(
            [
                "mean_density",
                "auto-corr_density",
                "mean_reciprocity",
                "mean_stability",
                "corr_density-stability",
                "mean_avg_out_degree",
                "mean_std_out_degree",
                "mean_skew_out_degree",
                "mean_avg_in_degree",
                "mean_skew_in_degree",
                "mean_avg_clutering",
                "mean_corr_past-trading",
                "mean_corr_rate-trading",
                "mean_avg_log_volume",
                "mean_std_log_volume",
                "mean_skew_log_volume",
                "mean_avg_rate",
                "auto-corr_avg_rate",
                "mean_std_rate",
                "mean_skew_rate",
                "corr_density-avg_rate",
                "auto-corr_avg_volumne",
            ]
        )

        self.results = np.block(
            [[self.statistics_names], [self.statistics], [self.variances]]
        ).T


def compute_density(observed_path):
    """
    Compute the mean and variance of the density of the interbank network across time.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the density.
    """
    # initialisation
    T = len(observed_path)
    N = len(observed_path[0].l)
    density = np.zeros(T)

    # building of the time serie of the statistic
    for t in np.arange(T):
        density[t] = np.sum(observed_path[t].l) / (N * (N - 1))

    # return the time serie
    return density


def compute_reciprocity(observed_path):
    """
    Compute the mean and variance of the reciprocity of the interbank network across time.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the reciprocity
    """
    # initialisation
    T = len(observed_path)
    reciprocity = ma.zeros(T)

    # building of the time serie of the statistic
    for t in np.arange(T):

        # if there are no transactions, we mask the density
        # as the reciprocity cannot be computed
        if np.sum(observed_path[t].l) == 0:
            reciprocity[t] = ma.masked
        else:
            reciprocity[t] = np.sum(
                observed_path[t].l * np.transpose(observed_path[t].l)
            ) / np.sum(observed_path[t].l)

    # return the time serie
    return reciprocity


def compute_stability(observed_path):
    """
    Compute the stability statistic of the interbank network across time.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the stability
    """
    # initialisation
    T = len(observed_path)
    N = len(observed_path[0].l)
    stability = np.zeros(T - 1)

    # building of the time serie of the statistic
    for t in np.arange(T - 1):
        stability[t] = np.sum(
            observed_path[t + 1].l * observed_path[t].l
            + (1 - observed_path[t + 1].l) * (1 - observed_path[t].l)
        ) / (N * (N - 1))

    # return the time serie
    return stability


def compute_out_degree(observed_path):
    """
    Compute, for each time pediod, the out_degree i) average, 
    (ii) standard deviation and (iii) skewness.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the i) average, (ii) standard deviation 
    and (iii) skewness.
    """
    # initialisation
    T = len(observed_path)
    avg_out_degree = np.zeros(T)
    std_out_degree = np.zeros(T)
    skw_out_degree = np.zeros(T)

    # building of the time serie of the statistic
    for t in np.arange(T):
        avg_out_degree[t] = np.mean(np.sum(observed_path[t].l, axis=1))
        std_out_degree[t] = np.std(np.sum(observed_path[t].l, axis=1))
        skw_out_degree[t] = skew(np.sum(observed_path[t].l, axis=1))

    # return the time-series in a tuple
    return avg_out_degree, std_out_degree, skw_out_degree


def compute_in_degree(observed_path):
    """
    Compute, for each time pediod, the out_degree i) average, 
    (ii) standard deviation and (iii) skewness.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the i) average, (ii) standard deviation 
    and (iii) skewness.
    """
    # initialisation
    T = len(observed_path)
    avg_in_degree = np.zeros(T)
    std_in_degree = np.zeros(T)
    skw_in_degree = np.zeros(T)

    # building of the time serie of the statistic
    for t in np.arange(T):
        avg_in_degree[t] = np.mean(np.sum(observed_path[t].l.T, axis=1))
        std_in_degree[t] = np.std(np.sum(observed_path[t].l.T, axis=1))
        skw_in_degree[t] = skew(np.sum(observed_path[t].l.T, axis=1))

    # return the time-series in a tuple
    return avg_in_degree, std_in_degree, skw_in_degree


def compute_clustering(observed_path):
    """
    Compute, for each time pediod, the average of the cluestering coeficient.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the average clustering,.
    """
    # initialisation
    T = len(observed_path)
    N = len(observed_path[0].l)
    avg_clustering = np.zeros(T)
    clustering = ma.zeros((N, T))
    in_and_out_degree = np.zeros((N, T))

    # building of the local cluestering
    for t in np.arange(T):
        in_and_out_degree[:, t] = np.sum(observed_path[t].l, axis=1) + np.sum(
            observed_path[t].l.T, axis=1
        )
        for i in np.arange(N):

            # we can only compute a cluesting for nodes connected, so when no connections we set
            # the cluestering value of the node to a masked value
            if (
                in_and_out_degree[i, t] * (in_and_out_degree[i, t] - 1)
                - 2
                * (
                    np.sum(observed_path[t].l[i] * observed_path[t].l.T[i])
                    - np.sum(np.diag(observed_path[t].l[i]))
                )
            ) == 0:
                clustering[i, t] = ma.masked

            else:
                clustering[i, t] = 0.5 * (
                    np.sum(
                        (
                            observed_path[t].l[i][np.newaxis].T
                            + observed_path[t].l.T[i][np.newaxis].T
                        )
                        * (observed_path[t].l[i] + observed_path[t].l.T[i])
                        * (observed_path[t].l + observed_path[t].l.T)
                    )
                    / (
                        in_and_out_degree[i, t] * (in_and_out_degree[i, t] - 1)
                        - 2
                        * (
                            np.sum(observed_path[t].l[i] * observed_path[t].l.T[i])
                            - np.sum(np.diag(observed_path[t].l[i]))
                        )
                    )
                )

    # building of the time serie of the statistic
    avg_clustering = ma.mean(clustering, axis=0)

    # return the time serie
    return avg_clustering


def compute_trading_relationship(observed_path, T_rolling_window):
    """
    Compute, for each time pediod, the correlation across pairs of institutions between 
    the intensity of past trading relationships and curent links and or curent loans.
    :param: observed_path: a dictionary of Network objects across time.
    :param: T_rolling_window: the lenghs of the rollwing window to recorde the past trading history.
    :return: the time serie of (i) the correlation across pairs between curent links 
    and past trading history and (ii) the correlation across pairs between curent loan rates 
    and past trading history.
    """
    # initialisation
    T = len(observed_path)
    N = len(observed_path[0].l)
    corr_links_trading = ma.zeros(
        T - T_rolling_window + 1
    )  # shorten the history by T_rolling_window
    corr_rates_trading = ma.zeros(T - T_rolling_window + 1)
    trading_history = {
        t: np.zeros((N ** 2)) for t in np.arange(T)
    }  # flatten all the i,j into a N**2 matrix
    path_flatten_l = {t: np.zeros((N ** 2)) for t in np.arange(T)}
    path_flatten_r = {t: np.zeros((N ** 2)) for t in np.arange(T)}

    # building of the local cluestering
    for t in np.arange(T):

        # flatten the links and rates matrices
        path_flatten_l[t] = observed_path[t].l.flatten()
        path_flatten_r[t] = observed_path[t].r.flatten()

        # fill the trading history dictionary from t = T_rolling_window - 1
        # (due to numbering from 0)
        if t >= T_rolling_window - 1:
            for t_prime in np.arange(t - T_rolling_window + 1, t + 1):
                trading_history[t] = trading_history[t] + path_flatten_l[t_prime]

        # building of the time serie of the statistic
        if t >= T_rolling_window - 1:

            # use of ma.corrcoef function so that results are flagged as masked
            # in case any of the standard deviation of the random values is nill (meaning
            # the correlation coeficient is not defined)
            corr_links_trading[t - T_rolling_window + 1] = ma.corrcoef(
                path_flatten_l[t], trading_history[t]
            )[0, 1]
            corr_rates_trading[t - T_rolling_window + 1] = ma.corrcoef(
                path_flatten_r[t], trading_history[t]
            )[0, 1]

    # return the time serie
    return corr_links_trading, corr_rates_trading


def compute_log_volume(observed_path):
    """
    Compute, for each time pediod, the log volumes i) average, 
    (ii) standard deviation and (iii) skewness.
    All the null volumnes are set to 1 to allow the computation of their logarithms.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the i) average, (ii) standard deviation 
    and (iii) skewnes of the log volumnes. 
    """
    # initialisation
    T = len(observed_path)
    avg_log_volume = ma.zeros(T)  # use masked arrays to handle missing values
    std_log_volume = ma.zeros(T)
    skw_log_volume = ma.zeros(T)

    # building of the time serie of the statistic
    for t in np.arange(T):

        # fill in the mean, standard deviation and skewness of the log loan size
        avg_log_volume[t] = ma.mean(ma.log(observed_path[t].y))
        std_log_volume[t] = ma.std(ma.log(observed_path[t].y))
        skw_log_volume[t] = skew(ma.log(observed_path[t].y).flatten())

    # return the time-series in a tuple
    return avg_log_volume, std_log_volume, skw_log_volume


def compute_rate(observed_path):
    """
    Compute, for each time pediod, the interbank rate i) average, 
    (ii) standard deviation and (iii) skewness.
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the i) average, (ii) standard deviation 
    and (iii) skewness of the interbank rate.
    """
    # initialisation
    T = len(observed_path)
    avg_rate = ma.zeros(T)
    std_rate = ma.zeros(T)
    skw_rate = ma.zeros(T)

    # building of the time serie of the statistic
    for t in np.arange(T):

        # fill in the mean, standard deviation and skewness of the log loan size
        avg_rate[t] = ma.mean(observed_path[t].r)
        std_rate[t] = ma.std(observed_path[t].r)
        skw_rate[t] = skew(observed_path[t].r.flatten())

    # return the time-series in a tuple
    return avg_rate, std_rate, skw_rate


def compute_volume(observed_path):
    """
    Compute, for each time pediod, the i) average, (ii) standard deviation
    and (iii) skewness of the loans' volume (i.e. nominal amount).
    :param: observed_path: a dictionary of Network objects across time.
    :return: the time serie of the i) average, (ii) standard deviation 
    and (iii) skewness.
    """
    # initialisation
    T = len(observed_path)
    avg_volume = ma.zeros(T)
    std_volume = ma.zeros(T)
    skw_volume = ma.zeros(T)

    # building of the time serie of the statistic
    for t in np.arange(T):

        # fill in the mean, standard deviation and skewness of the log loan size
        avg_volume[t] = ma.mean(observed_path[t].y)
        std_volume[t] = ma.std(observed_path[t].y)
        skw_volume[t] = skew(observed_path[t].y.flatten())

    # return the time-series in a tuple
    return avg_volume, std_volume, skw_volume
