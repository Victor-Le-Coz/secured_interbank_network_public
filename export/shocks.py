import numpy as np
from scipy import stats


def generate_bilateral_shocks(deposits, law, vol):
    # define middle of the list of banks
    N_max = len(deposits) - len(deposits) % 2  # can not apply a shock on
    # one bank if odd nb
    N_half = int(len(deposits) / 2)

    # create a permutation of all the deposits amounts
    ix = np.arange(len(deposits))  # create an index
    ix_p = np.random.permutation(ix)  # permutation of the index
    deposits_p = deposits[ix_p]  # define the permuted array of deposits

    # apply a negative relative shock on the first half of the banks
    if law == "uniform":
        rho_1 = np.random.uniform(-1, 0, size=N_half)

    elif law == "beta":
        rho_1 = -np.random.beta(1, 1, size=N_half)

    elif law == "log-normal":
        std_control = np.sqrt(np.log(1.0 + vol**2.0))
        rho_1 = get_trunc_lognorm(
            mu=-0.5 * std_control**2,
            sigma=std_control,
            lower_bound=0,
            upper_bound=1,
            size=N_half,
        )

    else:
        assert False, ""

    # apply a positive relative shock on the second half of the banks
    rho_2 = -rho_1 * deposits_p[0:N_half] / deposits_p[N_half:N_max]

    # concatenate the relative shocks
    if len(deposits) > N_max:
        rho = np.concatenate([rho_1, rho_2, [0]])
    elif len(deposits) == N_max:
        rho = np.concatenate([rho_1, rho_2])
    else:
        assert False, ""

    # build an un-permuted array of absolute shocks
    shocks = np.zeros(len(deposits))

    # compute the absolute shock from the deposit amount
    shocks[ix_p] = deposits_p * rho

    return shocks


def generate_multilateral_shocks(deposits, law, vol):
    # define middle of the list of banks
    N_max = len(deposits) - len(deposits) % 2  # can not apply a shock on
    # one bank if odd nb
    N_half = int(len(deposits) / 2)

    # create a permutation of all the deposits amounts
    ix = np.arange(len(deposits))  # create an index
    ix_p = np.random.permutation(ix)  # permutation of the index
    deposits_p = deposits[ix_p]  # define the permuted array of deposits

    # apply a shock on the first half of the banks
    if law == "uniform":
        rho = np.random.uniform(-0.1, 0.1, size=N_max)  # case uniform  law

    elif law == "beta":
        rho = -np.random.beta(1, 1, size=N_half)  # case beta  law

    else:
        assert False, ""

    rho_1 = rho[0:N_half]
    rho_2 = rho[N_half:N_max]

    correction_factor = -(
        np.sum(rho_1 * deposits_p[0:N_half])
        / np.sum(rho_2 * deposits_p[N_half:N_max])
    )

    rho_2 = rho_2 * correction_factor

    # concatenate the relative shocks
    if len(deposits) > N_max:
        rho = np.concatenate([rho_1, rho_2, [0]])
    elif len(deposits) == N_max:
        rho = np.concatenate([rho_1, rho_2])
    else:
        assert False, ""

    # build an un-permuted array of absolute shocks
    shocks = np.zeros(len(deposits))

    # compute the absolute shock from the deposit amount
    shocks[ix_p] = deposits_p * rho

    return shocks


def generate_dirichlet_shocks(deposits, initial_deposits, option, vol):

    std_control = 1.0 / (vol**2.0)

    if option == "dynamic":
        dispatch = np.random.dirichlet(
            (np.abs(deposits + 1e-8) / deposits.sum()) * std_control
        )
    elif option == "static":
        dispatch = np.random.dirichlet(
            (np.ones(len(deposits)) / len(deposits)) * std_control
        )
    elif option == "mean-reverting":
        dispatch = np.random.dirichlet(
            (initial_deposits / initial_deposits.sum()) * std_control
        )
    else:
        assert False, ""

    new_deposits = deposits.sum() * dispatch
    shocks = new_deposits - deposits

    return shocks


def generate_non_conservative_shocks(deposits, law, vol):
    if law == "log-normal":
        std_control = np.sqrt(np.log(1.0 + vol**2.0))
        new_deposits = (
            np.random.lognormal(
                mean=-0.5 * std_control**2,
                sigma=std_control,
                size=len(deposits),
            )
            * deposits
        )
    elif law == "normal":
        new_deposits = np.maximum(
            deposits + np.random.randn(len(deposits)) * vol, 0.0
        )
    else:
        assert False, ""
    shocks = new_deposits - deposits
    return shocks


def get_trunc_lognorm(mu, sigma, lower_bound, upper_bound=np.inf, size=10000):
    norm_lower = np.log(lower_bound)
    norm_upper = np.log(upper_bound)
    X = stats.truncnorm(
        (norm_lower - mu) / sigma,
        (norm_upper - mu) / sigma,
        loc=mu,
        scale=sigma,
    )
    norm_data = X.rvs(size)
    log_norm_data = np.exp(norm_data)
    return log_norm_data
