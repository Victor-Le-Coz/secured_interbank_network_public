from network import InterBankNetwork
from tqdm import tqdm

if __name__ == "__main__":
    network = InterBankNetwork(
        n_banks=50,
        alpha_pareto=0.5,
        perc_deposit_shock=0.1,
        beta_lcr=10.0,
        beta_star_lcr=10.0,
        initial_mr=1.0,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
    )
    network.simulate(1000)
