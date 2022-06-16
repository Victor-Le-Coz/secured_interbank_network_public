from network import InterBankNetwork
from tqdm import tqdm

if __name__ == "__main__":
    network = InterBankNetwork(
        n_banks=5,
        alpha_pareto=0.5,
        perc_deposit_shock=0.1,
        initial_mr=1.0,
        initial_lcr=100.0,
        initial_l2s=3.0,
        collateral_value=1.0,
    )
    for i in tqdm(range(1000)):
        network.step_network()
    for bank in network.banks:
        print(bank)
        print(
            "Bank Deposits {} Bank Cash {}".format(
                bank.liabilities["Deposits"], bank.assets["Cash"]
            )
        )
