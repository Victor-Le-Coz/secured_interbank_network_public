from dynamics import ClassDynamics
from network import ClassNetwork

# parameter setting
nb_banks=300
path_results="./results/single_run/rgm_300_test_init_log/"


# reg ratios
alpha_init=False # initial cash (< 1/(1-gamma) - beta)
alpha=0.01
beta_init=0.5 # initial collateral  (< 1/(1-gamma) - alpha)
beta_reg=0.5
beta_star=0.5
gamma=0.03
gamma_init = 3*gamma #3*gamma
collateral_value=1.0

# initialisation of deposits size or through money creation
initialization_method="pareto"
alpha_pareto=1.4
initial_deposits_size = False #40 if False, use the init money min and money creation
init_money_min = 1e-2 # 10 million money units, minimum for a bank license

# shocks on deposits
shocks_method="non-conservative"
shocks_law="normal-mean-reverting"
shocks_vol=0.07 # 0.08


# speed of learning
learning_speed = 0.5 #0.5

# min trans size
min_repo_trans_size=1e-8  # 1e-8

# dynamics & ploting
nb_steps=int(40e3) #int(10e3)
dump_period=int(40e3) #int(5e2)
plot_period=int(40e3) #int(5e2)
cp_option=True
heavy_plot=False

# LCR mgt
LCR_mgt_opt=True

# leverage mgt
end_repo_period=False # if int, periodic end repo / if false, leverage mgt
gamma_star = 1.5*gamma #1.5
check_leverage_opt = False # to avoid killing the run if one or several banks are below min leverage due to high shocks (there is not possibility of decrease balance sheet size if no interbank borrowings)

# money creation
loan_tenor=nb_steps #nb_steps # if int, money creation / if false. no new loans
loan_period=1
new_loans_vol = 5 #5 standard deviation around the mean creation of loans (if initial deposits size not False)
new_loans_mean = 10e-2/250 #2e-2/250 daily mean increase in loans expressed as a percentage of the intital loans, meaning linear growth (or of the current loans, then meaning exponential growth)
beta_new = beta_reg # if number, new colat / if false, no new colat 
gamma_new = 2*gamma_star


# substitution of collateral
substitution = False

# Quantitative easing scenario
QE_start = False
QE_stop = False

# GFC scenario
no_trust_start = False
no_trust_stop = False

# initialize ClassNetwork
Network = ClassNetwork(
    nb_banks=nb_banks,
    initial_deposits_size=initial_deposits_size,
    alpha_init=alpha_init,
    alpha=alpha,
    beta_init=beta_init,
    beta_reg=beta_reg,
    beta_star=beta_star,
    beta_new=beta_new,
    gamma_init=gamma_init,
    gamma=gamma,
    gamma_star=gamma_star,
    gamma_new=gamma_new,
    collateral_value=collateral_value,
    initialization_method=initialization_method,
    alpha_pareto=alpha_pareto,
    shocks_method=shocks_method,
    shocks_law=shocks_law,
    shocks_vol=shocks_vol,
    LCR_mgt_opt=LCR_mgt_opt,
    min_repo_trans_size=min_repo_trans_size,
    loan_tenor=loan_tenor,
    loan_period=loan_period,
    new_loans_vol=new_loans_vol,
    new_loans_mean=new_loans_mean,
    end_repo_period=end_repo_period,
    substitution=substitution,
    learning_speed=learning_speed,
    check_leverage_opt=check_leverage_opt,
    init_money_min=init_money_min,
    QE_start=QE_start,
    QE_stop=QE_stop,
    no_trust_start=no_trust_start,
    no_trust_stop=no_trust_stop,
)

# initialize ClassDynamics
Dynamics = ClassDynamics(
    Network,
    nb_steps=nb_steps,
    path_results=path_results,
    dump_period=dump_period,
    plot_period=plot_period,
    cp_option=cp_option,
    heavy_plot=heavy_plot,
)

# simulate
Dynamics.simulate()