# imports
import numpy as np
import pandas as pd
from IPython.display import display
import cufflinks as cf
from scipy.stats import norm
from utilities.UtilFunctions import gbm_generator, credit_spread_simulator, \
    spread_plotter
cf.go_offline(connected=True)



# define inputs
np.random.seed(0)
v0_base, D, T_base, sigma_base, r_base = 200, 150, 20, 0.25, 0.03
dt = 1 / 12
n_paths = 250_000
n_periods = int(T_base / dt)
maturities = np.linspace(start=dt, stop=20, num=240, dtype=np.float64)


### varying firm values
firm_vals = [155, 175, 200, 300]
spread_matrix_varying_vals = np.zeros([len(maturities), len(firm_vals)])
for j, val in enumerate(firm_vals):
    paths_data = gbm_generator(v0=val, T=T_base, sigma=sigma_base,
                               r=r_base, dt=dt, n_paths=n_paths)
    for i, tau in enumerate(maturities):
        spread_matrix_varying_vals[i][j] = credit_spread_simulator(
            paths=paths_data, r=r_base, D=D, T=tau, dt=dt) * 10000

spread_df_varying_vals = pd.DataFrame(spread_matrix_varying_vals, index=maturities,
                                        columns=[f"$V_0 = {v}$" for v in firm_vals])
spread_plotter(spread_df_varying_vals, image_name="spreads_varying_value")


### varying rates
# define set of inputs
rates = [0.01, 0.03, 0.06, 0.1]
spread_matrix_varying_rates = np.zeros([len(maturities), len(rates)])
for j, rate in enumerate(rates):
    paths_data = gbm_generator(v0=v0_base, T=T_base, sigma=sigma_base,
                               r=rate, dt=dt, n_paths=n_paths)
    for i, tau in enumerate(maturities):
        spread_matrix_varying_rates[i][j] = credit_spread_simulator(
            paths=paths_data, r=rate, D=D, T=tau, dt=dt) * 10000
spread_varying_rates = pd.DataFrame(spread_matrix_varying_rates, index=maturities,
                                    columns=[f"$r = {int(rt * 100)} \%$"
                                             for rt in rates])
spread_plotter(spread_varying_rates, image_name="spreads_varying_rates")


####(b)####
# define set of new inputs
sigmas = [.10, .25, .35, .50]
spread_matrix_varying_sigmas = np.zeros([len(maturities), len(sigmas)])

for j, sig in enumerate(sigmas):
    paths_data = gbm_generator(v0=v0_base, T=T_base, sigma=sig,
                               r=r_base, dt=dt, n_paths=n_paths)
    for i, tau in enumerate(maturities):
        spread_matrix_varying_sigmas[i][j] = credit_spread_simulator(
            paths=paths_data, r=r_base, D=D, T=tau, dt=dt) * 10000
spread_varying_sigmas = pd.DataFrame(spread_matrix_varying_sigmas, index=maturities,
                                     columns=[f"$\sigma_v = {int(sig * 100)}\%$"
                                              for sig in sigmas])
spread_plotter(spread_varying_sigmas, image_name="spreads_varying_vol")

