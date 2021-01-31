# imports
import numpy as np
import pandas as pd
import numpy.random as rnd
import cufflinks as cf  # library to use plotly plots along with pandas
import plotly.graph_objs as go
from tqdm import trange
from scipy.stats import norm

def gbm_generator(v0: float, T: int, sigma: float,
                  r: float, dt: float, n_paths=250_000):
    """
    :param v0: initial firm value
    :param T: number of years
    :param sigma: diffusion of GBM
    :param r: drift of GBM (risk-free rate)
    :param dt: inverse of frequency of observations per year (in years)
    :param n_paths: number of different paths to be simulated
    :return: a matrix of n=n_paths GBMs over T years sampled every dt year
    """
    n_periods = int(T/dt)       # total number of observations
    # create empty matrix with dimension (n_periods+1)*n_paths
    v = np.zeros((n_periods + 1, n_paths))
    v[0, :] = v0 * np.ones(n_paths)
    # simulate firm value path according to GBM model
    for i in trange(n_periods):
        rand_vec = np.random.normal(0, 1, n_paths)
        v[i+1, :] = v[i, :] * np.exp((r - sigma ** 2 / 2) * dt +
                                    sigma * (dt ** 0.5) * rand_vec)
    return v


def merton_minus_bc_squared(Dt_: np.array, pd_mert: float, paths: np.array):
    """
    :param Dt_: vector of discounted Face Value of Debt D
    :param pd_mert: reference default probability under Merton
    :param paths: matrix of simulated paths
    :return: squared error (p_mert - p_bc)^2
    """
    default_stat = np.any(paths < Dt_.reshape(-1, 1), axis=0)
    pd_bcox = len(default_stat[default_stat]) / len(default_stat)
    return (pd_mert - pd_bcox) ** 2


def credit_spread_simulator(paths, r: float, D: float, T, dt):
    """
    :param paths: matrix of simulated paths
    :param r: risk-free rate
    :param D: Face value of debt
    :param T: maturity
    :param dt:  inverse of frequency of observations per year (in years)
    :return: credit_spread
    """
    maturity_row = int(T / dt)
    payoffs_at_maturity = np.where(paths[maturity_row, :] - D >= 0,
                                   paths[maturity_row, :] - D, 0)
    # discounted expected payoff
    call = np.exp(-r * T) * np.mean(payoffs_at_maturity)
    v0 = paths[0, 0]
    b0 = v0 - call
    y0 = (1 / T) * np.log(D / b0)

    return y0 - r


# define function for plots
def spread_plotter(spread_data: pd.DataFrame, image_name: str):
    figure = spread_data.iplot(colorscale="polar", theme="white", asFigure=True,
                               yTitle="Credit Spread (bps)",
                               xTitle="Maturity (years)",)
    figure.update_layout(font=dict(family="Computer Modern"))
    figure.write_image("images/{}.pdf".format(image_name), format="pdf")
    figure.show()
