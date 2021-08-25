# imports
import numpy as np
import pandas as pd
import numpy.random as rnd
import cufflinks as cf  # library to use plotly plots along with pandas
import plotly.graph_objs as go
from tqdm import trange
from scipy.stats import norm
from utilities.UtilFunctions import gbm_generator, merton_minus_bc_squared
cf.go_offline(connected=True)

# define inputs
np.random.seed(0)
v0, D, T, sigma, r,  = 200, 100, 20, 0.25, 0.03
dt = 1 / 12
n_paths = 250_000
n_periods = int(T/dt)

paths_data = gbm_generator(v0, T, sigma, r, dt, n_paths)

# plot 10 randomly-chosen paths
rand_idx_plot = rnd.randint(low=0, high=n_paths-1, size=10)
paths_for_plot = paths_data[:, rand_idx_plot]
paths_for_plot_df = pd.DataFrame(paths_for_plot,
                                 columns=[f"p{i}" for i in range(1, 11)])

fig = paths_for_plot_df.iplot(colorscale="polar", xTitle="Time",
                              yTitle="Firm Value",
                              theme="white", asFigure=True)
# add debt threshold
fig.add_shape(type="line", line_color="LightSeaGreen", line_width=2,
               opacity=1, line_dash="dot", x0=0, x1=1, xref="paper", y0=D, y1=D)
fig.add_trace(go.Scatter(x=[23], y=[65], text=["Face Value of Debt"],
                         mode="text", showlegend=False))

fig.update_layout(font=dict(family="Computer Modern"))
fig.write_image("images/firm_value_simulated_paths_1.pdf", format="pdf")
fig.show()


### PD Merton
res_v_maturity = paths_data[-1, :] - D
pd_merton = len(res_v_maturity[res_v_maturity < 0]) / len(res_v_maturity)
print("estimated probability of default " 
      "under Merton model settings is:", round(pd_merton, 4))

### PD Black-Cox
# define default threshold considering progressive discount factor
discount_vec = np.array([np.exp(-dt * (i - 1) * r)
                         for i in range(n_periods + 1, 0, -1)])
Dt = D * discount_vec
# build array to check any default status in at least one period,  for each path
default_status = np.any(paths_data < Dt.reshape(-1, 1), axis=0)
pd_bc = len(default_status[default_status]) / len(default_status)
print("estimated probability of default "
      "under Black-Cox model settings is:", round(pd_bc, 4))


# GridSearch D between [0.25, 100] with 0.25 jumps to find
# min(merton_minus_bc_squared)
merton_minus_bc_squared_array = np.zeros(400)
for i in trange(400):
    D = (i + 1)/4
    Dt = D * discount_vec
    merton_minus_bc_squared_array[i] = \
        merton_minus_bc_squared(Dt_=Dt, pd_mert=pd_merton, paths=paths_data)

# show Face Value of Debt for which (p_merton - p_bc)^2 is minimum
merton_minus_bc_squared_df = pd.DataFrame(merton_minus_bc_squared_array,
                                              index=[(i+1) / 4 for i in range(400)])
print("Black-Cox - Face Value of Debt allowing equal"
      " default probabilities:", float(merton_minus_bc_squared_df.idxmin()))

