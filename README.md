# exoplanet-weather
 
This repository contains the Python pipeline used to systematically search for variability in the secondary eclipse depth of Kepler hot Jupiters in [Li and Shporer (2024)](https://ui.adsabs.harvard.edu/abs/2024AJ....167..245L/abstract), "A Search for Temporal Atmospheric Variability of Kepler and TESS Hot Jupiters". 

1. `hermes.py` is used to batch download Kepler and TESS light curves.
2. `seyfert.py` (SEcondarY eclipse FittER Tool) is used to fit the secondary eclipse depth in each Kepler quarter with Bayesian inference/MCMC. The secondary eclipse light curve model is in `util/exoplanets.py`.
3. `plot_seyfert.py` is used to plot the measured secondary eclipse and transit depths over time.