from __future__ import print_function, absolute_import, division
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import MaskedArray
from dataclasses import dataclass
from numba import njit
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from astropy.time import Time
from astropy.timeseries import BoxLeastSquares
from astropy.stats import sigma_clip as astropy_sigma_clip
from canislib.misc import print_bold, fstr_ratio

def gp_periodogram(t, y, yerr=None, min_period=1, max_period=70, start=None, sample_mcmc=False):
    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx
    import aesara_theano_fallback.tensor as tt
    from celerite2.theano import terms, GaussianProcess

    with pm.Model() as model:
        mean = pm.Normal('mean', mu=0.0, sd=10.0)
        
        """GP Parameters"""
        std = np.nanstd(y)
        rho_gp = pm.Uniform('rho_gp', lower=min_period, upper=max_period)
        sigma_gp = pm.Uniform('sigma_gp', lower=0, upper=2*std)
        if yerr is None:
            log_sigma_lc = pm.Normal('log_sigma_lc', 0, 10)
            sigma_lc = pm.Deterministic('sigma_lc', tt.exp(log_sigma_lc))
        else:
            sigma_lc = pm.Uniform('sigma_lc', lower=0, upper=np.nanmedian(yerr)*3)

        residuals = y - mean
        use_matern32 = False
        if use_matern32:
            print('!!!!!!!!!!!!!!!!!!! USING MATERN')
            kernel = terms.Matern32Term(
                sigma=sigma_gp,
                rho=rho_gp,
                #Q=1/np.sqrt(2),
            )
        else:
            print('NOT USING MATERN')
            kernel = terms.SHOTerm(
                sigma=sigma_gp,
                rho=rho_gp,
                Q=1/np.sqrt(2),
            ) 
        if yerr is None:
            gp = GaussianProcess(kernel, t=t, diag=sigma_lc**2)
        else:
            gp = GaussianProcess(kernel, t=t, diag=yerr**2 + sigma_lc**2)
        gp.marginal('gp', observed=residuals) # likelihood function

        if start is None:
            mapsoln = pmx.optimize(start=model.test_point, progress=True)
        else:
            mapsoln = pmx.optimize(start=start, progress=True)

        fits = dict(
            zip(
                ['gp_pred'],
                pmx.eval_in_model([gp.predict(residuals)], mapsoln),
            )
        )

        if sample_mcmc: 
            import platform
            import pymc3 as pm
            import time
            
            starttime = time.time()
            tune=1500
            draws=1000
            cores=2
            chains=2
            

            trace = pm.sample(
                tune=tune,
                draws=draws,
                start=mapsoln,
                # Parallel sampling runs poorly or crashes on macos
                cores=1 if platform.system() == "Darwin" else cores,
                chains=chains,
                target_accept=0.95,
                return_inferencedata=True,
                random_seed=[261136679, 261136680],
                init="adapt_full",
                progressbar = True,
            )

            print(f'Sampling MCMC (tune={tune}, draws={draws}, chains={chains}, cores={cores}) took {round((time.time()-starttime)/60)} minutes')
            return mapsoln, fits, trace
    
    return mapsoln, fits

def box_ls(t, y, period = None, min_period=1.0, max_period=15):
    """Run boxed least square periodogram"""
    min_period = min(period, min_period)
    try:
        t, y = remove_nans(t, y)
        t, y = sorted_by_time(t,y)
        t, y = sigma_clip(t, y, sigma=10)
        if period is None:
            period_grid = np.exp(np.linspace(np.log(min_period), np.log(max_period), 5000))
            duration_grid = np.exp(np.linspace(np.log(0.01), np.log(min_period * 0.99), 50))
        else:
            assert isinstance(period, float)
            period_grid = period # We know the period very accurately! Keep it this way.
            duration_grid = np.exp(np.linspace(np.log(0.01), np.log(min_period * 0.99), 50))
        bls = BoxLeastSquares(t, y)
        bls_power = bls.power(period_grid, duration_grid, oversample=20)
        index = np.argmax(bls_power.power)
        
        bls_period = bls_power.period[index]
        bls_t0 = bls_power.transit_time[index]
        bls_depth = bls_power.depth[index]
        bls_duration = bls_power.duration[index]

        transit_mask = bls.transit_mask(t, bls_period, bls_duration, bls_t0)

        return {'period': bls_period, 't0': bls_t0, 't14': bls_duration, 'bls_depth': bls_depth, 'transit_mask': transit_mask}
    except ValueError:
        print('Box ls failed. Make sure data has no nans')

def check_nans(t, *y):
    if np.sum(np.isnan(t)) > 0:
        print_bold('!!!!! check_nans: Warning, your data has nans in it !!!!!', color='red')
    else:
        for yi in y:
            if np.sum(np.isnan(yi)) > 0:
                print_bold('!!!!! check_nans: Warning, your data has nans in it !!!!!', color='red')
                break

def map_and_interpolate(t, *y, dt=None):
    """Remap data to make the cadence constant"""
    check_nans(t, y)
    if dt is None:
        dt = np.nanmedian(np.diff(t))
    #t_mapped = np.arange(len(t)) * dt
    assert len(y)==1
    y = y[0]
    t_mapped = np.arange(np.nanmin(t), np.nanmax(t) + dt, dt)
    y_mapped = np.interp(t_mapped, t, y)
    return t_mapped, y_mapped

def map_and_fill_gaps(t, y, dt_max=1, fill_value=1):
    tm, ym = map_and_interpolate(t, y)

    gaps = np.diff(t)
    gap_indices = np.where(gaps > dt_max)[0]
    for i in gap_indices:
        mask = np.logical_and(t[i] < tm, tm < t[i+1])
        ym[mask] = fill_value
    return tm, ym

def absolute_flux_to_ppt(y):
    if np.abs(np.median(y)) < 0.1:
        print_bold('canislib.TST: warning data might already be in relative flux/ppt', color='red')
    return (y/np.nanmedian(y) - 1) * 1000

def running_mean(y, window_size, function='mean'):
    """boxcar_smooth but not centered at a point. Allow even window sizes basically."""
    i = 0
    ym = []
    while i + window_size <= len(y):
        if function == 'mean':
            ym.append(np.nanmean(y[i:i+window_size]))
        if function == 'quadrature':
            ym.append(1/window_size * np.sqrt(np.sum(np.square(y[i:i+window_size]))))
        i += 1
    return np.array(ym)

def boxcar_smooth(y, window_size):
    """Smooth via median filter with certain window size in units of indices.
    Make sure to remove nans :-). 
    TODO: Speed up"""
    if window_size%2!= 1:
        raise ValueError('canislib.TST.boxcar_smooth: Window size must be odd')
    wing_size = int((window_size - 1)/2)
    y = np.pad(np.copy(y), pad_width=wing_size, mode='constant', constant_values=(np.nan))
    y_smoothed = []
    for i in range(wing_size, len(y) - wing_size):
        y_smoothed.append(np.mean(y[i-wing_size: i+wing_size+1]))
    return np.array(y_smoothed)

def find_slow_dips(y, window_size=5):
        """Find dips that are minima with respect to certain amount of neighboring points"""
        if window_size%2!= 1:
            raise ValueError('canislib.TST.find_slow_dips: Window size must be odd')
        wing_size = int((window_size - 1)/2)
        # Extrema near the border are negligible just ignore
        # y = np.pad(np.copy(y), pad_width=1, mode='constant', constant_values=(np.inf))
        dip_indices = []

        for i in range(wing_size, len(y)-wing_size):
            if y[i] == np.min(y[i-wing_size: i+wing_size+1]):
                dip_indices.append(i)
        return dip_indices

def compute_SDR(t=None, y=None, known_period=None, dip_spacings=None, do_rebin=False, do_smooth=True):
    """Perform boxcar smooth and then find position of dips to determine single/double ratio
    TODO: Speed up"""
    # Basri and Nguyen 2018 rebin Kepler data and perform boxcar smoothing
    if dip_spacings is None:
        if do_rebin:
            raise ValueError('canislib.TST.compute_SDR: rebinning not implemented yet')
        if do_smooth:
            window_size = int(known_period/8/(np.min(np.diff(t))))
            if window_size % 2 == 0:
                window_size += 1
            y = boxcar_smooth(np.copy(y), window_size=window_size)
        dip_indices = find_slow_dips(y, window_size=5)
        dip_times = t[dip_indices]
        dip_spacings = np.diff(dip_times)
    
    is_single_dip = dip_spacings > 0.75 * known_period
    sdr = np.log10(np.sum(is_single_dip) / len(is_single_dip))
    return sdr

def dict_to_tuple(data):
    """Dict to t,y, yerr tuples"""
    t, y, yerr = data['time'], data['flux'], data['flux_err']
    return t, y, yerr

def preprocess_transit_lc(t, y, yerr, orbital_period, transit_duration, transit_epoch,
                                    verbose=True, plot=True, savefig=None, skip_norm=False):
    """
    Deprecated
    0. Makes a copy of the data
    1. Removes nans                    
    2. Divides y and yerr by median
    3. Subtracts unity
    4. Masks eclipses
    5. σ=5 Sigma clips baseline data
    6. Fits GP and ubtracts all data with interpolated fit
    7. σ=5 Sigma clip again
    8. Manually remove transit data above 2σ above baseline
    """
    print_bold('Deprecated', color='red')
    texp = np.min(np.diff(t))
    transit_duration += 2 * texp # padding

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(12, 5), constrained_layout=True, sharex=True)
        ax[0][0].set_title('Raw data with eclipses masked')
        ax[0][1].set_title('Gaussian process fit')
        ax[1][0].set_title('Gaussian process subtracted')
        ax[1][1].set_title('Sigma clip')
        plt.suptitle(f'Preprocesssing baseline flux', fontweight='bold')
        if savefig is not None:
            os.makedirs(savefig, exist_ok=True)
    else:  
        ax = [([None]*2)]*2
    
    """Remove nans and mask eclipses"""
    if verbose:
        print_bold(f'Removing nans, median normalizing, subtracting one, and masking eclipses')
    t_raw, y_raw, yerr = np.copy(t), np.copy(y), np.copy(yerr)
    t_raw, y_raw, yerr = remove_nans(t_raw, y_raw, yerr, verbose=verbose)
    if not skip_norm:
        yerr /= np.median(y_raw)
        y_raw /= np.median(y_raw)
        y_raw -= 1
                            
    t_baseline, y_baseline, yerr_baseline = mask_eclipses(t_raw, y_raw, yerr, orbital_period=orbital_period,
                                            transit_duration=transit_duration, transit_epoch=transit_epoch,
                                            transit_multiplier=1, secondary_eclipse_multiplier=1, 
                                            return_masked_array=True, verbose=verbose)
    t_eclipse, y_eclipse, yerr_eclipse = get_masked(t_baseline, y_baseline, yerr_baseline)
    t_baseline, y_baseline, yerr_baseline = remove_masked(t_baseline, y_baseline, yerr_baseline)

    """Detrending"""
    if verbose:
        print_bold(f'Fitting Gaussian process')
    t_baseline, y_baseline, yerr_baseline = sigma_clip(t_baseline, y_baseline, yerr_baseline, sigma=5, 
                                                            plot=plot, ax=ax[0][0], verbose=verbose, s=16)
    
    # Sigma clipped data with trend[
    t_science = np.concatenate([t_baseline, t_eclipse])
    y_science = np.concatenate([y_baseline, y_eclipse])
    yerr_science = np.concatenate([yerr_baseline, yerr_eclipse])
    t_science, y_science, yerr_science = sorted_by_time(t_science, y_science, yerr_science)

    _, _, map_soln = detrend_with_GaussianModel(t_baseline, y_baseline,
                            log_rho_gp=None, log_sigma_gp=None,
                            return_map_soln=True, verbose=False, plot=plot, ax=[ax[0][1], ax[1][0]])
    t_gp = np.copy(t_baseline)
    y_gp = map_soln['pred']

    y_baseline_interp = np.interp(t_science, t_gp, y_gp)
    y_science -= y_baseline_interp

    t_science, y_science, yerr_science = mask_eclipses(t_science, y_science, yerr_science, orbital_period=orbital_period,
                                            transit_duration=transit_duration, transit_epoch=transit_epoch,
                                            transit_multiplier=1, secondary_eclipse_multiplier=1, 
                                            return_masked_array=True, verbose=verbose)
    t_eclipse, y_eclipse, yerr_eclipse = get_masked(t_science, y_science, yerr_science)
    t_science, y_science, yerr_science = remove_masked(t_science, y_science, yerr_science)

    t_science, y_science, yerr_science = sigma_clip(t_science, y_science, yerr_science, 
                                                    sigma=5, plot=plot, ax=ax[1][1], verbose=verbose)

    if savefig is not None:
        plt.savefig(f'{savefig}/1-detrending-baseline.png', dpi=300)
        plt.close(plt.gcf())

    """Try to remove bad eclipse values via manual sigma clip"""
    if verbose:
        print_bold(f'Manually clipping transit sections')
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True, sharex=True)
    eclipse_mask = y_eclipse < 5 * np.std(y_baseline_interp)
    t_eclipse_bad, y_eclipse_bad, yerr_eclipse_bad = (t_eclipse[~eclipse_mask], 
                                                      y_eclipse[~eclipse_mask], 
                                                      yerr_eclipse[~eclipse_mask])
    t_eclipse, y_eclipse, yerr_eclipse = (t_eclipse[eclipse_mask], 
                                           y_eclipse[eclipse_mask], 
                                           yerr_eclipse[eclipse_mask])

    bad = len(t_eclipse_bad)
    total = len(eclipse_mask)
    if verbose:
        print(f'Removed {bad}/{total}≈{round(bad/total*100, 1)}% of transit data above 5 sigma of baseline')                            

    t_science = np.concatenate([t_science, t_eclipse])
    y_science = np.concatenate([y_science, y_eclipse])
    yerr_science = np.concatenate([yerr_science, yerr_eclipse])
    t_science, y_science, yerr_science = sorted_by_time(t_science, y_science, yerr_science)

    if plot:
        ax.set_title(f'Manually clipping transit sections', fontweight='bold')
        ax.axhline(2 * np.std(y_baseline_interp), linewidth=1.5, color='red', label='Clip cutoff')
        ax.errorbar(t_eclipse_bad-np.min(t_science), y_eclipse_bad, yerr_eclipse_bad, ms=1, color='r', fmt='x', 
                    label=f'Manually clipped eclipse data ({fstr_ratio(bad, total)})')
        ax.errorbar(t_science-np.min(t_science), y_science, yerr_science, ms=1, color='k', fmt='o',
                        label=f'Final light curve ready for science ({fstr_ratio(len(t_science), len(y))} data points)')
        ax.legend()

        if savefig is not None:
            plt.savefig(f'{savefig}/2-transit-clipping.png', dpi=300)
            plt.close(plt.gcf())
    return t_science, y_science, yerr_science

def keplerized(t, y, interval_width=90):
    """Pythonic: sorted not sort"""
    """Keplerize according to procedure described in Li and Basri et al. 2023 :-)"""
    return t, flatten_intervals(t, y, interval_width=interval_width)


def keplerize_table(t, yspot=None, yfac=None, ynet=None, rotational_period=27, segment_width=400):
    """Emulate Kepler PDC-MAP by removing long term trends"""
    """ASSUME Y IS ALREADY IN PPT"""
    from canislib.timeseriestools import flatten_intervals, separate_intervals
    from canislib.basri import calculate_range_var
    from astropy.table import Table
    ynet_unflattened = np.copy(ynet)
    yfac_unflattened = np.copy(yfac)
    rows = []

    if yfac is None or yspot is None:
        nan_array = np.zeros(len(ynet)) * np.nan
        R_var_spot = np.nan
        R_var_fac = np.nan
        _, ynets_unflattened = separate_intervals(t, ynet_unflattened, interval_width=segment_width)
        ynet_flat = flatten_intervals(t, ynet, interval_width=200, deg=2)
        ts, ynets = separate_intervals(t, ynet_flat, interval_width=segment_width)
        for i in range(len(ts)):
            R_var_net = calculate_range_var(ynets[i])
            brightness = (np.median(ynets_unflattened[i])-np.median(ynet_unflattened))
            # print(np.median(ynets_unflattened[i]), np.median(ynet_unflattened))
            rows.append([ts[i], nan_array, nan_array, ynets[i], np.nan, np.nan, R_var_net, brightness])
    else:
        #yspot -= np.max(yspot)
        yfac -= np.min(yfac)

        """Convert absolute photometry lightcurve into what Kepler would see by removing long-term trends."""
        yfac_flat = flatten_intervals(t, yfac, interval_width=200, deg=2)
        ts, yfacs = separate_intervals(t, yfac_flat, interval_width=segment_width)
        _, yfacs_unflattened = separate_intervals(t, yfac, interval_width=segment_width)
        _, yspots = separate_intervals(t, yspot, interval_width=segment_width)
        for i in range(len(ts)):
            R_var_spot = calculate_range_var(yspots[i])
            R_var_fac = calculate_range_var(yfacs[i])
            R_var_net = calculate_range_var(yspots[i]+yfacs[i])
            brightness = (np.median(yfacs_unflattened[i])-np.median(yfac_unflattened))
            rows.append([ts[i], yspots[i], yfacs[i], yspots[i]+yfacs[i], R_var_spot, R_var_fac, R_var_net, brightness])
    tb = Table(rows=rows, names=('t', 'yspot', 'yfac', 'ynet', 'R_var_spot', 'R_var_fac', 'R_var_net', 'brightness'))
    tb.meta['rotational_period'] = rotational_period
    return tb

def detrend_with_gp(t, y, yerr):
    """
    Subtract the maximum a posteriori solution for a stochastic damped simple harmonic oscillator
    y must be stitched and normalized to relative flux otherwise LinAlgError
    """
    import pymc3 as pm
    import pymc3_ext as pmx
    import aesara_theano_fallback.tensor as tt
    from celerite2.theano import terms, GaussianProcess

    lc_std = np.nanstd(y)
    with pm.Model() as model:
        mean = pm.Normal('mean', mu=0.0, sd=10.0)
        log_sigma_lc = pm.Normal(
            "log_sigma_lc", mu=np.log(lc_std), sd=10
        )
        log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
        log_sigma_gp = pm.Normal(
            "log_sigma_gp", mu=np.log(lc_std), sd=10
        )
        pm.Deterministic('sigma_lc', tt.exp(log_sigma_lc))
        pm.Deterministic('rho_gp', tt.exp(log_rho_gp))
        pm.Deterministic('sigma_gp', tt.exp(log_sigma_gp))

        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            Q=1/np.sqrt(2),
        )

        residuals = y-mean
        gp = GaussianProcess(kernel, t=t, diag=yerr**2 + tt.exp(log_sigma_lc)**2)
        gp.marginal('gp', observed=residuals)
        pm.Deterministic('gp_pred', mean+gp.predict(residuals))
        
        map_soln = model.test_point
        map_soln = pmx.optimize(start=map_soln, vars=[mean],progress=False)
        map_soln = pmx.optimize(start=map_soln, vars=[mean, log_sigma_lc, log_rho_gp, log_sigma_gp],progress=False)
    return map_soln

def stitch_quarters(quarter_data, quarters='all', verbose=True, return_quality=False):
    """Input data['quarters']"""
    if len(quarters)==0 or quarters == 'all' or quarters == ['all']:
        quarters = np.arange(1, 18)
    ts = []
    ys = []
    yerrs = []
    qs = []
    available_quarters = sorted(list(quarter_data.keys()))

    for quarter in quarters:
        if quarter in available_quarters:
            ts.append(np.array(quarter_data[quarter]['time']))
            ys.append(np.array(quarter_data[quarter]['flux']))
            yerrs.append(np.array(quarter_data[quarter]['flux_err']))
            if return_quality:
                qs.append(np.array(quarter_data[quarter]['quality']))
            median = np.nanmedian(ys[-1])
            ys[-1] /= median
            yerrs[-1] /= median
        else:
            if verbose:
                print(f'stitch_quarters: Missing quarter {quarter}')

    if verbose:
        print(f'stitch_quarters: Tried to stich {len(quarters)} quarters {list(quarters)}')

    if len(ts) == 0:
        raise ValueError('Did not find any quarters. Double check what you inputted')

    if return_quality:
        return np.concatenate(ts), np.concatenate(ys), np.concatenate(yerrs), np.concatenate(qs)
    else:
        return np.concatenate(ts), np.concatenate(ys), np.concatenate(yerrs)

def stitch_sectors(sector_data, sectors='all', verbose=True, return_quality=False):
    """Same thing but for TESS sectors. Input data['sectors']."""
    if len(sectors) ==0 or sectors in ['all'] or sectors == ['all']:
        sectors = sorted(sector_data.keys())
    available_sectors = sorted(list(sector_data.keys()))

    ts = []
    ys = []
    yerrs = []
    qs = []

    for sector in sectors:
        if sector in available_sectors:
            ts.append(np.array(sector_data[sector]['time']))
            ys.append(np.array(sector_data[sector]['flux']))
            yerrs.append(np.array(sector_data[sector]['flux_err']))
            if return_quality:
                qs.append(np.array(sector_data[sector]['quality']))

            median = np.nanmedian(ys[-1])
            ys[-1] /= median
            yerrs[-1] /= median
            
            if np.sum(np.isnan(yerrs[-1])) == len(yerrs[-1]):
                print('stitch_sectors: All errorbars are nan, likely due to QLP. Replacing with zeros.')
                yerrs[-1] = np.zeros(len(yerrs[-1]))
        else:
            if verbose:
                print(f'stitch_sectors: Missing sector {sector}')

    if verbose:
        print(f'stitch_sectors: Tried to stitch {len(sectors)} sectors {list(sectors)}')

    if len(ts) == 0:
        raise ValueError('Did not find any sectors. Double check what you inputted')
    
    if return_quality:
        return np.concatenate(ts), np.concatenate(ys), np.concatenate(yerrs), np.concatenate(qs)
    else:
        return np.concatenate(ts), np.concatenate(ys), np.concatenate(yerrs)

def separate_into_continuous_sections(t, y, max_gap_width, max_jump_height, max_section_width=np.inf, min_section_size=1, verbose=True):
    """min_section_size: removes excess at end"""
    """Motivated by procedure from Shporer et. al 2011"""
    ts = []
    ys = []
    t_section = [t[0]]
    y_section = [y[0]]

    for i, (ti, yi) in enumerate(zip(t,y)):
        keep_going = True
        if ti - t_section[0] > max_section_width:
            keep_going = False
        elif ti - t_section[-1] > max_gap_width:
            keep_going=False
        elif np.abs(yi-y[i-1]) > max_jump_height:
            keep_going=False
            if i != len(t) - 1:
                # Almost never two outliers in a row
                if np.abs(y[i+1]-yi) > max_jump_height or np.abs(y[i-2]-y[i-1]) > max_jump_height:
                    keep_going=True
        if keep_going:
            t_section.append(ti)
            y_section.append(yi)
        else:
            if len(t_section) > min_section_size:
                ts.append(np.array(t_section))
                ys.append(np.array(y_section))
            t_section = [ti]
            y_section = [yi]
    """Add any leftovers"""
    if len(t_section) >= min_section_size and len(t_section) > 1: # nuanced - Jun 20.
        ts.append(np.array(t_section))
        ys.append(np.array(y_section))
    
    if verbose:
        durations = [(t_section[-1]-t_section[0]) for t_section in ts]
        print(f'Separated timeseries into {len(ts)} continuous sections with average duration {np.average(durations)}')
    return ts, ys

def lightkurve_to_dict(lc, return_quality=True):
    """Converts flux and flux_err to floats but keeps time metadata"""
    import lightkurve as lk
    assert issubclass(type(lc), lk.LightCurve)
    time = np.ascontiguousarray(lc.time.value, dtype=np.float64)
    flux = np.ascontiguousarray(lc.flux, dtype=np.float64)
    flux_err = np.ascontiguousarray(lc.flux_err, dtype=np.float64)
    data = {'time': time, 'flux': flux, 'flux_err': flux_err}

    if return_quality:
        quality = np.ascontiguousarray(lc.quality, dtype=int)
        data['quality'] = quality

    return data

def get_transit_mask(t, *y, t0=None, period=None, t14=None, ephemeris=None):
    if ephemeris is not None:
        t0=ephemeris['t0']
        period=ephemeris['period']
        t14=ephemeris['t14']
    texp = np.min(np.diff(t))
    t_rel_transit = np.abs((t-t0)%period)
    t14_eff = t14 + texp*2
    transit_mask = np.logical_or(t_rel_transit < t14_eff/2, t_rel_transit > period - t14_eff/2)

    return transit_mask

def get_se_mask(t, *y, t0=None, period=None, t14=None, ephemeris=None):
    if ephemeris is None:
        return get_transit_mask(t, *y, t0=t0+period/2, period=period, t14=t14)
    else:
        import copy
        ephemeris = copy.deepcopy(ephemeris)
        ephemeris['t0'] += ephemeris['period']/2
        return get_transit_mask(t, *y, ephemeris=ephemeris)

def mask_occultations(t, *y, t0=None, period=None, t14=None, ephemeris=None, t14_scalefactor=1.0,
                    verbose=True, return_masked_array=True, return_mask=False):
    """
    Ignores light time delay 2a/c
    Accounts for smearing effect from nonzero exposure time
    """
    if ephemeris is not None:
        t0 = ephemeris['t0']
        t14 = ephemeris['t14']
        period = ephemeris['period']
    else:
        if not (t0 and t14 and period):
            raise ValueError('provide ephemeris')

    texp = np.min(np.diff(t))
    t_rel_transit = np.abs((t-t0)%period)
    t_rel_se = np.abs((t-t0-period/2)%period)
    t14_eff = t14 *t14_scalefactor + texp*2
    transit_mask = np.logical_or(t_rel_transit < t14_eff/2, t_rel_transit > period - t14_eff/2)
    secondary_eclipse_mask = np.logical_or(t_rel_se < t14_eff/2, t_rel_se > period - t14_eff/2)
    eclipse_mask = np.logical_or(transit_mask, secondary_eclipse_mask)

    if verbose:
        mask_count = np.sum(eclipse_mask)
        mask_percent = round(100*np.sum(eclipse_mask)/len(t), 1)
        verb = 'Masked' if return_masked_array or return_mask else 'Removed'
        print(f'canislib.TST.mask_eclipses: {verb} {mask_count}/{len(t)}≈{mask_percent}% of data')

    if return_mask:
        return eclipse_mask
    if return_masked_array:
        return MaskedArray(t, mask=eclipse_mask), *[MaskedArray(yi, mask=eclipse_mask) for yi in y] 
    else:
        return t[np.invert(eclipse_mask)],*[yi[np.invert(eclipse_mask)] for yi in y] 

def detrend_with_polynomial(t, *y, order=5, return_fit=False):
    """
    Assume y_raw is the dot product of y_signal and y_trend. 
    However, fitting and dividing by a trend gives y_signal normalized"""
    import numpy.polynomial.polynomial as poly
    coeffs = poly.polyfit(t, y[0], order)
    r_trend = poly.polyval(t, coeffs)
    if return_fit:
        return t, *[yi/r_trend for yi in y], r_trend, coeffs
    else:
        return t, *[yi/r_trend for yi in y]

def sorted_by_time(t, *y):
    """Sort timeseries chronologically"""
    assert isinstance(t,np.ndarray) and np.logical_and.reduce([isinstance(yi,np.ndarray) for yi in y])
    idx = np.argsort(np.copy(t))
    return t[idx], *[yi[idx] for yi in y]

def phase_fold(t, *y, t0=None, period=None, ephemeris=None):
    """Phase fold relative to given epoch. If epoch is transit epoch, secondary eclipse is at phase 0.5"""
    if ephemeris is not None:
        t0 = ephemeris['t0']
        period = ephemeris['period']
    return sorted_by_time((t- t0) % period / period, *y)

def unmask_all(t, *y):
    """Ignore mask and extract all values from a np.ma.MaskedArray"""
    assert isinstance(t, MaskedArray)
    if len(y) > 0:
        isinstance(y[0], MaskedArray)
        assert np.array_equal(t.mask, y[0].mask)
    return t.data, *[yi.data for yi in y]

def get_masked(t, *y):
    """Extract masked values from a np.ma.MaskedArray"""
    assert isinstance(t, MaskedArray)
    if len(y) > 0:
        isinstance(y[0], MaskedArray)
        assert np.array_equal(t.mask, y[0].mask)
    masked = t.mask
    return t.data[masked], *[yi.data[masked] for yi in y]


def remove_masked(t, *y):
    """Extract unmasked values from a np.ma.MaskedArray"""
    assert isinstance(t, MaskedArray)
    if len(y) > 0:
        isinstance(y[0], MaskedArray)
        assert np.array_equal(t.mask, y[0].mask)
    unmasked = np.invert(t.mask)
    if len(y) > 0:
        return t.data[unmasked], *[yi.data[unmasked] for yi in y]
    return t.data[unmasked]

def remove_nans(t, *y, verbose=True):
    nan_mask = np.logical_or.reduce([np.isnan(t), *[np.isnan(yi) for yi in y]])
    unmasked = np.invert(nan_mask)
    if verbose:
        nan_count = np.sum(nan_mask)
        print(f'canislib.TST.remove_nans: Removed {nan_count}/{len(t)}≈{round(nan_count/len(t)*100, 1)}% nan values from original data')
    if len(y) == 0:
        return np.array(t)[unmasked]
    else:
        return np.array(t)[unmasked], *[np.array(yi)[unmasked] for yi in y]

def remove_infs(t, *y, verbose=True):
    inf_mask = np.logical_or.reduce([np.isinf(t), *[np.isinf(yi) for yi in y]])
    unmasked = np.invert(inf_mask)
    if verbose:
        inf_count = np.sum(inf_mask)
        print(f'canislib.TST.remove_infs: Removed {inf_count}/{len(t)}≈{round(inf_count/len(t)*100, 1)}% inf values from original data')
    return t[unmasked], *[yi[unmasked] for yi in y]

def sigma_clip(t, *y, sigma=None, return_masked_array=False,
               verbose=True, plot=False, ax=None, s=4, return_mask=False, **plotting_kwargs):
    """Supports putting in masked arrays"""
    if sigma is None:
        raise ValueError('Hey, you\'re supposed to pass sigma as a kwarg')
    initial_masked_count = 0
    if verbose and isinstance(y[0], np.ma.MaskedArray):
        initial_masked_count = np.sum(y[0].mask)
        print(f'canislib.TST.sigma_clip: {round(100*initial_masked_count/len(t), 1)}% of initial data is masked. Ignoring them and keeping them masked')

    clipped_mask = astropy_sigma_clip(y[0], sigma=sigma, masked=True).mask
    unclipped_mask = np.invert(clipped_mask)
    num_clipped = np.sum(clipped_mask) - initial_masked_count
    num_total = len(t) - initial_masked_count
    percentage = round(num_clipped/num_total*100, 1)

    if verbose:
        print(f'canislib.TST.sigma_clip: {sigma}σ clipped {num_clipped}/{num_total}≈{percentage}% of the unmasked data')

    if plot:
        if ax is None:
            plt.figure(figsize=(6,4), constrained_layout=True)
            ax = plt.gca()
        plt.sca(ax)
        plt.scatter(t, y[0], s=s, color='k', label='data', **plotting_kwargs)
        plt.scatter(t[clipped_mask], y[0][clipped_mask], s=40, color='r', marker='x', label=f'Outliers ({sigma}σ): {percentage}%')
        plt.legend(frameon=True)
    
    if return_mask:
        return unclipped_mask
    if return_masked_array:
        return MaskedArray(t, mask=clipped_mask), *[MaskedArray(yi, mask=clipped_mask) for yi in y] 
    else:
        return t[unclipped_mask],*[yi[unclipped_mask] for yi in y] 

# def unpack_lightkurve(lc, convert_to_jd=False):
#     """Option to converts to julian day"""
#     if convert_to_jd:
#         time = Time(lc.time).to_value('jd')
#     else: 
#         time = lc.time.value
#     flux = lc.flux
#     flux_err = lc.flux_err

#     # We want the unit to be relative flux not ppt for exoplanet to work.
#     x = np.ascontiguousarray(time, dtype=np.float64)
#     y = np.ascontiguousarray(flux, dtype=np.float64)
#     yerr = np.ascontiguousarray(flux_err, dtype=np.float64)

#     return {'time': x, 'flux': y, 'flux_err': yerr}

def gaussian_smooth(y, window_size=56, fwhm=18):
    """Uses gaussian filter NOT savitzky golay"""
    sigma = fwhm/2.355
    truncate=window_size/sigma
    y_smoothed = gaussian_filter(y, sigma=sigma, truncate=truncate)
    return y_smoothed

def compute_MAD(y):
    return np.nanmedian((np.abs(y-np.nanmedian(y))))

def cross_correlation(x, y):
        result = np.correlate(x, y, mode='full')
        return result[result.size//2:]

def lombscargle(t, y, min_period=1, max_period=50, compute_min_period=1, compute_max_period=100, plot=False, ax=None, verbose=True, return_LS=False, label_prefix='', dotcolor=None, **plotting_kwargs):
    """
    Uses astropy implementation https://docs.astropy.org/en/stable/timeseries/lombscargle.html
    Specify min and max period
    Plot=True does not create a new figure
    """
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y)
    freq, power = ls.autopower(minimum_frequency=1/max_period, 
                               maximum_frequency=1/min_period, 
                               samples_per_peak=20)
    
    mask = np.logical_and(min_period<1/freq, 1/freq<max_period)
    period = np.float(1/freq[mask][np.nanargmax(power[mask])])
    if verbose:
        print(f'Lomb Scargle period: {period}, FAP: {ls.false_alarm_probability(power.max())}')
    if plot:
        if ax is None:
            fig,ax = plt.subplots(1,1, figsize=(8,3))
        
        ax.plot(1/freq, power, **plotting_kwargs)
        if dotcolor is None:
            ax.scatter(period, np.nanmax(power[mask]), s=50, label=f'{label_prefix}{round(period, 3)} days')
        else:
            ax.scatter(period, np.nanmax(power[mask]), s=50, label=f'{label_prefix}{round(period, 3)} days', color=dotcolor)
    if return_LS:
        return period, freq, power
    return period

@njit 
def compute_ACF(y):
    """
    See section 2.1 of McQuillan et al. 2013 https://arxiv.org/pdf/1303.6787.pdf
    or see the IDL documentation (what Prof. Basri uses) https://www.l3harrisgeospatial.com/docs/a_correlate.html
    """
    N = len(y) 
    r = np.zeros(N)
    ybar = np.mean(y)
    SSTo = 0
    for i in range(0, N):
        SSTo += (y[i] - ybar)**2
    if SSTo == 0: # values are all zero
        return None
        
    for k in range(0, N):
        r_k = 0 
        for i in range(0, N-k):
            r_k += (y[i]-ybar) * (y[i+k]-ybar)
        r[k] = (r_k/SSTo)
    return r

class PeakNotFoundException(Exception):
    pass

def _identify_rotational_period(lags, r, smoothing=(56,18), min_lph = 0.1,
        ax=None, verbose=False, version=2014, is_not_spot=True, use_dumb_method=False):
    """
    See section 2.1 of McQuillan et al. 2013 https://arxiv.org/pdf/1303.6787.pdf
    or McQuillan et al. 2014
    """
    lags = np.copy(lags)
    r = np.copy(r)

    """Smooth ACF"""
    if smoothing is not None:
        window_size, fwhm = smoothing
        sigma = fwhm/2.355
        truncate=window_size/sigma
        r_smoothed = gaussian_filter(r, sigma=sigma, truncate=truncate)
    else:
        r_smoothed = np.copy(r)
    
    """Find ACF peaks and compute height, defined as mean distance to two adjacent minima"""
    indices, is_peak = find_peaks_and_valleys(r_smoothed)

    has_a_peak_at_27 = False

    peakperiods = []
    peakheights = []
    for i in range(len(indices)-1):
        if is_peak[i]:
            peakperiods.append(lags[indices[i]])

            if np.abs(peakperiods[-1]/27 -1) < 0.1:
                has_a_peak_at_27 = True
            peakheights.append(0.5 * ((r_smoothed[indices[i]]-r_smoothed[indices[i-1]]) + 
                                      (r_smoothed[indices[i]]-r_smoothed[indices[i+1]])))
    peakperiods = np.array(peakperiods); peakheights = np.array(peakheights);
    if use_dumb_method:
        mask = np.logical_and(1 < peakperiods, peakperiods < 70)
        if np.sum(mask) < 1: raise PeakNotFoundException('Did not find a peak between 1 and 70')
        i = np.argmax(peakheights[mask])
        dominant_period = peakperiods[mask][i]
        lph = peakheights[mask][i]
        if lph < min_lph:
            raise PeakNotFoundException('Dominant peak has LPH less than min_lph')
        first_two_peaks_info = None
    else:
        if len(peakheights) <= 1:
            raise PeakNotFoundException('Did not find more than two peaks')

        if is_not_spot and max(peakheights[0], peakheights[1]) < min_lph:
            raise PeakNotFoundException('Dominant peak has LPH less than min_lph')
        
        """Use the bigger of the first/second peak as the dominant peak"""
        first_two_peaks_info = (peakperiods[0], peakheights[0], peakperiods[1], peakheights[1])
        dominant_period = peakperiods[0] if peakheights[0] > peakheights[1]  else peakperiods[1]
        lph = max(peakheights[0], peakheights[1])
    if verbose:
        print(f'Dominant period: {dominant_period}')

    """Search for peaks near harmonics of the dominant peak"""
    harmonic_periods = []
    harmonic_period_heights = []
    
    for peakperiod, peakperiodheight in zip(peakperiods, peakheights):
        if version==2013 and len(harmonic_periods) == 10:
            break
        if version==2014 and len(harmonic_periods) == 4:
            break
        nearest_harmonic = round(peakperiod/dominant_period) * dominant_period
        if nearest_harmonic - 0.2 * dominant_period <= peakperiod <= nearest_harmonic + 0.2*dominant_period:
            if len(harmonic_periods)>0:
                if peakperiod - harmonic_periods[-1] > 0.4 * dominant_period:
                    harmonic_periods.append(peakperiod)
                    harmonic_period_heights.append(peakperiodheight)
            else:
                harmonic_periods.append(peakperiod)
                harmonic_period_heights.append(peakperiodheight)
    
    if len(harmonic_periods) == 0:
        raise ValueError("No peaks found")

    if verbose:
        print(f'Peaks that are harmonics of dominant period: {harmonic_periods}')

    """Estimate rotational period from harmonic peaks"""
    rotational_period_estimate = None
    rotational_period_uncertainty = None

    if version==2013:
        if verbose:
            print('Using ACF procedure described in McQuillan et al. 2013') 
        period_intervals = np.diff(harmonic_periods)
        rotational_period_estimate = np.median(period_intervals)
        """ See https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation"""
        median_absolute_deviation = np.mean(np.abs(period_intervals-rotational_period_estimate))
        rotational_period_uncertainty = 1.483 * median_absolute_deviation/np.sqrt(len(harmonic_periods)-1)
    elif version==2014:
        if verbose:
            print('Using ACF procedure described in McQuillan et al. 2014') 
        y = np.concatenate([[0], harmonic_periods])
        x = np.round(y/dominant_period) # nearest peak number
        lrr = linregress(x, y, alternative='greater')
        rotational_period_estimate = lrr.slope
        rotational_period_uncertainty = lrr.stderr

    """Plot"""
    if ax is not None:
        if verbose:
            ax.axvline(peakperiods[0], color='r', linestyle='--')
            ax.axvline(peakperiods[1], color='r', linestyle='--')
        if verbose:
            for i in range(1,11):
                plt.axvspan(xmin=dominant_period * i - 0.20*dominant_period,
                            xmax=dominant_period * i + 0.20*dominant_period, alpha=0.3, color='orange')
            plt.scatter(harmonic_periods, harmonic_period_heights, s=50, color='orange', zorder=1)

    return rotational_period_estimate, rotational_period_uncertainty, lph, r_smoothed, first_two_peaks_info, has_a_peak_at_27

def find_peaks_and_valleys(r):
    """Returns the indices of each peak and a boolean array of whether it is peak or valley."""
    """Try using a Gaussian smooth before using this function"""
    indices = []
    is_peak = []
    for i in range(1, len(r)-1):
        if (r[i]-r[i-1]) * (r[i+1]-r[i]) < 0:
            indices.append(i)
            is_peak.append(r[i] - r[i-1] > 0)
    return np.array(indices), np.array(is_peak)

def ACF_McQuillan(t, y, min_lph=0.01, plot=False, verbose=True, ax=None, version=2014, 
                    return_just_period=False, is_not_spot=True, suppress_errors=False, use_dumb_method=False):
    """
    Returns lags, r (autocorrelation coefficients), and rotational period
    Centers by subtracting mean first

    min_lph: Min local peak height

    Identical to
    1) McQuillan et al. 2013 https://arxiv.org/pdf/1303.6787.pdf
    2) IDL A_CORRELATE https://www.l3harrisgeospatial.com/docs/a_correlate.html
    """
    assert np.sum(np.isnan(y)) == 0
    if np.max(np.diff(t)) - np.min(np.diff(t)) > 1e-3:
        print_bold('!!! Warning: data is not evenly spaced. Need to interpolate !!!', color='red')

    if plot:
        if ax is None:
            plt.figure(figsize=(6,4), constrained_layout=True)
            ax = plt.gca()
        plt.sca(ax)
    
    """Subtract mean from y and compute ACF"""
    cadence = np.min(np.diff(t))
    y = np.copy(y)
    y -= np.nanmean(y)

    N = len(y)
    lag = t[1]-t[0]

    r = compute_ACF(y)
    if r is None:
        if verbose:
            print(f'ACF failed! Check if data is zero')
        if return_just_period:
            return np.nan, np.nan
        else:
            return None, None, np.nan, np.nan, (np.nan, np.nan, np.nan, np.nan), False
    r /= np.max(r)

    """Search for rotational period using algorithm described in 2013/2014"""
    lags = lag * range(0, N)
    flag = 'default'
    try:
        if cadence >= 0.499: 
            (rotational_period, rotational_period_uncertainty, lph ,
            r_smoothed, first_two_peaks_info, has_a_peak_at_27) = _identify_rotational_period(
                        lags, r, min_lph=min_lph, smoothing=(56, 2), ax=ax, is_not_spot=is_not_spot, verbose=verbose, version=version, use_dumb_method=use_dumb_method)
        elif cadence >= 0.2499: # MPI data, minimal smoothing. Add 0.05 padding because float errors
            (rotational_period, rotational_period_uncertainty, lph,
            r_smoothed, first_two_peaks_info, has_a_peak_at_27) = _identify_rotational_period(
                        lags, r, min_lph=min_lph, smoothing=(56, 3), ax=ax, is_not_spot=is_not_spot, verbose=verbose, version=version, use_dumb_method=use_dumb_method)
        else:
            (rotational_period, rotational_period_uncertainty, lph,
                r_smoothed, first_two_peaks_info, has_a_peak_at_27) = _identify_rotational_period(
                                lags, r, min_lph=min_lph, smoothing=(56,18), ax=ax, is_not_spot=is_not_spot, verbose=verbose, version=version, use_dumb_method=use_dumb_method)
        if np.isnan(rotational_period):
            raise PeakNotFoundException
    except PeakNotFoundException:
        try:
            if verbose:
                print(f'Could not find peaks above {min_lph}. Trying no smoothing')
            (rotational_period, rotational_period_uncertainty, lph,
                r_smoothed, first_two_peaks_info, has_a_peak_at_27) = _identify_rotational_period(
                            lags, r, min_lph=min_lph, smoothing=None, ax=ax, is_not_spot=is_not_spot, verbose=verbose, version=version, use_dumb_method=use_dumb_method)
            if np.isnan(rotational_period):
                raise PeakNotFoundException
            else:
                flag = 'no smoothing'
        except PeakNotFoundException:
            try:
                if verbose:
                    print(f'Still could not find peaks above {min_lph}. Trying hard smoothing')
                (rotational_period, rotational_period_uncertainty, lph,
                    r_smoothed, first_two_peaks_info, has_a_peak_at_27) = _identify_rotational_period(
                                lags, r, min_lph=min_lph, smoothing=(100, 60), ax=ax, is_not_spot=is_not_spot, verbose=verbose, version=version, use_dumb_method=use_dumb_method)
                if np.isnan(rotational_period):
                    raise PeakNotFoundException
                else:
                    flag = 'hard smoothing'
            except PeakNotFoundException:
                if not suppress_errors:
                    print_bold('ACF-MMA failure', color='red')
                rotational_period = np.nan
                rotational_period_uncertainty = np.nan
                lph = np.nan
                has_a_peak_at_27 = False
                first_two_peaks_info = None
                flag = 'failure'
 
    if verbose:
        print(f'McQuillan period: {rotational_period}±{rotational_period_uncertainty}')
    
    """Plot"""
    if plot:
        plt.plot(lags, r, alpha=0.5, color='k', linestyle='-.')
        if flag != 'failure':
            plt.plot(lags, r_smoothed, color='k')
        if not np.isnan(rotational_period):
            plt.axvline(rotational_period, color='red', zorder=1000)
            plt.axvline(27, linestyle='-.')
            # plt.scatter(rotational_period,r[int(rotational_period/lag)], s=50, 
            #     label=f'{round(rotational_period, 3)} days', zorder=1)
    
    if return_just_period: 
        return rotational_period, rotational_period_uncertainty
    else:
        results = {}
        results['lags'] = lags
        results['r_smoothed'] = r_smoothed if flag != 'failure' else r
        results['P_rot'] = rotational_period
        results['P_rot_err'] = rotational_period_uncertainty
        results['lph'] = lph
        results['ACF'] = (np.array(lags), np.array(r))
        results['has_a_peak_at_27d'] = has_a_peak_at_27
        results['first_two_peaks_info'] = first_two_peaks_info
        results['flag'] = flag
        return results

def find_max_index_in_interval(t, y, min, max):
        t = np.array(t)
        y = np.array(y)
        return t[np.where(y==np.max(y[np.logical_and(min<=t, t<=max)]))[0][0]] # ΩBRAIN


def ACF_numpy(t, y, min=1, max=1000,plot=False, plot_dompeak = False, **plotting_kwargs):
    """
    Subtracts mean first. Returns lag in time and normalized acf
    https://raw.githubusercontent.com/bmorris3/interp-acf/master/interpacf/interpacf.py
    Plot=True does not create a new figure"""
    # Calculate the grid of "lags" in units of ``times``
    y = np.copy(y)
    y -= np.mean(y)
    dt = np.median(np.diff(t))
    lag = dt*np.arange(len(t))

    # Compute the autocorrelation function on interpolated fluxes
    acf = cross_correlation(y, y)
    acf /= np.max(acf)
    
    period = dominant_period(lag, acf, min=min, max=max)
    #naiveperiod = find_max_index_in_interval(lag, acf, min, max)
    print(f'Autocorrelation period: {period}')
    if plot:
        plt.plot(lag, acf /np.max(acf), **plotting_kwargs)
        if plot_dompeak:
            plt.plot(period,(acf/np.max(acf))[int(period/dt)], 'o', label=f'{round(period, 3)} days')

    return lag, acf, period

autocorrelation = ACF_numpy # alias!

def gradient(y):
    """Compute gradient of consecutive elements"""
    grad = y - np.roll(y, 1)
    grad[0] = grad[1] # zeroth indice is nonsense
    return grad

def bin_by_time(t, *y, bin_width, method=['mean', 'errormean']): # Reimplement this to make y a variable number of arguments
    """Bin data with specified time intervals
    If passing multiple y, e.g y and yerr, specify how to combine bin data via `method` (mean, median, errormean)
    """
    t, *y = sorted_by_time(t,*y)
    total_duration = t[-1] - t[0]
    num_bins = int(np.ceil(total_duration/bin_width))
    t_binned = t[0] + bin_width/2 + bin_width * np.arange(num_bins)
    ys_binned = [[] for _ in range(len(y))]
    j = 0

    float_tolerance = 1e-3
    binsizes = []

    for i in range(num_bins):
        ys_bin_i = [[] for _ in range(len(y))]
        while j < len(t) and np.abs(t[j] - t_binned[i]) <= bin_width/2 + float_tolerance:
            for k in range(len(y)):
                ys_bin_i[k].append(y[k][j])
            j+=1
        for k in range(len(y)):
            if (bin_size:=len(ys_bin_i[k])) > 0:
                if method[k] in ['mean']:
                    ys_binned[k].append(np.nanmean(ys_bin_i[k]))
                elif method[k] in ['median']:
                    ys_binned[k].append(np.nanmedian(ys_bin_i[k]))
                elif method[k] in ['errormean']: # error propagation
                    ys_binned[k].append(1./bin_size * np.sqrt(np.sum(np.square(ys_bin_i[k]))))
            else:
                ys_binned[k].append(np.nan)
            binsizes.append(len(ys_bin_i[k]))

    binsizes = np.array(binsizes)
    print('median nonzero bin size', np.median(binsizes[binsizes>0]))
    print('max bin size', np.max(binsizes))
                
    t_binned = np.array(t_binned)
    return remove_nans(t_binned, *[np.array(yi_binned) for yi_binned in ys_binned])

def bin_by_size(y, binsize = 10):
    """Use this separately on y and t"""
    yb = []
    i = 0
    while i < len(y) - binsize:
        yb.append(np.mean(y[i:i+binsize]))
        i += binsize
    yb.append(np.mean(y[i:]))
    return np.array(yb)

def bin_by_size_with_err(t, y, yerr, binsize = 10):
    """Use this separately on y and t"""
    tb = []
    yb = []
    yerrb = []
    i = 0
    while i <= len(y) - 2 * binsize:
        tb.append(np.mean(t[i:i+binsize]))
        yb.append(np.mean(y[i:i+binsize]))
        yerrb.append(1/binsize * np.sqrt(np.sum(np.square(yerr[i:i+binsize]))))
        i += binsize
    tb.append(np.mean(t[i:]))
    yb.append(np.mean(y[i:]))
    yerrb.append(1/binsize * np.sqrt(np.sum(np.square(yerr[i:]))))
    tb = np.array(tb); yb = np.array(yb); yerrb = np.array(yerrb)
    return tb, yb, yerrb

def linear_interpolation(t, y, n= 10000):
    """Get higher resolution data with linear interpolation"""
    t_interp = np.linspace(np.min(t), np.max(t), n)
    return t_interp, np.interp(t_interp, t, y)

def flatten(t, y, deg=2):
    """Remove a n-degree polynomial trend"""
    fit = np.poly1d(np.polyfit(t, y, deg))
    return y - fit(t)

def flatten_intervals(t, y, interval_width=200, deg=2):
    """
    Flatten consecutive intervals of data with flatten(),
    interval_width must be in same units as t
    """
    n = len(t)
    dt = t[1] - t[0] # assume constant dt
    size_interval = int(interval_width/dt)
    y_flat = []
    i = 0
    while i + size_interval < n:
        y_flat = np.concatenate([y_flat, flatten(t[i:i+size_interval], y[i:i+size_interval], deg=deg)])
        i += size_interval
    y_flat = np.concatenate([y_flat, flatten(t[i:], y[i:], deg=deg)])
    return y_flat

def separate_intervals(t, *y, interval_width=None):
    """Bin by fixed time intervals"""
    """interval_width must be in same units as t"""
    n = len(t)
    dt = t[1] - t[0] # assume constant dt
    size_interval = int(interval_width/dt)
    ts = []
    ys = []
    i = 0
    for i in range(len(y)):
        ys.append([])
    while i + size_interval < n:
        ts.append(t[i:i+size_interval])
        for j, x in enumerate(y):
            ys[j].append(x[i:i+size_interval])
        i += size_interval
    # add remaining
    ts.append(t[i:])
    for j, x in enumerate(y):
        ys[j].append(x[i:])
    return ts, *ys

def dt64_to_elapsed_days(dates, t0=None):
    """Convert list of datetime64s to elapsed days since first element"""
    if isinstance(dates, np.timedelta64):
        return dates/np.timedelta64(1, 'D')
    if isinstance(dates[0], np.timedelta64):
        return dates/np.timedelta64(1, 'D')
    if isinstance(dates[0], np.datetime64):
        if t0 is None:
            t0 = dates[0]
        return np.array([(date-t0)/np.timedelta64(1, 'D') for date in dates]) 
    return dates

"""Add bkjd and btjd time format to astropy.time
https://github.com/lightkurve/lightkurve/blob/382fd3a01efdb8a48dba155407081c4d28b542ad/src/lightkurve/time.py#L13"""
"""Adds the BKJD and BTJD time format for use by Astropy's `Time` object.

Caution: AstroPy time objects make a distinction between a time's format
(e.g. ISO, JD, MJD) and its scale (e.g. UTC, TDB).  This can be confusing
because the acronym "BTJD" refers both to a format (TJD) and to a scale (TDB).

Note: the classes below derive from an AstroPy meta class which will automatically
register the formats for use in AstroPy Time objects.
"""
from astropy.time.formats import TimeFromEpoch

class TimeBKJD(TimeFromEpoch):
    """
    Barycentric Kepler Julian Date (BKJD): days since JD 2454833.0.

    For example, 0 in BTJD is noon on January 1, 2009.

    BKJD is the format in which times are recorded in data products from
    NASA's Kepler Space Telescope, where it is always given in the
    Barycentric Dynamical Time (TDB) scale by convention.
    """
    name = 'bkjd'
    unit = 1.0
    epoch_val = 2454833
    epoch_val2 = None
    epoch_scale = 'tdb'
    epoch_format = 'jd'

class TimeBJD0(TimeFromEpoch):
    """
    """
    name = 'bkjd'
    unit = 1.0
    epoch_val = 2400000
    epoch_val2 = None
    epoch_scale = 'tdb'
    epoch_format = 'jd'

class TimeBTJD(TimeFromEpoch):
    """
    Barycentric TESS Julian Date (BTJD): days since JD 2457000.0.

    For example, 0 in BTJD is noon on December 8, 2014.

    BTJD is the format in which times are recorded in data products from
    NASA's Transiting Exoplanet Survey Satellite (TESS), where it is
    always given in the Barycentric Dynamical Time (TDB) scale by convention.
    """
    name = 'btjd'
    unit = 1.0
    epoch_val = 2457000
    epoch_val2 = None
    epoch_scale = 'tdb'
    epoch_format = 'jd'

class TimeBJD(TimeFromEpoch):
    """
    Barycentric  Julian Date (BJD)
    """
    name = 'bjd'
    unit = 1.0
    epoch_val = 0
    epoch_val2 = None
    epoch_scale = 'tdb'
    epoch_format = 'jd'

def convert_time(t, f1, f2, ts1='tdb', ts2='tdb'):
    """Convert time from one format to another"""
    t = np.copy(t)
    from astropy.time import Time
    t = Time(t, scale=ts1, format=f1)
    if ts2 != ts1:
        if ts2 == 'tdb':
            t = t.tdb
        elif ts2 == 'utc':
            t = t.utc
    t.format = f2
    return t.value