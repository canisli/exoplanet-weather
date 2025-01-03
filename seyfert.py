#!/usr/bin/env python
""" SEYFERT: SEcondarY eclipse FittER Tool"""

"""Limit numpy cores"""
import os, socket
num_cores = 2
print(f'Setting max cores to {num_cores}')
os.environ.update(OMP_NUM_THREADS=f'{num_cores}', OPENBLAS_NUM_THREADS=f'{num_cores}', 
                  NUMEXPR_NUM_THREADS=f'{num_cores}', MKL_NUM_THREADS=f'{num_cores}')

"""Set separate compile dir for theano""" 
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
unixtime = int(time.time())
os.environ['THEANO_FLAGS'] = rf'base_compiledir={dir_path}/.parallel_temp/temp_compile_dir/{unixtime}/.theano'
print('Set theano compile dir to')
print(os.environ['THEANO_FLAGS'])

import time, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from inputimeout import inputimeout, TimeoutOccurred
from astropy.table import Table


from util.data import psave, pload, pickle_save, textload
from util.timeseriestools import (remove_nans, sigma_clip, phase_fold,
                                  sorted_by_time, stitch_quarters,
                                  dict_to_tuple, stitch_sectors)
from util.exoplanets import (load_lightcurve, fit_secondary_eclipse_minimal, fit_secondary_eclipse_multiplanet ,
                             add_eccentricity_priors, load_priors_versatile, is_multiplanet, is_eccentric, 
                             PlanetNotFoundError)
from util.logging import print_bold, print_hline

def make_groups(sectors, max_gap = 4): # max gap is difference in sector numbers
    """Group clumps of sectors together"""
    groups = []
    i = 1
    groupi = [sectors[0]]
    while i < len(sectors):
        if sectors[i] - groupi[-1] > max_gap:
            groups.append(groupi)
            groupi = [sectors[i]]
        else:
            groupi.append(sectors[i])
        i += 1
    groups.append(groupi)
    return groups

def bayesian_analysis(t, y, yerr, priors,
                        sample_mcmc=False, figdirout=None, multiplanet=False, **kwargs):
    """Run fit_secondary_eclipse_minimal twice, clipping residuals after the first run. 
    Option to sample MCMC for error analysis"""
    ### For median depth analysis and least squares analysis see github version prior to Oct 11

    progressbar = socket.gethostname() != 'uzay.mit.edu' # progressbar is buggy in uzay
    
    if multiplanet:
        fit = fit_secondary_eclipse_multiplanet
    else:
        fit = fit_secondary_eclipse_minimal
    
    """Fit twice, clipping residuals in between"""
    map_soln0, fits0 = fit(t, y, yerr, priors=priors, return_model=False, start=None,
                                    progressbar=progressbar, **kwargs)
    residuals = y - map_soln0['mean'] - fits0['gp_pred'] - fits0['lc_pred']
    mask = np.invert(sigma_clip(t, residuals, sigma=5, return_masked_array=True, s=2, plot=False)[0].mask)
    map_soln, fits, model = fit(t[mask], y[mask], yerr[mask], priors=priors, return_model=True, start=map_soln0, progressbar=progressbar, **kwargs)


    """Plot GP fit"""
    fig1, ax1 = plt.subplots(1,2, figsize=(16, 4))
    ax1[0].scatter(t, y * 1e6)

    ax1[0].plot(t, (map_soln0['mean'] + fits0['gp_pred']) * 1e6, linewidth=1, color='tab:green', zorder=1000)
    ax1[0].set_ylim(np.nanmin(fits0['gp_pred'])*1.1*1e6, np.nanmax(fits0['gp_pred'])*1.1*1e6)
    ax1[1].scatter(t[mask], y[mask] * 1e6)
    ax1[1].plot(t[mask], (map_soln['mean'] + fits['gp_pred']) * 1e6, linewidth=1, color='tab:green', zorder=1000)
    ax1[1].set_ylim(np.nanmin(fits['gp_pred'])*1.1*1e6, np.nanmax(fits['gp_pred'])*1.1*1e6)
    if figdirout is not None:
        fig1.savefig(f'{figdirout}/4.1-GP-fit.png', dpi=300)
        plt.close(fig1)
    else:
        plt.close(fig1) # do it anyways!

    if multiplanet:
        planets = [x for x in sorted(priors.keys()) if len(x) == 1]
        se_depth = dict(zip(planets, [map_soln[f'se_depth_{p}'] * 1e6 for p in planets]))
        transit_depth = dict(zip(planets, [map_soln[f'transit_depth_{p}'] * 1e6 for p in planets]))
    else:
        se_depth = map_soln['se_depth'] * 1e6
        transit_depth = map_soln['transit_depth'] * 1e6

    if not multiplanet:
        fig, ax = plt.subplots(2, 3, figsize=(12,6))

        if 't0' in priors:
            t0,period = priors['t0'], priors['period']
            t0m, periodm = map_soln['t0'], map_soln['period']
            ax[0][0].axvline(t0+(np.ceil((t[0]-t0)/period)+1)*period, color='red', zorder=1000)
            ax[0][0].axvline(t0m+(np.ceil((t[0]-t0m)/periodm)+1)*periodm, color='green', zorder=1000)
            ax[0][0].axvline(t0m+(np.ceil((t[0]-t0m)/periodm)+1.5)*periodm, color='blue', zorder=1000)

        plt.suptitle('Maximum a posteriori solution')
        ax[0][0].scatter(t,y*1e6, color='k', s=2)
        ax[0][0].set_ylabel('ppm')
        sigma_clip(t, residuals*1e6, sigma=5, return_masked_array=True, s=2, plot=True, ax=ax[1][0])
        ax[0][0].plot(t[mask],1e6*(map_soln['mean']+fits['gp_pred']), linewidth=1.5, label='gp', color='tab:green', zorder=1000,
                      alpha=0.8)
        ax[0][0].legend()
        ax[1][0].set_title('residuals before refit')
        # ax[1][0].set_yscale('symlog', linthresh=500)

        tf, yf = phase_fold(t[mask], y[mask]-map_soln['mean'] - fits['gp_pred'], t0=map_soln['t0'], period=map_soln['period'])
        _, yfit = phase_fold(t[mask], fits['lc_pred'], t0=map_soln['t0'], period=map_soln['period'])
        tf -= 0.5
        se_mask = np.abs(tf) < 0.2

        ax[0][1].scatter(tf[se_mask], yf[se_mask]*1e6, color='k', s=2)
        ax[0][1].plot(tf[se_mask], yfit[se_mask]*1e6, color='tab:orange', linewidth=2, alpha=0.5, zorder=1000)
        ax[0][1].set_ylim(-1000, 100)
        
        ax[1][1].set_title('SE residuals')
        ax[1][1].scatter(tf[se_mask], yf[se_mask]*1e6-yfit[se_mask]*1e6, s=2, color='k')
        # ax[1][1].set_yscale('symlog', linthresh=500)

        tf, yf = phase_fold(t[mask], y[mask]-map_soln['mean'] - fits['gp_pred'], t0=map_soln['t0'] + 0.5 * map_soln['period'], period=map_soln['period'])
        _, yfit = phase_fold(t[mask], fits['lc_pred'], t0=map_soln['t0'] + 0.5 * map_soln['period'], period=map_soln['period'])
        tf -= 0.5
        transit_mask = np.abs(tf) < 0.2

        ax[0][2].scatter(tf[transit_mask], yf[transit_mask]*1e6, color='k', s=2)
        ax[0][2].plot(tf[transit_mask], yfit[transit_mask]*1e6, color='tab:orange', linewidth=2,  alpha=0.5, zorder=1000)
        try:
            ax[0][1].axhline(-se_depth, color='blue', label=f'δ={round(se_depth, 1)} ppm')
        except ValueError:
            ax[0][1].axhline(-se_depth, color='blue', label=f'δ={se_depth} ppm')
            print(map_soln)
        try:
            ax[0][2].axhline(-transit_depth, color='blue', label=f'δ={round(transit_depth)} ppm')
        except ValueError:
            ax[0][2].axhline(-transit_depth, color='blue', label=f'δ={transit_depth} ppm')
            print(map_soln)
        ax[1][2].set_title('Transit residuals')
        ax[1][2].scatter(tf[transit_mask], yf[transit_mask]*1e6-yfit[transit_mask]*1e6, s=2, color='k')
        ax[0][1].legend()
        ax[0][2].legend()
    else:
        plt.figure()
        plt.scatter(t[mask], (y[mask] - map_soln['mean'] - fits['gp_pred']) * 1e6)
        plt.plot(t[mask], fits['lc_pred']*1e6, linewidth=2, color='orange', zorder=1000)
        plt.ylabel('Relative flux [ppm]')
    if figdirout is not None:
        plt.savefig(map_fout:=f'{figdirout}/4-MAP-fit.png', dpi=300)
        print(f'Saved fig to {map_fout}')
        plt.close(plt.gcf())

    """Perform MCMC with model"""
    if sample_mcmc:
        import platform
        import pymc3 as pm
        
        starttime = time.time()
        tune=1500
        draws=1000
        cores=num_cores # you 
        chains=2
        
        with model:
            trace = pm.sample(
                tune=tune,
                draws=draws,
                start=map_soln, #model.test_point,
                # Parallel sampling runs poorly or crashes on macos
                cores=1 if platform.system() == "Darwin" else cores,
                chains=chains,
                target_accept=0.95,
                return_inferencedata=True,
                random_seed=[261136679, 261136680],
                init="adapt_full",
                progressbar = socket.gethostname() != 'uzay.mit.edu',
        )

        print(f'Sampling MCMC (tune={tune}, draws={draws}, chains={chains}, cores={cores}) took {round((time.time()-starttime)/60)} minutes')
    
    inference_results = {'map_soln': map_soln,
                       'map_fit': fits,}
    if sample_mcmc:
        inference_results['trace'] = trace

    return inference_results

def run_analysis(planet, config):
    print_hline()
    print_bold(f'[{planet}]')
    
    """Read configuration"""
    mission = config.mission # Kepler, TESS
    pipeline = config.pipeline.upper() # SAP, PDCSAP, QLP
    bitmask = config.bitmask # default, none, hardest, _modS...
    
    method = config.method.upper() # MAP or MCMC
    mode = config.mode.lower() # quarters, sectors, sectors_grouped
    overwrite = config.overwrite

    do_cumulative = not config.skip_cumulative
    sample_mcmc = method == 'MCMC'
    sample_mcmc_cumulative = config.sample_mcmc_cumulative 
    retry_failed = config.retry_failed
    do_cumulative_only = config.do_cumulative_only
    overwrite = config.overwrite

    """I/O"""
    results_dir = 'data/results'
    figs_dir = 'figures'
    # pb_dir = 'data/PrayerBead'
    mcmc_dir = 'data/MCMC'

    """Load lightcurve"""
    data = load_lightcurve(planet, mission=mission, time_format='bkjd/btjd', pipeline=pipeline, bitmask=bitmask, return_dict=True)

    """Check if multiplanetary system"""
    multiplanet = is_multiplanet(planet)

    """Load priors"""
    time_format='btjd' if mission == 'TESS' else 'bkjd'
    priors = load_priors_versatile(planet, time_format=time_format, t=data['time'], y=data['flux'], override_with_BLS=config.force_BLS) # If missing t0, run BLS from all data stitched

    fit_eccentricity = is_eccentric(planet)
    if fit_eccentricity:
        add_eccentricity_priors(priors, planet)
        print_bold('fitting eccentricity')

    """Stop here at test mode"""
    if method == 'test':
        print(f'Successfully loaded {planet}')
        return

    """Load/create results"""
    fio_results = f'{results_dir}/results.{planet}.{mode}.{mission}.{pipeline}.{bitmask}.p'
    
    if not os.path.exists(fio_results):
        print('Could not find existing results file', fio_results)
        print(f'Created new results table for {planet}')
        if mission == 'TESS':
            create_results_tb(planet, mission=mission,
                fout=fio_results, sectors=sorted(list(data['sectors'].keys())), mode=mode)
        elif mission =='Kepler':
            create_results_tb(planet, mission=mission, fout=fio_results, 
                                mode=mode)
    else:
        print(f'Found existing results table for {planet}')

    results = pload(fio_results, verbose=True)

    def check_already_done(row, cumulative=False):
        span = row['span']
        if cumulative:
            already_done = (method=='median' and row['median_transit_depth'] != 0 or 
                                    method=='LS' and row['ls_transit_depth'] != 0 or
                                    method in ['MAP', 'MCMC'] and row['map_transit_depth'] != 0 and not sample_mcmc_cumulative or
                                    method == 'MCMC' and os.path.exists(f'{mcmc_dir}/{planet}_{span}_MCMC.{mission}.{pipeline}.{bitmask}.p') and sample_mcmc_cumulative)
        else:
            already_done = (method=='median' and row['median_transit_depth'] != 0 or 
                                    method=='LS' and row['ls_transit_depth'] != 0 or
                                    method in ['MAP', 'MCMC'] and row['map_transit_depth'] != 0 and not sample_mcmc or
                                    method == 'MCMC' and os.path.exists(f'{mcmc_dir}/{planet}_{span}_MCMC.{mission}.{pipeline}.{bitmask}.p') and sample_mcmc)
        return already_done

    def analyze_and_store(row, t, y, yerr, priors, figdirout, multiplanet):
        if multiplanet:
            assert method not in ['median', 'LS']
        try:
            """Separately detrend if not using Bayesian"""
            # if method=='median' or method=='LS':
            #     orbital_period = priors['period']
            #     transit_epoch = priors['t0']
            #     transit_duration = priors['t14']
            #     t, y, yerr = preprocess_transit_lc(t, y, yerr, orbital_period, transit_duration, transit_epoch,
            #                                                     savefig=figdirout, verbose=False, skip_norm=True)
            if method=='median':
                pass
            elif method=='LS':
                pass
            elif method in ['MAP', 'MCMC']:
                if pipeline == 'QLP':
                    sigma = 10
                else:
                    sigma = 20

                # if mission in ['TESS']:
                #     gp_preset = 'tess'
                # elif mission in ['Kepler']:
                #     gp_preset = 'kepler'

                print(f'% Pre-analysis sigma clipping: {sigma}')
                t, y, yerr = sigma_clip(t, y, yerr, sigma=sigma)


                inference_results = bayesian_analysis(t, y, yerr, priors, 
                            sample_mcmc=sample_mcmc, figdirout=figdirout, multiplanet=multiplanet, 
                            fit_eccentricity=fit_eccentricity)
                
                if sample_mcmc:
                    fout_mcmc = f'{mcmc_dir}/{planet}_{row["span"]}_MCMC.{mission}.{pipeline}.{bitmask}.p'
                    pickle_save(inference_results, fout_mcmc)
                
                row['map_transit_depth'] = inference_results['map_soln']['transit_depth']
                row['map_se_depth'] = inference_results['map_soln']['se_depth']
                row['map_lc'] = inference_results['map_fit']['lc_pred']
                row['map_soln'] = inference_results['map_soln']
            row['failed'] = False
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            plt.close(plt.gcf())

            if str(e) == 'Not enough samples to build a trace.': # Equivalent to keyboard interrupt; don't count as error
                sys.exit()
            print_bold('Traceback', color='red')
            import traceback
            traceback.print_exc()
            
            print_bold('Failure!', color='red')
            row['failed'] = True
            
            if method=='median':
                row['median_se_depth'] = np.nan
                row['median_se_depth_err'] = np.nan
                row['median_transit_depth'] = np.nan
                row['median_transit_depth_err'] = np.nan
            if method=='LS':
                row['ls_se_depth'] = np.nan
                row['ls_se_depth_err'] = np.nan
                row['ls_transit_depth'] = np.nan
                row['ls_transit_depth_err'] = np.nan
                row['lmfit_params'] = None
            if method in ['MAP', 'MCMC']:
                row['map_transit_depth'] = np.nan
                row['map_se_depth'] = np.nan
                row['map_lc'] = None
                row['map_soln'] = None
            
            if str(e) == 'Theano Assert failed!':
                return 'TheanoAssertErr'

    if do_cumulative:
        figdirout = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/cumulative'
        os.makedirs(figdirout, exist_ok=True)
        print('[Cumulative]')

        indices = np.where(np.logical_and(results['planet']==planet, results['span']==f'cumulative'))[0]
        assert len(indices) == 1
        row = results[indices[0]]
        already_done = check_already_done(row, cumulative=True)

        if not overwrite and not retry_failed and row['failed'] == True:
            print(f'Cumulative already done (failed)')
        elif not overwrite and already_done and row['failed'] == False:
            print('Cumulative already done ')
        else:
            if row['failed']:
                print('Retrying failed cumulative')
            if mission == 'Kepler':
                t, y, yerr = stitch_quarters(data['quarters'])
            elif mission == 'TESS':
                t, y, yerr = stitch_sectors(data['sectors'])
            t, y, yerr = sorted_by_time(*remove_nans(t, y, yerr))
            y -= 1
            ### Custom sample_mcmc setting for cumulative.
            temp = sample_mcmc
            sample_mcmc = sample_mcmc_cumulative
            print(f'{sample_mcmc = }')
            status = analyze_and_store(row, t, y, yerr, priors=priors, figdirout=figdirout, multiplanet=multiplanet)
            sample_mcmc = temp
            ###
            psave(results, fio_results, verbose=False)
            if status == 'TheanoAssertErr':
                return 'TheanoAssertErr'
            if sample_mcmc_cumulative:
                print('Done sampling cumulative. Bye.')
                return
    else:
        print('Not doing cumulative as requested')
    print()

    if do_cumulative_only:
        return

    """Check that cumulative didn't fail"""
    cumulative = results[results['span'] == 'cumulative'][0]
    if cumulative['failed'] == False:
        if do_cumulative:
            assert cumulative['map_transit_depth'] != 0
        if reuse_priors_from_cumulative := False:
            if multiplanet:
                # for key in ['u_star', 'log_rho_gp', 'log_sigma_gp', 'log_sigma_lc']:
                #     try:
                #         priors[key] = np.float64(cumulative['map_soln'][key])
                #     except TypeError:
                #         continue
                planets = [x for x in sorted(priors.keys()) if len(x) == 1]
                for p in planets:
                    for key in ['a', 'b', 'ror', 'se_depth']:
                        try:
                            priors[p][key] = np.float64(cumulative['map_soln'][f'{key}_{p}'])
                        except TypeError:
                            continue

            else:
                for key in ['a', 'b', 'ror', 'se_depth', 'u_star']:
                    try:
                        priors[key] = np.float64(cumulative['map_soln'][key])
                    except TypeError:
                        continue

    else:
        print_bold('Skipping individual sectors/quarters since cumulative failed', color='red')
        return "fail"
    
    if mode in ['sectors_grouped']:
        sectors = sorted(list(data["sectors"].keys()))
        groups = make_groups(sectors)
        print('[Sector groups]')
        print(f'{len(groups)} groups, {len(sectors)} sectors: {groups}\n')

        for group in groups:
            span = 'G' + '.'.join([str(sector) for sector in group])
            figdirout = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/{span}'
            os.makedirs(figdirout, exist_ok=True)
            indices = np.where(np.logical_and(results['planet']==planet, results['span']==span))[0]
            assert len(indices) == 1
            row = results[indices[0]]
            already_done = check_already_done(row)

            if not overwrite and not retry_failed and row['failed'] == True:
                print(f'{span} already done (failed)')
            elif not overwrite and already_done and row['failed'] == False:
                print(f'{span} already done')
            else:
                if row['failed']:
                    print('Retrying failed sector group')
                print(f'%%%%%%%%%%%%% {planet} [Sector group {group}]')
                t, y, yerr = stitch_sectors(data['sectors'], sectors=group)
                if np.sum(np.isnan(yerr)) == len(yerr):
                    yerr = np.zeros(len(yerr))
                t, y, yerr = sorted_by_time(*remove_nans(t, y, yerr))
                yerr /= np.median(y)
                y /= np.median(y)
                y -= 1
                analyze_and_store(row, t, y, yerr, priors=priors, figdirout=figdirout, multiplanet=multiplanet)
                psave(results, fio_results, verbose=False)

    elif mode in ['sectors']:
        sectors = sorted(list(data["sectors"].keys()))
        print('[Sectors]')
        print(f'{len(sectors)} sectors: {sectors}\n')

        for sector in sectors:
            if config.sectors is not None:
                if sector not in config.sectors:
                    print(f'S{sector}: skipping as requested by user.')
                    continue
            figdirout = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/S{sector}'
            os.makedirs(figdirout, exist_ok=True)

            indices = np.where(np.logical_and(results['planet']==planet, results['span']==f'S{sector}'))[0]
            assert len(indices) == 1
            row = results[indices[0]]
            already_done = check_already_done(row)

            if not overwrite and not retry_failed and row['failed'] == True:
                print(f'S{sector} already done (failed)')
            elif not overwrite and already_done and row['failed'] == False:
                print(f'S{sector} already done')
            else:
                if row['failed']:
                    print('Retrying failed sector')
                print(f'%%%%%%%%%%%%% {planet} [Sector {sector}]')
                t, y, yerr = dict_to_tuple(data['sectors'][sector])
                if np.sum(np.isnan(yerr)) == len(yerr):
                    yerr = np.zeros(len(yerr))
                t, y, yerr = sorted_by_time(*remove_nans(t, y, yerr))
                yerr /= np.median(y)
                y /= np.median(y)
                y -= 1
                if planet == 'WASP-17~b' and sector == 12:
                    print('REMOVING TREND IN SECTOR 12 OF WASP-17 b')
                    t = t[100:]
                    y = y[100:]
                    yerr = yerr[100:]
                
                analyze_and_store(row, t, y, yerr, priors=priors, figdirout=figdirout, multiplanet=multiplanet)
                psave(results, fio_results, verbose=False)
    elif mode in ['window', 'windows']:
        print(f'priors: {priors}\n')
        for start in np.arange(1, 15):
            figdirout = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/W{start}'
            os.makedirs(figdirout, exist_ok=True)
            
            indices = np.where(np.logical_and(results['planet']==planet, results['span']==f'W{start}'))[0]
            assert len(indices) == 1
            row = results[indices[0]]
            already_done = check_already_done(row)
            
            if not overwrite and row['failed'] == True:
                print(f'Window starting with {start} already done (failed)')
            elif not overwrite and already_done:
                print(f'Window starting with {start} already done ')
            else:
                print(f'%%%%%%%%%%%%% {planet} [Window starting with {start}]')

                t, y, yerr = stitch_quarters(data['quarters'], quarters=[start, start+1, start+2, start+3])
                t, y, yerr = sorted_by_time(*remove_nans(t, y, yerr))
                yerr /= np.median(y)
                y /= np.median(y)
                y -= 1
                analyze_and_store(row, t, y, yerr, priors=priors, figdirout=figdirout, multiplanet=multiplanet)  
                psave(results, fio_results, verbose=False)
    elif mode in ['twelfth', 'twelfths']:
        """Do the same for each quarter"""
        if mission == 'Kepler':
            print('[Twelfths: quarters:]')
            print(f'{len(data["quarters"].keys())} quarters: {list(data["quarters"].keys())}\n')
        elif mission == 'TESS':
            raise ValueError
        
        print(f'priors: {priors}\n')
        
        for quarter in data['quarters'].keys():
            for j in range(1, 3+1):
                figdirout = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/Q{quarter}_{j}'
                os.makedirs(figdirout, exist_ok=True)

                indices = np.where(np.logical_and(results['planet']==planet, results['span']==f'Q{quarter}_{j}'))[0]
                assert len(indices) == 1
                row = results[indices[0]]
                already_done = (method=='median' and row['median_transit_depth'] != 0 or 
                                    method=='LS' and row['ls_transit_depth'] != 0 or
                                    method in ['MAP', 'MCMC'] and row['map_transit_depth'] != 0 and not sample_mcmc or 
                                    method == 'MAP' and row['map_transit_depth'] != 0 and os.path.exists(f'{mcmc_dir}/{planet}_{row["span"]}_MCMC.{mission}.p') and sample_mcmc)

                if not overwrite and row['failed'] == True:
                    print(f'Q{quarter}_j already done (failed)')
                elif not overwrite and already_done:
                    print(f'Q{quarter}_j already done')
                else:
                    print(f'%%%%%%%%%%%%% {planet} [Quarter {quarter}]_{j}') 
                    t, y, yerr = dict_to_tuple(data['quarters'][quarter])
                    t, y, yerr = sorted_by_time(*remove_nans(t, y, yerr))
                    n = len(t)

                    # Split into thirds
                    t = t[n//3 * (j-1): n//3 * j]
                    y = y[n//3 * (j-1): n//3 * j]
                    yerr = yerr[n//3 * (j-1): n//3 * j]

                    yerr /= np.median(y)
                    y /= np.median(y)
                    y -= 1
                    analyze_and_store(row, t, y, yerr, priors=priors, figdirout=figdirout, multiplanet=multiplanet)
                    psave(results, fio_results, verbose=False)
    elif mode in ['quarter', 'quarters']:
        """Do the same for each quarter"""
        print('[Quarters]')
        print(f'{len(data["quarters"].keys())} quarters: {list(data["quarters"].keys())}\n')
        print(f'priors: {priors}\n')
        
        for quarter in data['quarters'].keys():
            figdirout = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/Q{quarter}'
            os.makedirs(figdirout, exist_ok=True)

            indices = np.where(np.logical_and(results['planet']==planet, results['span']==f'Q{quarter}'))[0]
            assert len(indices) == 1
            row = results[indices[0]]
            already_done = check_already_done(row)

            if not overwrite and row['failed'] == True:
                print(f'Q{quarter} already done (failed)')
            elif not overwrite and already_done:
                print(f'Q{quarter} already done')
            else:
                print(f'%%%%%%%%%%%%% {planet} [Q{quarter}]')
                t, y, yerr = dict_to_tuple(data['quarters'][quarter])
                t, y, yerr = sorted_by_time(*remove_nans(t, y, yerr))
                yerr /= np.median(y)
                y /= np.median(y)
                y -= 1
                analyze_and_store(row, t, y, yerr, priors=priors, figdirout=figdirout, multiplanet=multiplanet)
                psave(results, fio_results, verbose=False)

    return "success"


def create_results_tb(planet, mission, fout, mode='quarters', sectors=None):
    results = Table(names=('planet', 'mission', 'span', 'failed',
                'median_transit_depth', 'median_transit_depth_err', 'median_se_depth', 'median_se_depth_err',
                'map_transit_depth', 'map_se_depth', 'map_lc', 'map_soln', 'mcmc_transit_depth_err', 'mcmc_se_depth_err',
                'ls_transit_depth', 'ls_transit_depth_err', 'ls_se_depth', 'ls_se_depth_err', 'lmfit_params'),
                dtype= # planet dtype for strings
                ('object', 'object', 'object',  bool,
                np.float64, np.float64, np.float64, np.float64,
                'object', 'object', np.ndarray, dict, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64, lmfit.Parameters))
    
    if mission in ['TESS']:
        assert sectors is not None
        if mode in ['sectors']:
            for sector in sectors:
                results.add_row()
                row = results[-1]
                row['planet'] = planet
                row['mission'] = 'TESS'
                row['span'] = f'S{sector}'
        elif mode in ['sectors_grouped']:
            groups = make_groups(sectors)
            for group in groups:
                span = 'G' + '.'.join([str(sector) for sector in group])
                print(span)
                results.add_row()
                row = results[-1]
                row['planet'] = planet
                row['mission'] = 'TESS'
                row['span'] = span
        else: 
            raise ValueError
    if mission in ['Kepler']:
        if mode in ['window', 'windows']:
            for i in range(1, 14+1):
                results.add_row()
                row = results[-1]
                row['planet']=planet
                row['mission'] = 'Kepler'
                row['span'] = f'W{i}'
        elif mode in ['year', 'years']:
            for i in range(1, 4+1):
                results.add_row()
                row = results[-1]
                row['planet'] = planet
                row['mission'] = 'Kepler'
                row['span'] = f'Y{i}'
        elif mode in ['twelfth', 'twelfths']:
            for i in range(17+1):
                for j in range(1, 3+1):
                    results.add_row()
                    row = results[-1]
                    row['planet'] = planet
                    row['mission'] = 'Kepler'
                    row['span'] = f'Q{i}_{j}' 
        elif mode in ['quarter', 'quarters']:
            for i in range(17+1):
                results.add_row()
                row = results[-1]
                row['planet'] = planet
                row['mission'] = 'Kepler'
                row['span'] = f'Q{i}'
        else:
            raise ValueError
    
    """Add cumulative"""
    results.add_row()
    row = results[-1]
    row['planet'] = planet
    row['mission'] = mission
    row['span'] = 'cumulative'
    
    psave(results, fout)


def main():
    parser = argparse.ArgumentParser(
                    prog='SEYFERT',
                    description='SEcondarY eclipse FittER')
    parser.add_argument('planets', metavar='planets', type=str, nargs='*')
    parser.add_argument('--input_file', default=None)

    parser.add_argument('-o', '--overwrite',
                action='store_true', default=False)
    # Execution
    parser.add_argument('-me', '--method') # MAP or MCMC
    parser.add_argument('-mo', '--mode', default='default')
    parser.add_argument('-c', '--sample_mcmc_cumulative', action='store_true', default=False)
    parser.add_argument('--do_cumulative_only', action='store_true', default=False)
    # parser.add_argument('--bin_data', action='store_true', default=False)
    parser.add_argument('--retry_failed', action='store_true', default=False)
    parser.add_argument('--skip_cumulative', action='store_true', default=False)
    parser.add_argument('--force_BLS', action='store_true', default=False)
    # Data
    parser.add_argument('-mi', '--mission', default='TESS')
    parser.add_argument('-p', '--pipeline', default='PDCSAP')
    parser.add_argument('-b', '--bitmask', default='hardest')
    
    parser.add_argument('--sectors', type=int, nargs='*') # only run analysis on certain sectors. Defaults to all sectors.
    
    args = parser.parse_args()


    """Config"""
    method = args.method
    mode = args.mode
    mission = args.mission
    pipeline = args.pipeline.upper()
    bitmask = args.bitmask
    planets = args.planets
    input_file = args.input_file
    sectors = args.sectors
    force_BLS = args.force_BLS

    if mission not in ['Kepler', 'TESS']:
        print_bold('Specify correct mission: [Kepler, TESS]', color='red')
        sys.exit(2)
    if method not in ['median', 'LS', 'MAP', 'MCMC', 'test']:
        print_bold('Specify correct method', color='red')
        sys.exit(2)
    if mode not in ['quarters', 'sectors', 'sectors_grouped']:
        print_bold('Specify correct mode', color='red')
        sys.exit(2)
    if pipeline not in ['PDCSAP', 'SAP', 'QLP']:
        print_bold('Specify correct pipeline', color='red')
        sys.exit(2)
    
    if input_file is not None:
        planets = textload(input_file).replace(' ', '~').split('\n')
        print(f'Loaded {len(planets)} kics from {input_file}')
    else:
        if len(planets) == 0:
            print(f'Using default list of targets: {(fin_targets_paper1 := "data/targets_mag15.p")}')
            planets = pload(fin_targets_paper1)
            planets = [obj for obj in planets if not is_multiplanet(obj)]
        planets = [obj.replace('Kepler', 'Kepler') for obj in planets]
    
    do_cumulative = not args.skip_cumulative
    sample_mcmc_cumulative = args.sample_mcmc_cumulative 
    retry_failed = args.retry_failed
    overwrite = args.overwrite

    print_bold('\nConfiguration')
    print(f"""
    {method = }
    {do_cumulative = }, {sample_mcmc_cumulative = } (do_cumulative must be on)
    {force_BLS = }
    {retry_failed = }
    {overwrite = }
    {num_cores = } # for numpy and pymc3
    -------
    {mission = }, {mode = }, {pipeline = }, {bitmask = }
    {planets = }
    """)

    try:
        inputimeout(prompt=f'Press enter to continue or wait {(timeout:=10)} seconds\n', timeout=timeout)
    except TimeoutOccurred:
        pass

    
    results_dir = f'{root}/data/results'
    figs_dir = f'{root}/figures'
    pb_dir = f'{root}/data/PrayerBead'
    mcmc_dir = f'{root}/data/MCMC'
    for fdir in results_dir, figs_dir, pb_dir, mcmc_dir:
        os.makedirs(fdir, exist_ok=True)

    missing_planets = []
    theanoassertionerr_planets = []
    assertionerr_planets = []
    failed_planets = []

    for planet in planets:
        try:
            status = run_analysis(planet, config=args)
            if status == "fail":
                failed_planets.append(planet)
            if status == 'TheanoAssertErr':
                theanoassertionerr_planets.append(planet)
        # except AssertionError:        
        #     print_bold(f'AssertionError for {planet}', color='red')
        #     assertionerr_planets.append(planet)
        #     failed_planets.append(planet)
        except FileNotFoundError:
            print_bold(f'FileNotFoundError for {planet}', color='red')
            missing_planets.append(planet)
            failed_planets.append(planet)
        except PlanetNotFoundError:
            print_bold(f'PlanetNotFoundError for {planet}', color='red')
            missing_planets.append(planet)
            failed_planets.append(planet)
    
    print_hline()
    print_bold('Summary')
    print('Planets')
    print(planets)
    print('Planets where fitting cumulative failed:')
    print(failed_planets)
    print('Planets causing FileNotFoundError/PlanetNotFoundError:')
    print(missing_planets)
    print('Planets causing Theano assertion error')
    print(theanoassertionerr_planets)
    print('Planets causing other assertion error')
    print(assertionerr_planets)

if __name__ == '__main__':
    main()