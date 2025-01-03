#!/users/canis/dev/astro/shporer/venv/bin/python
import os, argparse, time
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from canislib.data import psave, pload
from canislib.timeseriestools import remove_nans, running_mean, sorted_by_time
from canislib.misc import print_bold, print_hline
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from inputimeout import inputimeout, TimeoutOccurred
from dill import UnpicklingError
from canislib.exoplanets import compute_transit_depth, load_results, is_multiplanet, tildes_to_spaces

warnings.filterwarnings('ignore')

"""make sure to investigate these later"""
bad_quarters = {
    'Kepler-4~b': [14, 15],
    'Kepler-488~b': [7]
}
bad_sectors_QLP = {
    'KELT-7~b': [45]
}

def plot_depths(ax, timestamps, results, emph=False, annotation="", low_alpha=False):
    """Plot temporal 2nd eclipse and transit depths"""
    x = timestamps
    if emph:
        ax[0].errorbar(x, results['se_depth'], np.transpose(results['se_depth_err']),
            marker='D', linewidth=2, ls='-', ms=6, capsize=4, label=annotation, color='tab:blue', zorder=int(1e6))
        ax[1].errorbar(x, results['transit_depth'], np.transpose(results['transit_depth_err']),
            marker='D', linewidth=2, ls='-', ms=6, capsize=4, color='tab:blue', zorder=int(1e6))
    else:
        ax[0].errorbar(x, results['se_depth'], np.transpose(results['se_depth_err']),
            marker='o', linewidth=2, ls='-', ms=6, capsize=4, label=annotation, alpha=0.2 if low_alpha else 1)
        ax[1].errorbar(x, results['transit_depth'], np.transpose(results['transit_depth_err']),
            marker='o', linewidth=2, ls='-', ms=6, capsize=4, alpha=0.2 if low_alpha else 1)

    if annotation != "":
        ax[0].legend()

    """Draw a dotted line at 0 if visible"""
    if np.nanmin(results['se_depth']) - 3*np.nanmax(results['se_depth_err']) < 0:
        ax[0].axhline(0, linestyle='dotted', color='k', linewidth=2)

def plot_params(ax, timestamps, results, parameters, emph=False, low_alpha=False):
    """Plot parameters vs time"""
    x = timestamps
    for parameter in parameters:
        if parameter == 'se_depth':
            subax = ax[0][0]
        elif parameter == 'transit_depth':
            subax = ax[0][1]
        elif parameter == 'a':
            subax = ax[1][0]
        elif parameter == 'b':
            subax = ax[2][1]
        elif parameter == 'ror':
            subax = ax[1][1]
        elif parameter == 'sigma_lc':
            subax = ax[2][0]
        elif parameter == 'rho_gp':
            subax = ax[3][1]
        elif parameter == 'sigma_gp':
            subax = ax[3][0]
        elif 'u_star' in parameter:
            subax = ax[4][0]
        elif 'period' in parameter:
            subax = ax[4][1]
        else:
            raise ValueError

        subax.annotate(parameter, xy=(0.03, 0.85), xycoords='axes fraction', size=30, ha='left')
        if parameter in ['se_depth', 'transit_depth', 'sigma_lc', 'sigma_gp']:
            subax.errorbar(x, 1e6*results[f'{parameter}'], 1e6*np.transpose(results[f'{parameter}_err']),
                        marker='o',linewidth=2, ms=6, capsize=4, zorder=1000,alpha=0.2 if low_alpha else 1)
        elif 'u_star' in parameter:
            subax.errorbar(x, results[f'{parameter}'], np.transpose(results[f'{parameter}_err']), label=parameter,
                        marker='o',linewidth=2, ms=6, capsize=4, zorder=1000,alpha=0.2 if low_alpha else 1)
            subax.legend()
        else:
            subax.errorbar(x, results[f'{parameter}'], np.transpose(results[f'{parameter}_err']),
                        marker='o',linewidth=2, ms=6, capsize=4, zorder=1000,alpha=0.2 if low_alpha else 1)

def create_results_plots(planet, mission, pipeline, bitmask,
                         mode, method,
                         overwrite, verbose, 
                         fig=None, ax=None, fig2=None, ax2=None, write=True, hide_title=False):
    multiplanet = is_multiplanet(planet)
    planet_id = planet[-1]

    if mode not in ['all', 'sectors', 'sectors_grouped', 'both', 'quarter', 'quarters', 'years', 'years', 'window', 'windows', 'twelfth', 'twelfths']:
        print_bold('Choose correct plotting mode', color='red')
        raise ValueError

    results_dir = 'data/results'
    mcmc_dir = 'data/mcmc'
    figs_dir = 'figures'
    obj_fig_dir = f'{figs_dir}/{planet}'
    os.makedirs(obj_fig_dir, exist_ok=True)

    fin_results = f'{results_dir}/results.{planet}.{mode}.{mission}.{pipeline}.{bitmask}.p'
    fout_depths = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/depths.{mode}.png'
    fout_params = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/params.{mode}.png'

    if method.lower() == 'map':
        fout_depths = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/depths.{mode}.MAP.png'
        fout_params = f'{figs_dir}/{planet}/{mission}.{pipeline}.{bitmask}/params.{mode}.MAP.png'

    fin_cumulative = f'{mcmc_dir}/{planet}_cumulative_MCMC.{mission}.{pipeline}.{bitmask}.p'

    if not os.path.exists(fin_results):
        print_bold('Results file does not exist. Skipping', color='red')
        return

    already_done = not overwrite and os.path.exists(fout_depths) and os.path.getmtime(fout_depths) > os.path.getmtime(fin_results)
    if os.path.exists(fin_cumulative):
        already_done = already_done and os.path.exists(fout_depths) and os.path.getmtime(fout_depths) > os.path.getmtime(fin_cumulative)

    if fig is None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8)) # depths plot
        fig2, ax2 = plt.subplots(5, 2, figsize=(20, 15)) # params plot
        
    if hide_title:
        if planet == 'Kepler-13~b':
            fig.suptitle('KOI-13 b ($Kepler$)')
    else:
        if planet in ['Kepler-2b']:
            fig.suptitle(f'Kepler-2 b (HAT-P-7 b)', fontweight='bold')
        elif planet in ['Kepler-1b']:
            fig.suptitle(f'Kepler-1 b (TrES-2 b)', fontweight='bold')
        else:
            fig.suptitle(tildes_to_spaces(planet) + f' ({mission}, {pipeline}, {bitmask})', fontweight='bold')
        fig2.suptitle(tildes_to_spaces(planet) + f' ({mission}, {pipeline}, {bitmask})', fontweight='bold')

    if already_done:
        print('Already up to date. Skipping depths plot')
    else:
        print(f'Reading MCMC SE and Transit depths')

        cumulative_has_mcmc = method.lower() == 'mcmc'
        try:
            results_c = load_results(planet, mission=mission, pipeline=pipeline, bitmask=bitmask,
                                    method=method, mode='cumulative', verbose=verbose)
            assert not np.isnan(results_c['se_depth'])
        except (UnpicklingError, FileNotFoundError, AssertionError):
            cumulative_has_mcmc=False

        parsed_map_soln = load_results(planet, mission=mission, pipeline=pipeline, bitmask=bitmask,
                                            method='MAP', mode='cumulative', verbose=verbose)
        print('cumulative has mcmc:', cumulative_has_mcmc)

        
        parameters = ['se_depth', 'transit_depth', 
                      'b', 'a', 
                      'ror', 'sigma_gp', 
                      'rho_gp', 'sigma_lc', 
                      'u_star_0', 'u_star_1', 'period']

        if mode in ['sectors', 'sectors_grouped']:
            try:
                sectors, results_s = load_results(planet, mission=mission, pipeline=pipeline, bitmask=bitmask,
                                                    mode='sectors', method=method, parameters=parameters, verbose=verbose)
                plot_depths(ax, sectors, results_s, low_alpha=mode=='sectors_grouped')
                plot_params(ax2, sectors, results_s, parameters, low_alpha=mode=='sectors_grouped')
            except FileNotFoundError:
                pass
        if mode in ['sectors_grouped']:
            x, results_g, sector_groups = load_results(planet, mission=mission, pipeline=pipeline, bitmask=bitmask,
                                                        mode='sectors_grouped', method=method, parameters=parameters, verbose=verbose)
            plot_depths(ax, x, results_g, emph=True)
            plot_params(ax2, x, results_g, parameters, emph=True)
        if mode in ['quarter', 'quarters', 'both', 'all']:
            qs, results_q = load_results(planet, mission=mission, pipeline=pipeline, bitmask=bitmask,
                method=method, parameters=parameters, mode='quarters', verbose=verbose)

            if planet in bad_quarters.keys():
                for badq in bad_quarters[planet]:
                    results_q['se_depth'][qs.tolist().index(badq)] = np.nan
                    results_q['transit_depth'][qs.tolist().index(badq)] = np.nan

            plot_depths(ax, qs, results_q)
            plot_params(ax2, qs, results_q, parameters)
        if mode in ['window', 'windows', 'all', 'both']:
            ws, results_w = load_results(planet, mission=mission, pipeline=pipeline, bitmask=bitmask,method=method,
                    mode='window', parameters=parameters, verbose=verbose)
            ws = np.array(np.float64(ws)) + 1.5
            #mask = [window in [3.5, 7.5, 11.5, 15.5] for window in ws]
            mask = [True for window in ws]
            ws = ws[mask]

            plot_depths(ax, ws, results_w)
            plot_params(ax2, ws, results_w, parameters)
        if mode in ['twelfth', 'twelfths', 'all']:
            tws, results_tw = load_results(planet, mission=mission, pipeline=pipeline, bitmask=bitmask,
                                method=method, mode='twelfths', parameters=parameters, verbose=verbose)

            plot_depths(ax, tws, results_tw)
            plot_params(ax2, tws, results_tw, parameters)


        """Labels"""
        ax[0].annotate('Secondary Eclipse', xy=(0.99, 0.90), xycoords='axes fraction', size=24, ha='right')
        ax[1].annotate('Transit', xy=(0.99, 0.90), xycoords='axes fraction', size=24, ha='right')

        """Plot horiz line at cumulative result"""
        if pipeline != 'QLP':
            if cumulative_has_mcmc:
                print(results_c['se_depth'])
                ax[0].axhline(results_c['se_depth'], zorder=0, linewidth=2, color='maroon', label='cumulative', linestyle='dotted')
                ax[0].axhspan(results_c['se_depth'] - results_c['se_depth_err'][0],
                                results_c['se_depth'] + results_c['se_depth_err'][1],
                                color='maroon', alpha=0.1)
            else:
                ax[0].axhline(parsed_map_soln[f'se_depth'], zorder=0, linewidth=2, color='maroon', label='cumulative', linestyle='dotted')
            if cumulative_has_mcmc:
                ax[1].axhline(results_c['transit_depth'], zorder=0, linewidth=2, color='maroon', label='cumulative', linestyle='dotted')
                ax[1].axhspan(results_c['transit_depth'] - results_c['transit_depth_err'][0],
                                results_c['transit_depth'] + results_c['transit_depth_err'][1],
                                color='maroon', alpha=0.1)
            else:
                ax[1].axhline(parsed_map_soln[f'transit_depth'], zorder=0, linewidth=2, color='maroon', label='cumulative', linestyle='dotted')

        """Only label axes if the figure is on the left column in the manuscript"""
        # if planet in ['Kepler-1b', 'Kepler-5b', 'Kepler-7b', 'Kepler-12b', 'Kepler-15b', 'Kepler-18c',
        #             'Kepler-76b', 'Kepler-427b', 'Kepler-1658b', 'Kepler-77b', 'Kepler-471b']:
        if True:
            ax[0].set_ylabel('Relative flux [ppm]')
            ax[1].set_ylabel('Relative flux [ppm]')
            # ax[0].set_ylabel('Fractional change [%]')
            # ax[1].set_ylabel('Fractional change [%]')
        if mission in ['Kepler']:
            ax[1].set_xlabel('Kepler quarter')
            ax[1].set_xticks(np.arange(1, 18))
            ax[0].set_xticks(np.arange(1, 18))
        if mission in ['TESS']:
            ax[1].set_xlabel('TESS sector')
            ax[0].set_xticks(xticks:=np.unique([int(s) for s in sectors]))
            ax[1].set_xticks(xticks)
            plt.setp(ax[0].get_xticklabels(), rotation=60, horizontalalignment='center')
            plt.setp(ax[1].get_xticklabels(), rotation=60, horizontalalignment='center')
        plt.setp(ax[0].get_xticklabels(), visible=False)
        ax[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

        if write:
            # fig.patch.set_alpha(0.0)
            # for axi in ax:
            #     axi.patch.set_alpha(1.0)
            # fig2.patch.set_alpha(0.0)
            # for axi in np.ravel(ax2):
            #     axi.patch.set_alpha(1.0)
            fig.savefig(fout_depths, dpi=300)
            print(f'Saved to {fout_depths}')
            plt.close(fig)

            fig2.savefig(fout_params, dpi=300)
            print(f'Saved to {fout_params}')
            plt.close(fig2)

def main():
    plt.rc('font', size=24)
    plt.rc('axes', labelsize=24)
    plt.rc('legend', fontsize=16)
    plt.rc('xtick',labelsize=24) # previously 16
    plt.rc('ytick',labelsize=24)

    parser = argparse.ArgumentParser(
                    prog='plot_SEYFERT',
                    description='Plot SEcondarY eclipse FittER Results')

    parser.add_argument('-o', '--overwrite',
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('--hide_title', action='store_true')

    parser.add_argument('-me', '--method', required=False, default='mcmc')
    parser.add_argument('-mo', '--mode', required=False, default='sectors')

    parser.add_argument('-mi', '--mission', required=False, default='TESS')
    parser.add_argument('-p', '--pipeline', required=False, default='PDCSAP')
    parser.add_argument('-b', '--bitmask', required=False, default='hardest')

    parser.add_argument('planets', metavar='planets', type=str, nargs='*')

    args = parser.parse_args()

    overwrite = args.overwrite
    verbose = args.verbose
    planets = args.planets
    method = args.method
    mode = args.mode
    hide_title = args.hide_title

    mission = args.mission
    pipeline = args.pipeline
    bitmask = args.bitmask

    """Load planets and print config"""
    results_dir = 'data/figures'

    if not isinstance(planets, list):
        planets = [planets]

    print_hline()
    print_bold('\nConfiguration')
    print(f"""
    {planets = }
    {method = }
    {mode = }
    {overwrite = }

    {mission = }
    {pipeline = }
    {bitmask = }
    """)
    print_hline()
    try:
        inputimeout(prompt=f'Press enter to continue or wait {(timeout:=10)} seconds\n', timeout=timeout)
    except TimeoutOccurred:
        pass

    if pipeline == 'all':
        pipelines = ['SAP', 'PDCSAP', 'QLP']
    else:
        pipelines = [pipeline]

    if bitmask == 'all':
        bitmasks = ['hardest', 'default', 'none']
    else:
        bitmasks = [bitmask]

    for pl in planets:
        for bmask in bitmasks:
            for pline in pipelines:
                print_bold(f'{pl}: {mission}/{pline} ({bmask})')
                if pline == 'both':
                    fig, ax = plt.subplots(2, 1, figsize=(12, 8)) # depths plot
                    fig2, ax2 = plt.subplots(4, 2, figsize=(20, 15)) # params plot
                    create_results_plots(pl, mission=mission, pipeline='PDCSAP', bitmask=bmask,
                                        mode=mode, method=method, verbose=verbose, overwrite=True, 
                                        fig=fig, ax=ax, fig2=fig2, ax2=ax2, write=False, hide_title=hide_title)
                    create_results_plots(pl, mission=mission, pipeline='SAP', bitmask=bmask,
                                        mode=mode, method=method, verbose=verbose, overwrite=True,
                                        fig=fig, ax=ax, fig2=fig2, ax2=ax2, write=False, hide_title=hide_title)
                    fig.suptitle(tildes_to_spaces(pl) + f' ({mission}, {pipeline}, {bitmask}) (blue=SAP)', fontweight='bold')
                    fig2.fig.suptitle(tildes_to_spaces(pl) + f' ({mission}, {pipeline}, {bitmask}) (blue=SAP)', fontweight='bold')
                    fout_depths = f'figures/{pl}/depths.{mission}.both.png'
                    fout_params = f'figures/{pl}/params.{mission}.both.png'
                    fig.savefig(fout_depths, dpi=300)
                    print(f'Saved to {fout_depths}')
                    plt.close(fig)

                    fig2.savefig(fout_params, dpi=300)
                    print(f'Saved to {fout_params}')
                    plt.close(fig2)
                else:
                    create_results_plots(pl, mission=mission, pipeline=pline, bitmask=bmask,
                                        mode=mode, method=method, verbose=verbose, overwrite=overwrite,
                                        hide_title= hide_title)

    print('Done. Don\'t forget to plot_corner_plots.py')

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    main()
