"""
Tools for working with exoplanet data
"""

import os, copy, socket
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from numba import njit

from astropy.table import Table

from canislib.timeseriestools import stitch_sectors, stitch_quarters, remove_nans, get_transit_mask
from canislib.timeseriestools import convert_time
from canislib.misc import fstr_ratio, print_bold
from canislib.data import pload, psave
from pickle import UnpicklingError


# ------------------------------ Useful numbers ---------------------------------
au = 1.496e11
rsun = 6.957e8

quarter_boundaries = {0: [120, 131],
                      1: [131, 165],
                      2: [169, 259],
                      3: [260, 350],
                      4: [352, 443],
                      5: [443, 539],
                      6: [539, 630],
                      7: [630, 720],
                      8: [735, 803],
                      9: [808, 906],
                      10: [906, 1001],
                      11: [1001, 1099],
                      12: [1099, 1183],
                      13: [1183, 1274],
                      14: [1274, 1372],
                      15: [1373, 1472],
                      16: [1472, 1558],
                      17: [1559, 1592]}

# ------------------------------ Radiometry ---------------------------------

root = '.'

def load_passband(mission):
    match mission:
        case 'Kepler':
            wl, tr = np.transpose(np.loadtxt('f{root}/data/passband/Kepler_transmission.txt', skiprows=1))
        case 'TESS':
            wl, tr = np.transpose(np.loadtxt('f{root}/data/passband/tess-response-function-v1.0.csv', skiprows=8, delimiter=','))
        case 'CHEOPS':
            wl, QE, throughput, tr = np.transpose(np.loadtxt('f{root}/data/passband/CHEOPS_bandpass.csv', skiprows=1, delimiter=','))

    return wl, tr

h = 6.62607015e-34
c = 299_792_458
k_b = 1.380649e-23
π = np.pi
σ = 2 * π**5 * k_b**4/(15* c**2 * h**3)

@njit
def compute_spectral_radiance(T, λs=None):
    return 2 *h*c**2/np.power(λs,5) * 1/(np.exp(h*c/(λs*k_b*T))-1)

def compute_exittance(T, λs=None, passband=None):
    if passband is None:
        if λs is None:
            λs = 1e-9*np.linspace(10, 20_000_000, 1_000_000)
        transmission = 1
    else:
        wl, tr = load_passband(passband)
        wl *= 1e-9
        if λs is None:
            λs = np.linspace(wl[0]*0.99, wl[-1]*1.01, 1_000_000)
        transmission = np.interp(λs, wl, tr, left=0, right=0)

    dλs = np.concatenate([np.diff(λs), [0]])
    
    spectral_radiance = compute_spectral_radiance(T, λs)
    # print(np.max(spectral_radiance))
    lambertian = π # Assuming hemispherical emission
    spectral_exittance = spectral_radiance * lambertian

    spectral_exittance *= transmission
    if max(spectral_exittance[0], spectral_exittance[-1]) > 1e-2:
        print('Warning: may need to use wider wavelength range. Computed values at boundary are large:')
        print(spectral_exittance[0], spectral_exittance[-1])

    exittance = np.sum(spectral_exittance * dλs)
    return exittance


# ------------------------------ Datasets ---------------------------------

class PlanetNotFoundError(Exception):
    pass
class TODO(Exception):
    pass

def tildes_to_spaces(planet):
    # if not planet[-1].isdigit() and planet[-2] not in [' ', '~']:
        # planet = planet[:-1] + ' ' + planet[-1]
    return planet.replace('~', ' ')

def spaces_to_tildes(planet):
    # if not planet[-1].isdigit() and planet[-2] not in [' ', '~']:
        # planet = planet[:-1] + ' ' + planet[-1]
    return planet.replace(' ', '~')

def load_planet_table(tablename, verbose=True):
    tables_dir = f'{root}/data/tables'
    tablename = tablename.lower()
    if tablename in ['nea']:
        neatb = pload(fin:=f'{tables_dir}/NEA_table_2024-08-01.p', verbose=verbose) # unmodified version of the NEA table. Binary is faster than CSV
        neatb = neatb[neatb['default_flag']==1]
        return neatb
    elif tablename in ['koi', 'koi_dr24']: # USING DR24!!
        koitb = Table.read(fin:=f'{tables_dir}/KOI_table_DR24_2015-09-24.csv', comment='#')
        return koitb
    elif tablename in ['kepler', 'kepler_nea']:
        ktb = pload(fin:=f'{tables_dir}/kepler_nea_table_2023-10-14.p', verbose=verbose)
        return ktb
    elif tablename in ['toi']:
        toitb = Table.read(fin:=f'{tables_dir}/toi_catalog_2024-01-20.csv', comment='#')
        if verbose:
            print(f'💽 Loaded {type(toitb)} from {fin}')
        return toitb
    else:
        raise ValueError('Invalid planet table specified')

def query_planet(planet, tablename='auto', verbose=False):
    """Query for row of table for a planet"""

    if tablename in ['auto']:
        tablename = 'Kepler' if 'Kepler-' in planet else 'NEA'
    tablename = tablename.lower()
    if verbose:
        print(f'query_planet: Using {tablename} for {planet}')
    planettb = load_planet_table(tablename, verbose=verbose)

    if tablename in ['nea', 'kepler', 'kepler']:
        row = planettb[planettb['pl_name'] == tildes_to_spaces(planet)]
    if tablename in ['koi', 'koi_dr24']:
        row = planettb[planettb['kepler_name'] == tildes_to_spaces(planet)]
    if tablename in ['toi']:
        raise TODO('There shouldn\'t be a need to query TOI')

    if len(row) == 0:
        raise PlanetNotFoundError
    else:
        return row


def is_multiplanet(planet, verbose=True):
    """Check if a system has multiple planets"""

    if planet.replace(' ', '~') in ['WASP-18~b', 'TOI-1694~b', '55~Cancri~e', '55~Cnc~e']:
        return False

    row = query_planet(planet, tablename='auto', verbose=False)
    if len(row) > 0:
        if verbose:
            print(f'is_multiplanet: {planet}:', row['sy_pnum'][0] > 1)
        return row['sy_pnum'][0] > 1
    else:
        raise PlanetNotFoundError

def is_eccentric(planet, verbose=False):
    """Check if a planet's orbit has a significant eccentricity"""

    return spaces_to_tildes(planet) in ['Kepler-63~b', 'Kepler-488~b',
                'Kepler-1658~b', 'BD-14~3065~b'] and not is_multiplanet(planet) # 'Kepler-427b' #'Kepler-425b'

def add_eccentricity_priors(priors, pl):
    match pl:
        case 'BD-14~3065~b':
            priors['sqrt(e)cosω'] = np.sqrt(0.0660)*np.cos(54*np.pi/180)
            priors['sqrt(e)sinω'] = np.sqrt(0.0660)*np.sin(54*np.pi/180)
        case 'Kepler-488~b':
            priors['sqrt(e)cosω'] = 0.0192
            priors['sqrt(e)sinω'] = -0.369
    
    # return {'e':e, 'e_err_lower':, 'e_err+'}

def load_lightcurve(planet, pipeline='PDCSAP', mission='TESS', bitmask='hardest', return_dict=False,    
                    return_quality = False, quarters=None, sectors=None, time_format='bkjd/btjd', verbose=True):
    """Load Kepler or TESS light curve (downloaded by hermes)"""

    if mission not in ['Kepler', 'TESS']:
        raise ValueError('Invalid Mission')
    if quarters is not None:
        parts = quarters
    elif sectors is not None:
        parts = sectors
    else:
        parts = 'all'

    if isinstance(parts, int):
        parts = [parts]

    orig_time_format = 'bkjd' if mission == 'Kepler' else 'btjd' if mission == 'TESS' else None

    fin = f'{root}/data/lightcurves/'
    if mission == 'TESS':
        fin += 'TESS/'
    fin += f'{planet}.{pipeline}.{bitmask}.p'
    data = pload(fin, verbose=verbose)

    if return_dict:
        return data
    else:
        if return_quality:
            if mission in ['TESS', 'QLP']:
                t, y, yerr, q = stitch_sectors(data['sectors'], sectors=parts, return_quality=True, verbose=verbose)
            else:
                t, y, yerr, q = stitch_quarters(data['quarters'], quarters=parts, return_quality=True, verbose=verbose)

            if time_format != 'bkjd/btjd':
                t = convert_time(t, orig_time_format, time_format)

            t, y, yerr, q = remove_nans(t, y, yerr,q, verbose=verbose)
            y -= 1

            return t, y, yerr, q
        else:
            if mission in ['TESS', 'QLP']:
                t, y, yerr = stitch_sectors(data['sectors'], sectors=parts, verbose=verbose)
            else:
                t, y, yerr = stitch_quarters(data['quarters'], quarters=parts, verbose=verbose)

            if time_format != 'bkjd/btjd':
                t = convert_time(t, orig_time_format, time_format)

            t, y, yerr = remove_nans(t, y, yerr, verbose=verbose)
            y -= 1

            return t, y, yerr

def load_quality(planet, pipeline='PDCSAP', bitmask='none'):
    """Simple"""
    data =  pload(f'{root}/data/lightcurves/TESS/{planet}.{pipeline}.{bitmask}.p')

    return data['time'], data['quality']

def load_priors(planet, mission='Kepler', table='auto', get_transit_params=True, time_format='bkjd/btjd', verbose=False):
    from canislib.timeseriestools import convert_time
    if mission not in ['Kepler', 'TESS']:
        raise ValueError('Invalid Mission')

    if table.lower() in ['koi']:
        assert mission in ['Kepler']
        row = query_planet(planet, 'koi')
        priors = {'period': row['koi_period'][0], 't0': row['koi_time0bk'][0], 't14':row['koi_duration'][0]/24}

        if time_format != 'bkjd/btjd':
            priors['t0'] = convert_time(priors['t0'], 'bkjd', time_format)
        else:
            time_format = 'bkjd' if mission == 'Kepler' else 'btjd' if mission == 'TESS' else None
            priors['t0'] = convert_time(priors['t0'], 'bkjd', time_format)

        if get_transit_params:
            priors['ror'] = row['koi_ror'][0]
            priors['a'] = (row['koi_sma'][0] * 1.496e11) / (row['koi_srad'][0] * 6.957e8) # a/r_star
            priors['b'] = row['koi_impact'][0] # b/r_star

        for key in priors.keys(): # KOI table shouldn't be missing parameters.
            assert(not np.ma.is_masked(priors[key]) and not np.isnan(priors[key]))
    elif table.lower() in ['auto', 'kepler', 'nea']:
        row = query_planet(planet, table if table != 'auto' else ('kepler' if 'Kepler-' in planet else 'nea'))
        priors = {'period': row['pl_orbper'][0], 't0': row['pl_tranmid'][0], 't14':row['pl_trandur'][0]/24}

        for p in priors.keys():
            if np.ma.is_masked(priors[p]):
                priors[p] = np.nan

        if not np.isnan(priors['t0']):
            try:
                nea_time_scale = 'utc' if 'utc' in row['pl_tsystemref'][0].lower() else 'tdb'
                nea_time_format = row['pl_tsystemref'][0].lower().replace('-tdb', '').replace('-utc', '')
                if nea_time_format == 'bjd-tt': # terrestrial time
                    nea_time_format = 'bjd'
                elif nea_time_format == 'hjd':
                    nea_time_format = 'bjd' # APPROXIMATELY (up to ±4s error)
                if time_format != 'bkjd/btjd':
                    priors['t0'] = convert_time(priors['t0'], nea_time_format, time_format, ts1=nea_time_scale, ts2='tdb')
                else:
                    time_format = 'bkjd' if mission == 'Kepler' else 'btjd' if mission == 'TESS' else None
                    priors['t0'] = convert_time(priors['t0'], nea_time_format, time_format, ts1=nea_time_scale, ts2='tdb')
            except AttributeError: # Masked Column
                print_bold('Attribute error in getting t0 and timescale!', color='red')
                priors['t0'] = np.nan # Use BLS later instead

        if get_transit_params:
            priors['b'] = row['pl_imppar'][0] # b/r_star
            priors['ror'] = row['pl_ratror'][0]
            priors['a'] = row['pl_ratdor'][0]

        def is_good(val):
            if np.isnan(val) or np.isinf(val) or np.ma.is_masked(val):
                return False
            return True

        for p in sorted(priors.keys()): # Check for missing parameters
            if np.ma.is_masked(priors[p]) or np.isnan(priors[p]): # ephemeris can be made nan above.
                if p in ['a']:
                    if True:
                        print(f'Row missing a. Manually computing with provided planet/star values')
                    try:
                        Rs_sun = row['st_rad'][0]
                        a_au = row['pl_orbsmax'][0]
                        priors['a'] = compute_aor(a_au, Rs_sun)
                        assert is_good(priors['a'])
                    except AssertionError:
                        priors['a'] = 'missing'
                elif p in ['ror']:
                    if True:
                        print(f'Row missing ror. Manually computing with provided planet/star values')
                    try:
                        Rp_earth = row['pl_rade'][0]
                        Rs_sun = row['st_rad'][0]
                        priors['ror'] = compute_ror(Rp_earth, Rs_sun)
                        assert is_good(priors['ror'])
                    except (TypeError, AssertionError):
                        print('Missing ror:', Rp_earth, Rs_sun)
                        priors['ror'] = 'missing'
                elif p in ['b']:
                    print(f'Row missing b. Manually computing with a and inclination')
                    try:
                        assert priors['a'] != 'missing'
                        inc = row['pl_orbincl'][0]
                        priors['b'] = priors['a'] * np.cos(inc * np.pi/180)
                        assert is_good(priors['b'])
                    except AssertionError:
                        priors['b'] = 'missing'
                else:
                    if True:
                        print(f'Row missing {p}')
                    priors[p] = 'missing'
    else:
        raise ValueError

    if priors['b'] == 0:
        priors['b'] = 0.01 # 0 is invalid starting point

    return priors

def load_priors_multiplanet(planet=None, system=None, table='auto', get_transit_params=True, time_format='bkjd/btjd', verbose=False):
    from canislib.timeseriestools import convert_time
    if system is None:
        system = tildes_to_spaces(planet)[:-2]

    if table.lower() in ['koi']:
        koitb = load_planet_table('koi_dr24')
        rows = []
        for row in koitb:
            if isinstance(row['kepler_name'], str) and row['kepler_name'][:-2] == system:
                rows.append(row)
        assert len(rows) >= 0
        priors = {}
        for row in rows:
            priors[row['kepler_name'][-1]] = load_priors(row['kepler_name'], table=table, get_transit_params=get_transit_params,
                                            time_format=time_format) # slow but idc
    elif table.lower() in ['auto', 'kepler', 'nea']:
        planettb = load_planet_table(table if table != 'auto' else ('kepler' if 'Kepler-' in system else 'nea'))
        rows = []
        for row in planettb:
            if row['hostname'] == system:
                rows.append(row)
        assert len(rows) >= 0
        priors = {}
        for row in rows:
            priors[row['pl_letter']] = load_priors(row['pl_name'], table=table, get_transit_params=get_transit_params,
                                            time_format=time_format) # slow but idc
    else:
        raise ValueError
    assert len(priors.keys()) > 0
    return priors

def load_priors_versatile(planet, time_format, t, y, override_with_BLS=False):
    from canislib.timeseriestools import box_ls
    # if planet in ['KELT-1~b', 'TOI-163~b']:
    #     print(f'............... manual override with BLS for {planet}!')
    #     override_with_BLS = True
    if multiplanet:=is_multiplanet(planet):
        priors = load_priors_multiplanet(planet, table='auto', get_transit_params=True, 
                                                            time_format=time_format)
        if 'Kepler-' in planet:
            priors_koi = load_priors_multiplanet(planet, table='KOI', get_transit_params=True, 
                                                                        time_format=time_format)
        for pl in priors.keys():
            for p in priors[pl].keys():
                if priors[pl][p] == 'missing':
                    print(f'NEA Missing {p} (planet {pl})')
                    if 'Kepler-' in planet:
                        try:
                            priors[pl][p] = priors_koi[pl][p]
                        except KeyError:
                            print('Couldn\'t substitute KOI value')
                            pass
                elif p in['t0']:
                    priors[pl]['t0'] += np.ceil((t[0] - priors[pl]['t0'])/priors[pl]['period'])*priors[pl]['period']
    else:
        priors = load_priors(planet, table='auto', get_transit_params=True, time_format=time_format)
        if 'Kepler-' in planet:
            priors_koi = load_priors(planet, table='KOI', get_transit_params=True, time_format=time_format)
        for p in priors.keys():
            if priors[p] == 'missing':
                print(f'NEA Missing {p}')
                if 'Kepler-' in planet:
                    try:
                        priors[p] = priors_koi[p]
                    except KeyError:
                            print('Couldn\'t substitute KOI value')
                            pass
            elif p in['t0']:
                print(priors['t0'])
                priors['t0'] += np.ceil((t[0] - priors['t0'])/priors['period'])*priors['period']
                print(priors['t0'])
    """Remove long period planets"""
    to_pop = []
    if multiplanet:
        for pl in priors.keys():
            if priors[pl]['period'] > 90:
                print_bold(f'!!!!!!!!! Removing long period planet: {planet[:-1]}{pl}')
                to_pop.append(pl)
    for pl in to_pop:
        priors.pop(pl)
    print(priors)

    """Replace missing t0 or t14 with box least squares"""
    if multiplanet:
        for pl in priors.keys():
            if override_with_BLS or priors[pl]['t0'] == 'missing' or priors[pl]['t14'] == 'missing':
                print_bold(f'Running BLS to obtain ephemeris (forced: {override_with_BLS})', color='red')
                bls = box_ls(t, y, period=priors[pl]['period'])
                priors[pl]['t0'] = bls['t0']
                priors[pl]['t14'] = bls['t14']
    else:
        if override_with_BLS or priors['t0'] == 'missing':
            print_bold(f'Running BLS to obtain ephemeris (forced: {override_with_BLS})', color='red')
            bls = box_ls(*remove_nans(t, y, verbose=False), period=priors['period'])
            priors['t0'] = bls['t0']
            priors['t14'] = bls['t14']

        if priors['t14'] == 'missing':
            print(f'Computing t14')
            a = get_prior(priors, 'a', 4)
            b = get_prior(priors, 'b', 1)
            ror = get_prior(priors, 'ror', 0.1)
            priors['t14'] = priors['period']/np.pi * np.arcsin(np.sqrt((1+ror)**2-b**2)/a)
    
    print('Priors', priors)
    return priors

def canisplot(t=None, y=None, yerr=None, quality=None, data=None,
                ephemeris=None, mission='TESS',
                ax=None, title=None, **kwargs):
    """Plot a exoplanet lightcurve with transit epoch marked"""
    if data is not None:
        t, y, yerr = data['time'], data['flux'], data['flux_err']
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,6))
    plt.sca(ax)

    if yerr is None:
        plt.plot(t, y*1e6, **kwargs)
    else:
        if len(t) < 10000:
            plt.errorbar(t, y*1e6, yerr*1e6, c='0.2', ms=1, marker='o', **kwargs)
        else:
            markers, caps, bars = plt.errorbar(t, y*1e6, yerr*1e6, c='0.2', ms=1, marker='o', linestyle='none', **kwargs)
            [bar.set_alpha(0.5) for bar in bars]
    if ephemeris is not None:
        ephemeris['t0'] += ((t[0]-ephemeris['t0'])//ephemeris['period'] + 1)*ephemeris['period']
        if 't0' in ephemeris:
            plt.axvline(ephemeris['t0'], color='red', zorder=1000)
        else: # multiplanet
            for p, color in zip(list(ephemeris.keys()), ['red', 'orange', 'purple'] + ['blue'] * 100):
                plt.axvline(ephemeris[p]['t0'], color=color, zorder=1000, label=p)
            plt.legend()

    if quality is not None:
        from canislib.timeseriestools import separate_into_continuous_sections
        import lightkurve as lk
        decode = lk.TessQualityFlags.decode

        if mission == 'TESS':
            print('Using TESS max_gap_width')
            max_gap_width = 1000/86400
        else:
            print('Using Kepler max_gap_width') 
            max_gap_width = 10/48

        colors = ['tab:blue', 'tab:orange']
        for i, [times, qualityflag] in enumerate(np.transpose(separate_into_continuous_sections(t, quality, max_gap_width=max_gap_width, max_jump_height=0))):
            q = qualityflag[0] # TODO MAKE THIS SMARTER AND USE THE INDIV BINARY DIGITS
            print(i, decode(q))
            if q==0:
                plt.axvspan(times[0], times[-1], alpha=0.1, color='green')
            else:
                plt.axvspan(times[0], times[-1], alpha=0.1, color=colors[i%2])
                if times[-1] - times[0] > 0.5:
                    plt.annotate(decode(q), xy=(times[0], np.percentile(y, 1)))
                else:
                    pass
                    #plt.axvspan(times[0], times[-1], alpha=0.1) # gray

        # ax2 = ax.twinx()
        # ax2.plot(t, np.log(quality)/np.log(2), marker='o', color='red')


    if mission == 'Kepler':
        for q in quarter_boundaries.keys():
            t_start, t_end = quarter_boundaries[q]
            plt.axvspan(t_start, t_end, color= 'blue' if q %2 == 0 else 'white', alpha=0.1)
            plt.annotate(f'Q{q}', xy=((t_start + t_end)/2, np.nanmin(y)), xycoords='data', horizontalalignment='center')

    if title is not None:
        plt.title(title, fontweight='bold')



def canisplot2(pl):
    import sys
    sys.path.insert(1, '/users/canis/dev/astro/shporer')

    t, y, yerr = load_lightcurve(pl:='HAT-P-33~b', mission='TESS', pipeline='PDCSAP', bitmask='hardest')
    priors = load_priors_versatile(planet=pl,multiplanet=False,time_format='btjd',t=t,y=y)
    canisplot(t,y,yerr, ephemeris=priors)
    plt.show()


def canisview(pl, bitmask='hardest', mission='TESS'):
    fig, ax = plt.subplots(2,1, figsize=(10,10), sharex=True, sharey=True)
    t, y, yerr, quality = load_lightcurve(pl, pipeline='PDCSAP', mission=mission, bitmask=bitmask, return_quality=True)
    canisplot(t=t, y=y, yerr=yerr, quality=quality, mission=mission, ax=ax[0])
    try:
        t, y, yerr, quality = load_lightcurve(pl, pipeline='SAP', mission=mission, bitmask=(sap_bitmask:='none'), return_quality=True)
        canisplot(t=t, y=y, yerr=yerr, quality=quality, mission=mission, ax=ax[1])
    except FileNotFoundError:
        pass
    ax[0].annotate(f'PDCSAP flux: quality_bitmask={bitmask}', xy=(0.03, 0.90), xycoords='axes fraction', ha='left', fontsize=14, fontweight='bold')
    ax[1].annotate(f'SAP flux: quality_bitmask={sap_bitmask}', xy=(0.03, 0.90), xycoords='axes fraction', ha='left', fontsize=14, fontweight='bold')
    
    ax[0].set_ylabel('relative flux')
    ax[1].set_ylabel('relative flux')
    ax[1].set_xlabel('btjd')
    
    plt.suptitle(tildes_to_spaces(pl), fontweight='bold')


# ------------------------------ Astronomy ---------------------------------

def compute_ror(Rp_earth, Rs_sun):
    from astropy.constants import R_earth, R_sun
    return float((Rp_earth * R_earth)/(Rs_sun * R_sun))

def compute_aor(a_au, Rs_sun):
    from astropy.constants import au, R_sun
    return float((a_au * au)/(Rs_sun * R_sun))

def compute_ror_aor_manual(planet):
    row = query_planet(planet, 'nea')
    Rp_earth = row['pl_rade'][0]
    Rs_sun = row['st_rad'][0]
    a_au = row['pl_orbsmax'][0]
    print('ror', compute_ror(Rp_earth, Rs_sun))
    print('aor', compute_aor(a_au, Rs_sun))

def compute_T_eq(T_star, a):
    # a = a/R*
    return T_star * np.sqrt(1/(2*a))

def compute_T_dayside(T_star, a):
    """assume no circulation between dayside and nightside"""
    # a = a/R*
    return compute_T_eq(T_star, a) * 2**(1/4)

def compute_blackbody_flux(T_eff, passband=None):
    raise ValueError('Too slow')
    import astropy.units as u
    import astropy.modeling.models as models
    bb = models.BlackBody(temperature=T_eff*u.K, scale=1*u.watt/(u.m**3 * u.sr))
    if passband is None or passband == 'none':
        return bb.bolometric_flux.to(u.W/u.m**2) * np.pi
    elif passband in ['Kepler', 'kepler']:
        wl, tr = load_kepler_passband()
    elif passband in ['TESS', 'tess']:
        wl, tr = load_tess_passband()
    else:
        raise ValueError

    flux = 0
    n = 100 # number of riemann sum intervals per wavelength bin
    for i in range(len(wl)-1):
        flux += (wl[i+1]-wl[i])/n*(u.meter*1e-9) * np.sum(bb(np.linspace(wl[i], wl[i+1], n) * u.meter * 1e-9)) * (tr[i] + tr[i+1])/2 *np.pi * u.sr
    return flux

def compute_blackbody_se_depth(T_planet, T_star, ror, passband='Kepler'):
    """Return secondary eclipse depth in ppm assuming both planet and star are blackbodies"""
    planet_flux = compute_blackbody_flux(T_planet, passband=passband)
    star_flux = compute_blackbody_flux(T_star, passband=passband)
    return planet_flux/star_flux*ror**2*1e6

def compute_2nd_eclipse_offset(ecc, omega):
    """Compute the offset (in units of rotation period) of secondary eclipse epoch from exactly 1/2 between transits"""
    """ Offset is positive if secondary eclipse is slightly to the right of the center"""
    import scipy.integrate as integrate # https://docs.scipy.org/doc/scipy/tutorial/integrate.html
    e = ecc
    ω = omega
    π = np.pi
    return 1/(2*π*np.sqrt(1-e**2)) * integrate.quad(lambda v: ((1-e**2)/(1+e*np.cos(v))**2), -π/2 - ω, π/2-ω)[0] - 0.5

def _compute_transit_depth(flat_samps, multiplanet=False, planet_id='b', use_trace=True):
    """compute the transit depth using Mandel & Agol formalism instead of
    small planet approximation"""
    """ WARNING, the computation uses an exptime of Kepler, but this distinction doesn't really matter when simply displaying plots"""
    import batman
    mcmc_sample = {}
    keys = ['period', 'a', 'ror', 'b', 'u_star']

    has_ecc = 'ecc' in flat_samps
    if has_ecc:
        keys += ['ecc', 'omega']

    for key in keys:
        if use_trace:
            mcmc_sample[key] = np.array(flat_samps[key+(f'_{planet_id}' if multiplanet and key != 'u_star' else '')])
        else:
           # print('warning this function is probably not working.')
            mcmc_sample[key] = np.array([flat_samps[key+(f'_{planet_id}' if multiplanet and key != 'u_star' else '')]])
    mcmc_sample['inc'] = np.arccos(mcmc_sample['b']/mcmc_sample['a']) * 180/np.pi
    mcmc_sample['u_star'] = np.transpose(mcmc_sample['u_star']) if use_trace else mcmc_sample['u_star']
    if not has_ecc:
        mcmc_sample['ecc'] = np.zeros(len(mcmc_sample['b'])) if use_trace else np.array([0])
        mcmc_sample['omega'] = np.zeros(len(mcmc_sample['b'])) if use_trace else np.array([0])

    params = batman.TransitParams()       #planet to store transit parameters
    params.limb_dark = 'quadratic'
    t = np.linspace(0, np.mean(mcmc_sample['period']), 1000)
    params.t0 = np.max(t)/2
    depths = []
    print('computing transit depth with batman (exptime = KEPLER)')
    for i in range(len(mcmc_sample['period']) if use_trace else 1):
        params.per = mcmc_sample['period'][i]
        params.rp = mcmc_sample['ror'][i]
        params.a = mcmc_sample['a'][i]
        params.inc = mcmc_sample['inc'][i]
        params.u = mcmc_sample['u_star'][i]
        params.ecc = mcmc_sample['ecc'][i]
        params.w = mcmc_sample['omega'][i]

        m = batman.TransitModel(params, t,  supersample_factor = 33, exp_time = 1/48) # Kepler exposure time
        y = m.light_curve(params) - 1
        depths.append(-1 * np.min(y) * 1e6)
    return np.array(depths) if use_trace else depths[0]

def _compute_geometric_albedo(flat_samps, multiplanet=False, planet_id='b'):
    """Assumes no component due to thermal emissio"""
    ror_sample = np.array(flat_samps['ror'+(f'_{planet_id}' if multiplanet else '')])
    a_sample = np.array(flat_samps['a'+(f'_{planet_id}' if multiplanet else '')])
    se_depth_sample = np.array(flat_samps['se_depth'+(f'_{planet_id}' if multiplanet else '')])

    return se_depth_sample*np.square(a_sample/ror_sample)

def generate_sample_from_distr(mean, err_right, err_left, n):
    """err1 is the right tail, err2 is the lower tail"""
    mask = np.random.rand(n) > 0.5
    distr_right = np.abs(np.random.normal(loc=0, scale=err_right, size=n))
    distr_left = -1 * np.abs(np.random.normal(loc=0, scale=err_left, size=n))
    sample = np.ones(n) * mean
    sample[mask] += distr_right[mask]
    sample[~mask] += distr_left[~mask]

    return sample

def sample_se_depth_reflection(params, n):
    """
        params = {ror, a, ror_err1, etc.}
    """
    ror_sample = generate_sample_from_distr(mean=params['ror'], err1=params['ror_err1'], err2=params['ror_err2'], n=n)
    T_star_sample = generate_sample_from_distr(mean=params['T_star'], err1=params['T_star_err1'], err2=params['T_star_err2'], n=n)
    a_sample = generate_sample_from_distr(mean=params['a'], err1=params['a_err1'], err2=params['a_err2'], n=n)
    flat_samps = {'ror': ror_sample, 'T_star': T_star_sample, 'a': a_sample, 'A_g': 1}
    return flat_samps['A_g'] * np.square(flat_samps['ror']/flat_samps['a'])


def _compute_equilibrium_temperature(flat_samps, T_star, T_star_err=0, multiplanet=False, planet_id='b', use_trace=True,
                                        nocirculation=False):
    if T_star is None:
        raise ValueError('Provide T_star')

    a_sample = np.array(flat_samps['a'+(f'_{planet_id}' if multiplanet else '')])

    np.random.seed(93006) # ensure predictable random number generation
    T_star = np.random.normal(loc=T_star, scale=T_star_err, size=len(a_sample))
    np.random.seed()
    T_eq_sample = np.divide(T_star, np.sqrt(2*a_sample))
    if nocirculation:
        T_eq_sample *= 2**(1/4)
    return T_eq_sample

def _compute_sqrtesin_sqrtecos(flat_samps, multiplanet=False, planet_id='b', use_trace=True):
    ecc_sample = np.array(flat_samps['ecc'+(f'_{planet_id}' if multiplanet else '')])
    omega_sample = np.array(flat_samps['omega'+(f'_{planet_id}' if multiplanet else '')])

    sqrtesin = np.sqrt(ecc_sample) * np.sin(omega_sample)
    sqrtecos = np.sqrt(ecc_sample) * np.cos(omega_sample)
    return sqrtesin, sqrtecos

def summarize_eccentricity(flat_samps):
    ecc_sample = flat_samps['ecc']
    omega_sample = flat_samps['omega']

    esin, ecos = ecc_sample * np.sin(omega_sample), ecc_sample * np.cos(omega_sample)
    sqrtesin, sqrtecos = np.sqrt(ecc_sample) * np.sin(omega_sample), np.sqrt(ecc_sample) * np.cos(omega_sample)

    for key, trace_sample in zip(['ecc', 'omega', 'esinω', 'ecosω', 'sqrt(e)sinω', 'sqrt(e)cosω'],[ecc_sample, omega_sample, esin, ecos, sqrtesin, sqrtecos]):
        mcmc = np.nanpercentile(trace_sample, [50-34.135, 50, 50 + 34.135])
        diff = np.diff(mcmc)
        mean = mcmc[1]
        err = (diff[0],diff[1])
        print(key)
        print(mean, err)


def unpack_trace(fin, parameters=['se_depth', 'transit_depth', 'a', 'b'], planet_id=None, multiplanet=False,
                    T_star=None, T_star_err=None, small_planet_approx = False, verbose=False):
    """Load the trace files stored in data/results/MCMC_pdc"""
    try:
        inference_results = pload(fin, verbose=verbose)
        if isinstance(inference_results, list):
            map_soln, fits, _, trace = inference_results
        else:
            map_soln = inference_results['map_soln']
            # fits = inference_results['map_fit']
            trace = inference_results['trace']

        flat_samps = trace.posterior.stack(sample=("chain", "draw"))
        trace_results = {}

        for parameter in parameters:
            if not small_planet_approx and parameter == 'transit_depth':
                """Check for cache so we don't have to keep recomputing with batman"""
                if os.path.exists(fin_cache := f'data/cache/{os.path.basename(fin)}') and os.path.getmtime(fin_cache) > os.path.getmtime(fin):
                    if verbose:
                        print(f'Found updated {fin_cache}')
                    trace_sample = pload(fin_cache, verbose=verbose)
                else:
                    if verbose:
                        print('Computing transit depth using Mandel & Agol (caching when done)')
                    trace_sample = _compute_transit_depth(flat_samps, multiplanet=multiplanet, planet_id=planet_id)
                    psave(trace_sample, fin_cache, verbose=verbose)
            elif parameter == 'T_eq':
                print('Taking equilibrium temperature from assuming no dayside-nightside circulation')
                trace_sample = 2**(1/4)*_compute_equilibrium_temperature(flat_samps, T_star=T_star, T_star_err=T_star_err,
                                                        multiplanet=multiplanet, planet_id=planet_id)
            elif parameter == 'A_g':
                trace_sample = _compute_geometric_albedo(flat_samps,
                                                        multiplanet=multiplanet, planet_id=planet_id)
            elif parameter == 'sqrtesin':
                trace_sample, _ = _compute_sqrtesin_ecos(flat_samps, multiplanet=multiplanet, planet_id=planet_id)
            elif parameter == 'sqrtecos':
                _, trace_sample = _compute_sqrtesin_ecos(flat_samps, multiplanet=multiplanet, planet_id=planet_id)
            elif parameter == 'residual_MAD':
                y = np.transpose(flat_samps['actual_residuals'].data)
                trace_sample = np.nanmedian((np.abs(np.transpose(y)-np.nanmedian(y, axis=1))), axis=0)
            else:
                if not multiplanet or parameter in ['log_sigma_gp', 'log_sigma_lc', 'log_rho_gp',
                                                        'sigma_gp', 'sigma_lc', 'rho_gp']:
                    try:
                        trace_sample = flat_samps[parameter].data
                    except KeyError:
                        trace_sample = [np.nan]
                else:
                    trace_sample = flat_samps[f'{parameter}_{planet_id}'].data
                if parameter in ['se_depth', 'transit_depth']: # small planet approximatino
                    trace_sample = np.copy(trace_sample) * 1e6
            mcmc = np.percentile(trace_sample, [50-34.135, 50, 50 + 34.135])
            diff = np.diff(mcmc)
            mean = mcmc[1]
            err = (diff[0],diff[1])
            trace_results[parameter] = {'mean': mean, 'err': err}
    except (FileNotFoundError, UnpicklingError) as e:
        if verbose:
            print(f'% unpack_trace: {fin} not found or unpickling err: {e}')
        trace_results = {}
        for parameter in parameters:
            trace_results[parameter] = {'mean': np.nan, 'err': (np.nan, np.nan)}

    return trace_results


def load_results(planet, mission='TESS', pipeline='PDCSAP', bitmask='hardest',
                        method='mcmc', mode='quarters/sectors',
                         parameters=['se_depth', 'transit_depth', 'a', 'b', 'sigma_gp', 'sigma_lc', 'rho_gp', 'ecc', 'omega'],
                            use_pdc=True, T_star=None, T_star_err=None, small_planet_approx=False, data_was_binned=False,
                            verbose=False):
    """Read my results for Kepler quarters from either MAP or MCMC. Somewhat convoluted"""
    multiplanet = is_multiplanet(planet, verbose=verbose)
    planet_id = planet[-1]
    method = method.lower()

    results_dir = f'{root}/data/results'
    mcmc_dir = f'{root}/data/MCMC/{planet}'

    if mission is None:
        raise ValueError('Specify mission')

    if mode == 'quarters/sectors':
        mode = 'quarters' if mission == 'Kepler' else 'sectors'

    if mode == 'cumulative':
        if mission == 'TESS':
            fin_results = f'{results_dir}/results.{planet}.sectors.{mission}.{pipeline}.{bitmask}.p'
        else:
            fin_results = f'{results_dir}/results.{planet}.quarters.{mission}.{pipeline}.{bitmask}.p'

        results = pload(fin_results, verbose=verbose)
        results = results[results['span'] == 'cumulative']
    else:
        results = pload(f'{results_dir}/results.{planet}.{mode}.{mission}.{pipeline}.{bitmask}.p', verbose=verbose)
        results = results[results['span'] != 'cumulative']
        if mission == 'Kepler':
            results = results[results['span'] != 'Q0'] # SKIP Q0

    if mode == 'cumulative':
        if method == 'mcmc':
            trace_results = unpack_trace(f'{mcmc_dir}/{planet}_cumulative_MCMC.{mission}.{pipeline}.{bitmask}.p', parameters=parameters,
                                            multiplanet=multiplanet, planet_id=planet_id, T_star=T_star, T_star_err=T_star_err,
                                            small_planet_approx=small_planet_approx)
            mcmc_results = {}
            for parameter in parameters:
                mcmc_results[parameter] = trace_results[parameter]['mean']
                mcmc_results[f'{parameter}_err'] = trace_results[parameter]['err']
            return mcmc_results
        elif method == 'map':
            map_results = {}
            for parameter in parameters:
                map_results[f'{parameter}_err'] = np.array((np.nan, np.nan))
                if results[0]['map_soln'] is None or results[0]['map_soln'] == 0:
                    # print('! nan map result')
                    map_results[f'{parameter}'] = np.nan
                else:
                    if parameter == 'transit_depth' and not small_planet_approx:
                        map_results[parameter] = _compute_transit_depth(results[0]['map_soln'], multiplanet=multiplanet, planet_id=planet_id,
                                                use_trace=False)
                    elif parameter == 'T_eq':
                        map_results[parameter] = _compute_equilibrium_temperature(results[0]['map_soln'], T_star=T_star, T_star_err=T_star_err,
                                        multiplanet=multiplanet, planet_id=planet_id, use_trace=False)
                    elif parameter == 'A_g':
                        map_results[parameter] = _compute_geometric_albedo(results[0]['map_soln'], multiplanet, planet_id)
                    elif parameter == 'u_star_0':
                        map_results[parameter] = map_results['u_star'][0]
                    elif parameter == 'u_star_1':
                        map_results[parameter] = map_results['u_star'][1]
                    else:
                        try:
                            map_results[parameter] = float(results[0]['map_soln'][f'{parameter}_{planet_id}' if multiplanet and 'gp' not in parameter and 'lc' not in parameter else parameter])
                        except KeyError:
                            map_results[parameter] = np.nan
                        if parameter in ['se_depth', 'transit_depth']:
                            map_results[parameter] *= 1e6
                            map_results[f'{parameter}_err'] *= 1e6
            return map_results
        else:
            raise ValueError
    else:
        if mission in ['TESS']:
            results_tb = pload(f'{results_dir}/results.{planet}.{mode}.{mission}.{pipeline}.{bitmask}.p', verbose=verbose) # get sectors
            sectors = []
            sector_groups = []
            trace_results = {}
            map_results = {}
            for parameter in parameters:
                map_results[parameter] = []
                map_results[f'{parameter}_err'] = []
            for span in list(results_tb['span']):
                if span == 'cumulative':
                    continue
                if mode in ['sectors']:
                    try:
                        s = int(span[1:])
                    except ValueError:
                        s = float(span[1:])
                elif mode in ['sectors_grouped']:
                    s = np.mean(sector_group := [int(x) for x in span[1:].split('.')])
                    sector_groups.append(sector_group)
                else:
                    raise ValueError
                sectors.append(s)

                if method == 'mcmc':
                    trace_results[s] = unpack_trace(f'{mcmc_dir}/{planet}_{span}_MCMC.{mission}.{pipeline}.{bitmask}.p', parameters=parameters,
                                        multiplanet=multiplanet, planet_id=planet_id, T_star=T_star, T_star_err=T_star_err, small_planet_approx=small_planet_approx,
                                        verbose=verbose)
                    mcmc_results = {}
                    for parameter in parameters:
                        mcmc_results[parameter] = np.array([trace_results[s][parameter]['mean'] for s in sectors])
                        mcmc_results[f'{parameter}_err'] = np.array([trace_results[s][parameter]['err'] for s in sectors])
                elif method == 'map':
                    for parameter in parameters:
                        try:
                            row = results_tb[results_tb['span']==span]
                            map_results[f'{parameter}_err'].append((np.nan, np.nan))
                            if row[0]['failed']:
                                map_results[parameter].append(np.nan)
                            else:
                                map_results[parameter].append(float(row[0]['map_soln'][f'{parameter}_{planet_id}' if multiplanet and 'gp' not in parameter and 'lc' not in parameter else parameter]))
                        except (IndexError, KeyError):
                            map_results[parameter].append(np.nan)

            for parameter in parameters: # No need to reuse this for mcmc solutions. Unpack trace already does it
                map_results[parameter] = np.array(map_results[parameter])
                map_results[f'{parameter}_err'] = np.array(map_results[f'{parameter}_err'])
                if parameter in ['se_depth', 'transit_depth']:
                    map_results[parameter] *= 1e6
                    map_results[f'{parameter}_err'] *= 1e6

        if mission in ['Kepler']:
            if mode in ['quarters', 'quarter', 'twelfths', 'twelfth']:
                qs = np.arange(1, 17+1)
            elif mode in ['window', 'windows']:
                qs = np.arange(1, 14+1)
            if method == 'mcmc':
                trace_results = {}
                for q in tqdm(qs):
                    if mode in ['quarters', 'quarter']:
                        trace_results[q] = unpack_trace(f'{mcmc_dir}/{planet}_Q{q}_MCMC.{mission}.{pipeline}.{bitmask}.p', parameters=parameters,
                                        multiplanet=multiplanet, planet_id=planet_id, T_star=T_star, T_star_err=T_star_err, small_planet_approx=small_planet_approx,
                                        verbose=verbose)
                    elif mode in ['window', 'windows']:
                        raise ValueError
                    elif mode in ['twelfth', 'twelfths']:
                        raise ValueError
                    else:
                        raise ValueError
                if mode == 'twelfths':
                    qs = trace_results.keys()

                mcmc_results = {}
                for parameter in parameters:
                    mcmc_results[parameter] = np.array([trace_results[q][parameter]['mean'] for q in qs])
                    mcmc_results[f'{parameter}_err'] = np.array([trace_results[q][parameter]['err'] for q in qs])
            elif method == 'map':
                map_results = {}
                for parameter in parameters:
                    map_results[parameter] = []
                    map_results[f'{parameter}_err'] = []
                if mission in ['Kepler']:
                    if mode in ['quarters', 'quarter']:
                        for parameter in parameters:
                            for q in qs:
                                try:
                                    row = results[results['span']==f'Q{q}']
                                    if parameter == 'u_star_0':
                                        map_results[parameter].append(row[0]['map_soln']['u_star'][0])
                                    elif parameter == 'u_star_1':
                                        map_results[parameter].append(row[0]['map_soln']['u_star'][1])
                                    else:
                                        map_results[parameter].append(float(row[0]['map_soln'][f'{parameter}_{planet_id}' if multiplanet and 'gp' not in parameter and 'lc' not in parameter else parameter]))
                                    map_results[f'{parameter}_err'].append((np.nan, np.nan))
                                except IndexError:
                                    map_results[parameter].append(np.nan)
                                    map_results[f'{parameter}_err'].append((np.nan, np.nan))
                    elif mode in ['window', 'windows']:
                        for parameter in parameters:
                            for q in qs:
                                try:
                                    row = results[results['span']==f'W{q}']
                                    map_results[parameter].append(float(row[0]['map_soln'][f'{parameter}_{planet_id}' if multiplanet and 'gp' not in parameter and 'lc' not in parameter else parameter]))
                                    map_results[f'{parameter}_err'].append((np.nan, np.nan))
                                except IndexError:
                                    map_results[parameter].append(np.nan)
                                    map_results[f'{parameter}_err'].append((np.nan, np.nan))
                    elif mode in ['twelfth', 'twelfths']:
                        for parameter in parameters:
                            for q in qs:
                                for j in range(1, 3+1):
                                    try:
                                        row = results[results['span']==f'Q{q}_{j}']
                                        map_results[parameter].append(float(row[0]['map_soln'][f'{parameter}_{planet_id}' if multiplanet and 'gp' not in parameter and 'lc' not in parameter else parameter]))
                                        map_results[f'{parameter}_err'].append((np.nan, np.nan))
                                    except IndexError:
                                        map_results[parameter].append(np.nan)
                                        map_results[f'{parameter}_err'].append((np.nan, np.nan))
                for parameter in parameters: # No need to reuse this for mcmc solutions. Unpack trace already does it
                    map_results[parameter] = np.array(map_results[parameter])
                    map_results[f'{parameter}_err'] = np.array(map_results[f'{parameter}_err'])
                    if parameter in ['se_depth', 'transit_depth']:
                        map_results[parameter] *= 1e6
                        map_results[f'{parameter}_err'] *= 1e6
            qs = np.array(qs)

        if mode in ['twelfth', 'twelfths']:
            qs = np.arange(2, 52+1)/3

        if mission == 'TESS':
            qs = np.array(sectors)

        if mode in ['sectors_grouped']:
            return qs, mcmc_results, sector_groups
        else:
            if method == 'mcmc':
                return qs, mcmc_results
            elif method == 'map':
                return qs, map_results

load_mcmc_results = load_results

# def plot_depths(qs, eds, edes, tds, tdes, ax=None):
#     """Plot depths as a function of quarter/4-quarter,window"""
#     if ax is None:
#         fig, ax = plt.subplots(2, 1, figsize=(12, 6))
#     ax[0].set_title('Secondary Eclipse Depths')
#     ax[0].errorbar(qs, eds, edes, marker='o', capsize=1)
#     ax[0].set_xticks(np.arange(1, 18))
#     ax[1].set_title('Transit Depths')
#     ax[1].errorbar(qs, tds, tdes, marker='o', capsize=1)
#     ax[1].set_xticks(np.arange(1, 18))

def compute_transit_depth(ror, b, u_star):
    """Assumes quadratic limb darkening. Returns transit depth in ppm"""
    u1, u2 = u_star
    f0 = 1 - 2 * u1 / 6.0 - 2 * u2 / 12.0
    arg = 1 - np.sqrt(1 - b**2)
    f = 1 - u1 * arg - u2 * arg**2
    factor = f0/f
    depth = ror**2/factor
    return depth


def compute_geometric_albedo(ror, a, se_depth):
    """See Heng 2017. a is a/R_star"""
    return se_depth * (a/ror)**2

# def compute_transit_depth_and_error(ror, dror, a, da, i, di, u_1, du_1, u_2, du_2):
#     """compute error using traditional error propagation rules and first order approximation."""
#     print('dD is wrong!')
#     try:
#         I = i * np.pi/180
#         phi = 1 - np.sqrt(1-a**2*np.cos(I)**2) # phi = 1 - μ
#         U = 1 - 2/6 * u_1 - 2/12 * u_2
#         L = 1 - phi * u_1 - phi**2 * u_2
#         factor = U/L
#         D = ror**2/factor

#         dI = di * np.pi/180
#         dphi = np.sqrt((a*da/np.tan(I)**2/(1-phi))**2 + (a**2*np.cos(I)**-2*dI/(1-phi)/(np.tan(I)**3))**2)
#         dD = D * np.sqrt((2*dror/ror)**2 + 1/U**2*((2/6*du_1)**2+(2/12*u_2)**2) + 1/L**2 * ((phi*du_1)**2 + (u_1*dphi)**2 + (phi**2*du_2)**2+(2*u_2*phi*dphi)**2))

#         return D, dD
#     except TypeError:
#         return D, np.nan

# ------------------------------ Fitting  ---------------------------------
# def detrend_with_GP(t, y, yerr, priors, start=None):
    # Go to timeseriestools

# def create_gp_priors(std, gp_preset):
#     import pymc3 as pm
#     """Hold on this is stupid"""
#     match gp_preset:
#         case 'kepler': # Li and Shporer 2024
#             rho_gp = pm.Uniform('rho_gp', lower=0.1, upper=100)
#             sigma_gp = pm.Uniform('sigma_gp', lower=0, upper=2.0*std)
#             sigma_lc = pm.Uniform('sigma_lc', lower=0, upper=500e-6) # SPECIFIC TO KEPLER DATA!!!
#         case 'tess': # TESS Science Conference 3
#             rho_gp = pm.Uniform('rho_gp', lower=0.1, upper=100)
#             sigma_gp = pm.Uniform('sigma_gp', lower=0, upper=2.0*std)
#             sigma_lc = pm.Uniform('sigma_lc', lower=0, upper=50000e-6) # SPECIFIC TO TESS DATA!!!
#         case 'tess2':
#             rho_gp = pm.Uniform('rho_gp', lower=0.1, upper=100)
#             sigma_gp = pm.Uniform('sigma_gp', lower=0, upper=2.0*std)
#             sigma_lc = pm.Uniform('sigma_lc', lower=0, upper=2.0*std) # SPECIFIC TO TESS DATA!!!
#         case 'tess_long':
#             rho_gp = pm.Uniform('rho_gp', lower=5, upper=100)
#             sigma_gp = pm.Uniform('sigma_gp', lower=0, upper=2.0*std)
#             sigma_lc = pm.Uniform('sigma_lc', lower=0, upper=1e-6) # SPECIFIC TO TESS DATA!!!
#         case _:
#             raise ValueError('Provide a valid value for gp_preset')

#     return rho_gp, sigma_gp, sigma_lc

def get_prior(priors, key, default):
    if key not in priors.keys() or priors[key] == 'missing':
        return default
    else:
        return priors[key]

def generate_synthetic_curve(params, t=None, noise=100e-6):
    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx

    if t is None:
        texp = 1/720 # TESS
        texp = 1/48 # Kepler
        t_model = np.arange(int(27/texp)) * texp
    else:
        texp = np.min(np.diff(t))
        t_model = t
    
    yerr_model = np.ones(len(t_model)) * noise

    with pm.Model() as model:
        """Baseline and ephemeris parameters"""
        t0 = params['t0']
        period = params['period']
        

        """Orbit shape parameters"""
        r_star = 1
        a = params['a']
        ror = params['ror']
        b = params['b']
        r_planet = ror * r_star
        """Limb darkening and 2nd eclipse parameters"""
        u_star = np.array([0.3, 0.2])
        u_planet = np.array([0.0, 0.0])

        se_depth = params['se_depth']

        surface_brightness_ratio = se_depth/ror**2

        """Set up a Keplerian orbit for the planets"""
        if 'ecc' in params:
            ecc = params['ecc']
            omega = params['omega']
            orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, r_star=r_star, a=a,
                            ecc=ecc, omega=omega)
        else:
            orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, r_star=r_star, a=a)

        light_curves = xo.SecondaryEclipseLightCurve(u_primary = u_star, u_secondary = u_planet,
                        surface_brightness_ratio=surface_brightness_ratio).get_light_curve(
                        orbit=orbit, r=r_planet, t=t_model, texp=texp)
        light_curve = pm.math.sum(light_curves, axis=-1)
        y = pmx.eval_in_model(light_curves) 
        y_model = np.ascontiguousarray(np.transpose(y)[0])
    
    return t_model, y_model, yerr_model

def fit_secondary_eclipse_minimal(t, y, yerr, priors, return_model=False, start=None,
                                    fit_eccentricity = False, fit_planet_limbdarkening=False,
                                    progressbar=True, fix_2nd_eclipse_depth_to_zero=False, 
                                    confidence_in_priors='strong',
                                    # gp_preset='tess', 
                                    diagnostic_mode=False):
    """
    Optimize for the parameters of a secondary eclipse light curve. Doesn't sample MCMC

        priors : dict of initial values for distributions. t0 and period are required
            priors.keys(): 't0', 'period', 'a', etc

    """
    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx
    import aesara_theano_fallback.tensor as tt
    from celerite2.theano import terms, GaussianProcess

    texp = np.min(np.diff(t)) # Account for the blending effect from exposure time

    assert 't0' in priors.keys() and priors['t0'] not in ['missing', np.nan]
    assert 'period' in priors.keys() and priors['period'] not in ['missing', np.nan]
    if fit_eccentricity:
        print('%%%%%%%%% Fitting eccentricity')
    lc_std = np.std(y[~get_transit_mask(t, y, ephemeris=priors)]) # std of out of transit data

    with pm.Model() as model:
        """Baseline and ephemeris parameters"""
        mean = pm.Normal('mean', mu=0.0, sd=10.0)
        t0 = pm.Normal('t0', mu=priors['t0'], sd=0.1)
        period = pm.Normal('period', mu=priors['period'], sd=0.1)

        """Orbit shape parameters"""
        r_star = 1
        # a = pm.Uniform('a', lower=1, upper=1000, testval=get_prior(priors, 'a', 4))
        log_a = pm.Normal('log_a', mu=np.log(get_prior(priors, 'a', 4)), sd=10)
        a = pm.Deterministic('a', tt.exp(log_a))
        ror = pm.Uniform('ror', lower=0.0, upper=1.0, testval=get_prior(priors, 'ror', 0.06))
        b = pm.Uniform('b', lower=0.0, upper=1.0, testval=min(get_prior(priors, 'b', 0.01), 0.99)) # Sometimes the NEA value will be > 1 (grazing)
        r_planet = ror * r_star
        if fit_eccentricity:
            # if 'ecc' in priors.keys():
            #     print('Using known priors for eccentricity')
            #     ecc = pm.TruncatedNormal('ecc', mu=priors['ecc'], sd=priors['ecc_err'], lower=0, upper=1)
            #     omega = pm.TruncatedNormal('omega', mu=priors['omega']*np.pi/180, sd=priors['omega_err']*np.pi/180)
            # else:

            ecs = pmx.UnitDisk("ecs", testval=np.array([
                get_prior(priors, 'sqrt(e)cosω', 0.01),
                get_prior(priors, 'sqrt(e)sinω', 0.00)
            ]))
            ecc = pm.Deterministic("ecc", tt.sum(ecs**2))
            omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
            xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)


        """Limb darkening and 2nd eclipse parameters"""
        u_star = xo.distributions.QuadLimbDark('u_star', testval=get_prior(priors, 'u_star', np.array([0.3, 0.2])))
        if fit_planet_limbdarkening:
            print('%%%%%%%%%%%%%%%%%% FITTING PLANET LIMB DARKENING!!!')
            u_planet = xo.distributions.QuadLimbDark('u_planet', testval=np.array([0.3, 0.2]))
        else:
            u_planet = [0, 0] # uniform brightness

        sed_prior = (-1e-3, 1e-3, 1e-6) # Allow a negative 2nd eclipse depth
        if fix_2nd_eclipse_depth_to_zero:
            print_bold('Fixing 2nd eclipse depth to zero')
            se_depth = 0
        else:
            se_depth = pm.Uniform('se_depth', lower=sed_prior[0], upper=sed_prior[1], testval=get_prior(priors, 'se_depth', sed_prior[2]));
        print(f'Prior for 2nd eclipse depth: {tuple(np.array(sed_prior)*1e6)} ppm')

        surface_brightness_ratio = pm.Deterministic('surface_brightness_ratio', se_depth/ror**2)
        pm.Deterministic('transit_depth', compute_transit_depth(ror=ror, b=b, u_star=(u_star[0], u_star[1])))
        """Set up a Keplerian orbit for the planets"""
        if fit_eccentricity:
            orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, r_star=r_star, a=a,
                            ecc=ecc, omega=omega)
        else:
            orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, r_star=r_star, a=a)

        """Compute the model light curve using starry"""
        light_curves = xo.SecondaryEclipseLightCurve(u_primary = u_star, u_secondary = u_planet,
                        surface_brightness_ratio=surface_brightness_ratio).get_light_curve(
                        orbit=orbit, r=r_planet, t=t, texp=texp)
        light_curve = pm.math.sum(light_curves, axis=-1) + mean
        pm.Deterministic('lc_pred', light_curve)

        """Fit residuals with GP"""
        residuals = y - light_curve
        # print(f'Using GP preset: {gp_preset}')
        # rho_gp, sigma_gp, sigma_lc = create_gp_priors(std, gp_preset=gp_preset)
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

        gp = GaussianProcess(kernel, t=t, diag=yerr**2 + tt.exp(log_sigma_lc)**2)
        gp.marginal('gp', observed=residuals)
        pm.Deterministic('gp_pred', gp.predict(residuals))

        """Optimize for MAP parameters
           Different from least squares (see https://emcee.readthedocs.io/en/stable/tutorials/line/) """
        if start is None:
            start = model.test_point
        map_soln = start

        if fix_2nd_eclipse_depth_to_zero:
            vars = [ror, b, period, t0, log_a, u_star, mean]
        else:
            vars = [ror, b, period, t0, log_a, se_depth, u_star, mean]
        if fit_eccentricity:
            vars.append(ecs)
        if fit_planet_limbdarkening:
            vars.append(u_planet)

        print(f'{confidence_in_priors = }')
        
        count = 0

        def plot_map_soln(map_soln,title=''):
            nonlocal count
            lc_pred, gp_pred = pmx.eval_in_model([light_curves, gp.predict(residuals)], map_soln)
            plt.figure()
            plt.scatter(t,y)
            plt.plot(t,lc_pred, lw=2, color='tab:orange', zorder=1000)
            plt.plot(t,gp_pred, lw=2, color='tab:green', zorder=1000)
            plt.savefig(f'/users/canis/downloads/{count}.png')
            plt.title(count)
            count += 1

        match confidence_in_priors:
            case 'strong':
                # map_soln = pmx.optimize(start=map_soln, vars=[t0, mean, ror])
                # map_soln = pmx.optimize(start=map_soln, vars=[t0, ror, log_a, b, u_star, se_depth]) 
                # map_soln = pmx.optimize(start=map_soln, vars=[mean,log_sigma_lc, log_sigma_gp, log_rho_gp]) 
                # map_soln = pmx.optimize(start=map_soln, vars=[ecs, se_depth] if fit_eccentricity else [se_depth])
                # map_soln = pmx.optimize(start=map_soln, vars=vars) # Everything but gp params
                # map_soln = pmx.optimize(start=map_soln, vars=[mean,log_sigma_lc, log_sigma_gp, log_rho_gp])
                # map_soln = pmx.optimize(start=map_soln, vars=[ecs, se_depth] if fit_eccentricity else [se_depth])
                # map_soln = pmx.optimize(start=map_soln)
                map_soln = pmx.optimize(start=map_soln, vars=vars)
                map_soln = pmx.optimize(start=map_soln)

            case 'weak':
                map_soln = pmx.optimize(start=map_soln, vars=vars+[log_sigma_lc, log_sigma_gp, log_rho_gp], progress=progressbar)
            case _:
                raise ValueError('Provide a valid value for confidence_in_priors')

        fits = dict(
            zip(
                ['lc_pred', 'gp_pred'],
                pmx.eval_in_model([light_curves, gp.predict(residuals)], map_soln),
            )
        )
        fits['lc_pred'] = np.transpose(fits['lc_pred'])[0]
        fits['t'] = t

    if return_model:
        return map_soln, fits, model
    else:
        return map_soln, fits

def fit_secondary_eclipse_multiplanet(t, y, yerr, priors, return_model=False, start=None, progressbar=True, data_is_from_tess=False):
    """
    TODO - Combine this with above function
    Optimize for the parameters of a seconda ry eclipse light curve. Doesn't sample MCMC

        priors : dict of initial values for distributions. For each exoplanet letter, t0 and period are required
            priors.keys(): 'b', 'c', etc.
            priors['a'].keys(): 't0', 'period', 'a', etc.
    """
    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx
    import aesara_theano_fallback.tensor as tt
    from celerite2.theano import terms, GaussianProcess

    texp = np.min(np.diff(t)) # Account for the blending effect from exposure time

    planets = [x for x in sorted(priors.keys()) if len(x) == 1]
    num_planets = len(planets)
    print(f'{num_planets} planets: {planets}')

    with pm.Model() as model:
        """Baseline and ephemeris parameters"""
        mean = pm.Normal('mean', mu=0.0, sd=10.0)
        t0s = [pm.Normal(f't0_{p}', mu=priors[p]['t0'], sd=0.1) for p in planets]
        periods = [pm.Normal(f'period_{p}', mu=priors[p]['period'], sd=0.1) for p in planets]

        """Orbit shape parameters"""
        r_star = 1
        a_s = [pm.Uniform(f'a_{p}', lower=1, upper=1000, testval=get_prior(priors[p], 'a', 4)) for p in planets]
        rors = [pm.Uniform(f'ror_{p}', lower=0.0, upper=0.5, testval=get_prior(priors[p], 'ror', 0.06)) for p in planets]
        r_planets = [ror * r_star for ror in rors]
        bs = []
        for i, p in enumerate(planets):
            bs.append(pm.Uniform(f'b_{p}', lower=0.0, upper=1.0, testval=get_prior(priors[p], 'b', 0.01)))

        """GP Parameters"""
        BoundedNormal = pm.Bound(pm.Normal, lower=0)
        mask = np.full(len(t), False)
        for p in planets:
            mask = np.logical_or(mask, get_transit_mask(t, y, ephemeris=priors[p]))
        std = np.std(y[~mask]) # std of out of transit data
        rho_gp, sigma_gp, sigma_lc = create_gp_priors(std, data_is_from_tess=data_is_from_tess)

        """Secondary Eclipse parameters"""
        u_star = xo.distributions.QuadLimbDark('u_star', testval=get_prior(priors, 'u_star', np.array([0.3, 0.2])))
        u_planet = [0,0] # uniform brightness
        se_depths = [pm.Uniform(f'se_depth_{p}', lower=-1e-3, upper=1e-3, testval=get_prior(priors[p], 'se_depth', 0.0)) for p in planets]
        surface_brightness_ratios = []
        transit_depths = []
        for i, p in enumerate(planets):
            surface_brightness_ratios.append(pm.Deterministic(f'surface_brightness_ratio_{p}', se_depths[i]/rors[i]**2))
            transit_depths.append(pm.Deterministic(f'transit_depth_{p}', compute_transit_depth(ror=rors[i], b=bs[i], u_star=(u_star[0], u_star[1]))))

        """Set up a Keplerian orbit for the planets"""
        orbits = []
        light_curves = []
        for i, p in enumerate(planets):
            orbits.append(xo.orbits.KeplerianOrbit(period=periods[i], t0=t0s[i], b=bs[i], r_star=r_star, a=a_s[i]))
            """Compute the model light curve using starry"""
            light_curves.append(xo.SecondaryEclipseLightCurve(u_primary = u_star, u_secondary = u_planet,
                        surface_brightness_ratio=surface_brightness_ratios[i]).get_light_curve(
                        orbit=orbits[i], r=r_planets[i], t=t, texp=texp))
        light_curve = mean
        for x in light_curves:
            light_curve += pm.math.sum(x, axis=-1)

        """Fit residuals with GP"""
        residuals = y - light_curve
        use_matern32 = False
        if use_matern32:
            print('!!!!!!!!!!!!!!!!!!! USING MATERN')
            kernel = terms.Matern32Term(
                sigma=sigma_gp,
                rho=rho_gp,
                #Q=1/np.sqrt(2),
            )
        else:
            print('Using SHOTerm')
            kernel = terms.SHOTerm(
                sigma=sigma_gp,
                rho=rho_gp,
                Q=1/np.sqrt(2),
            )

        gp = GaussianProcess(kernel, t=t, diag=yerr**2 + sigma_lc**2)
        gp.marginal('gp', observed=residuals)

        """Optimize for MAP parameters
           Different from least squares (see https://emcee.readthedocs.io/en/stable/tutorials/line/) """
        if start is None:
            start = model.test_point
        map_soln = start
        map_soln = pmx.optimize(start=map_soln, vars=[sigma_lc, sigma_gp, rho_gp], progress=progressbar)
        map_soln = pmx.optimize(start=map_soln, progress=progressbar)

        fits = dict(
            zip(
                [*[f'light_curve_{p}' for p in planets], 'gp_pred'],
                pmx.eval_in_model([*light_curves, gp.predict(residuals)], map_soln),
            )
        )
        for p in planets:
            fits[f'light_curve_{p}'] = np.transpose(fits[f'light_curve_{p}'])[0]
        fits['light_curve'] = np.sum([fits[f'light_curve_{p}'] for p in planets], axis=0)

    if return_model:
        return map_soln, fits, model
    else:
        return map_soln, fits


# def fit_secondary_eclipse_active(t, y, yerr, priors, return_model=False, start=None, progressbar=True):
#     """
#     Optimize for the parameters of a secondary eclipse light curve. Doesn't sample MCMC

#         priors : dict of initial values for distributions. t0 and period are required
#             priors.keys(): 't0', 'period', 'a', etc
#     """
#     import exoplanet as xo
#     import pymc3 as pm
#     import pymc3_ext as pmx
#     import aesara_theano_fallback.tensor as tt

#     texp = np.min(np.diff(t)) # Account for the blending effect from exposure time

#     assert 't0' in priors.keys()
#     assert 'period' in priors.keys()

#     with pm.Model() as model:
#         """Baseline and ephemeris parameters"""
#         mean = pm.Normal('mean', mu=0.0, sd=10.0)
#         t0 = pm.Normal('t0', mu=priors['t0'], sd=0.1)
#         period = pm.Normal('period', mu=priors['period'], sd=0.1)

#         """Orbit shape parameters"""
#         r_star = 1
#         a = pm.Uniform('a', lower=1, upper=1000, testval=priors.get('a', 4))
#         ror = pm.Uniform('ror', lower=0.0, upper=0.5, testval=priors.get('ror', 0.06))
#         r_planet = ror * r_star
#         b = xo.distributions.ImpactParameter('b', ror=ror, testval=priors.get('b', 0.0))

#         """GP Parameters"""
#         # A jitter term describing excess white noise
#         log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sd=2)
#         sigma_rot = pm.InverseGamma("sigma_rot", **pmx.estimate_inverse_gamma_parameters(1, 5))
#         log_prot = pm.Normal('log_prot', mu=np.log(priors['prot']), sd=0.02)
#         prot = pm.Deterministic("prot", tt.exp(log_prot))
#         log_Q0 = pm.Normal("log_Q0", mu=0, sd=2)
#         log_dQ = pm.Normal("log_dQ", mu=0, sd=2)
#         f = pm.Uniform("f", lower=0.01, upper=1)

#         """Secondary Eclipse parameters"""
#         u_star = xo.distributions.QuadLimbDark('u_star', testval=priors.get('u_star', np.array([0.3, 0.2])))
#         u_planet = [0, 0] # uniform brightness
#         se_depth = pm.Uniform('se_depth', lower=-1e-3, upper=1e-3, testval=priors.get('se_depth', 0))
#         surface_brightness_ratio = pm.Deterministic('surface_brightness_ratio', se_depth/ror**2)
#         pm.Deterministic('transit_depth', compute_transit_depth(ror=ror, b=b, u_star=(u_star[0], u_star[1])))

#         """Set up a Keplerian orbit for the planets"""
#         orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, r_star=r_star, a=a)

#         """Compute the model light curve using starry"""
#         light_curves = xo.SecondaryEclipseLightCurve(u_primary = u_star, u_secondary = u_planet,
#                         surface_brightness_ratio=surface_brightness_ratio).get_light_curve(
#                         orbit=orbit, r=r_planet, t=t, texp=texp)
#         light_curve = pm.math.sum(light_curves, axis=-1) + mean
#         residuals = y - light_curve

#         """Fit residuals with GP"""
#         kernel = terms.RotationTerm(
#             sigma=sigma_rot,
#             period=prot,
#             Q0=tt.exp(log_Q0),
#             dQ=tt.exp(log_dQ),
#             f=f,
#         )

#         gp = GaussianProcess(
#             kernel,
#             t=t,
#             diag=yerr ** 2 + tt.exp(2 * log_jitter),
#             quiet=True,
#         )
#         gp.marginal('gp', observed=residuals)

#         print('hello')
#         """Optimize for MAP parameters
#            Different from least squares (see https://emcee.readthedocs.io/en/stable/tutorials/line/) """
#         if start is None:
#             map_soln = pmx.optimize(start=model.test_point, progress=progressbar)
#         else:
#             map_soln = pmx.optimize(start=start, progress=progressbar)

#         fits = dict(
#             zip(
#                 ['light_curve', 'gp_pred'],
#                 pmx.eval_in_model([light_curves, gp.predict(residuals)], map_soln),
#             )
#         )
#         fits['light_curve'] = np.transpose(fits['light_curve'])[0]

#     if return_model:
#         return map_soln, fits, model
#     else:
#         return map_soln, fits


def lmfit_transit(t, y, texp, t0, period, return_model=True, plot=True, ax=None, dirout=None, verbose=True):
    """Least squares fit for a transit with batman. Can input phase folded data"""

    """Define model and params"""
    import batman
    from canislib.fitting import Fitter
    params = batman.TransitParams()       #planet to store transit parameters
    params.t0 = t0                     #time of inferior conjunction
    params.per = period                       #orbital period
    params.rp = 0.063                       #planet radius (in units of stellar radii)
    params.a = 4.29                        #semi-major axis (in units of stellar radii)
    params.inc = 87.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 0.                        #longitude of periastron (in degrees) doesn't matter if e=0
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = [0.3, 0.2]      #limb darkening coefficients [u1, u2, u3, u4]
    supersample_factor=21

    lm_params = {}
    lm_params['rp'] = 0.063
    lm_params['a'] = 4.29
    lm_params['u1'] = 0.3
    lm_params['u2'] = 0.2
    lm_params['inc'] = 87
    m = batman.TransitModel(params, t, supersample_factor=supersample_factor, exp_time=texp)
    fac = m.fac
    def transit_lc(t, lm_params):
        m = batman.TransitModel(params, t, fac=fac, supersample_factor=supersample_factor, exp_time=texp)
        params.a = lm_params['a']
        params.inc = lm_params['inc']
        params.rp = lm_params['rp']
        params.u = [lm_params['u1'], lm_params['u2']]
        return m.light_curve(params) -1

    """Run fitter"""
    transit_fitter = Fitter(t, y, lm_params, transit_lc, verbose=verbose)
    transit_fitter.set_vary(['rp', 'a', 'inc'])
    transit_fitter.set_vary(['u1', 'u2'])
    transit_fitter.set_limits('rp', 0.01, 0.5)
    transit_fitter.set_limits('a', 1, 15)
    transit_fitter.set_limits('u1', -1.0,  1.0)
    transit_fitter.set_limits('u2', -1.0, 1.0)
    transit_fitter.set_limits('inc', 70, 90)

    transit_fitter.run_minimizer()
    transit_lmfit_params = transit_fitter.get_params()
    yfit = transit_fitter.get_fit()
    residuals = (y-yfit)*1e6
    if verbose:
        print(transit_fitter.get_result())

    # """Compute depth and residuals"""
    # params_copy = copy.deepcopy(params)
    # params_copy.t0 = t0
    # params_copy.a = transit_lmfit_params['a']
    # params_copy.inc = transit_lmfit_params['inc']
    # params_copy.rp = transit_lmfit_params['rp']
    # params_copy.u = [transit_lmfit_params['u1'], transit_lmfit_params['u2']]
    # yfit_unfolded = (batman.TransitModel(params_copy, t, fac=fac, supersample_factor=supersample_factor, exp_time=texp).light_curve(params_copy) - 1)
    # y_notransit = y - yfit_unfolded # remove transit signal

    from canislib.exoplanets import compute_transit_depth_and_error
    ror = transit_lmfit_params['rp'].value
    dror = transit_lmfit_params['rp'].stderr
    a = transit_lmfit_params['a'].value
    da = transit_lmfit_params['a'].stderr
    i = transit_lmfit_params['inc'].value
    di = transit_lmfit_params['inc'].stderr
    u_1 = transit_lmfit_params['u1'].value
    du_1 = transit_lmfit_params['u1'].stderr
    u_2 = transit_lmfit_params['u2'].value
    du_2 = transit_lmfit_params['u2'].stderr
    transit_depth, transit_depth_err = compute_transit_depth_and_error(ror, dror, a, da, i, di, u_1, du_1, u_2, du_2)
    transit_depth *= 1e6
    transit_depth_err *= 1e6

    phase = (t-t0)/period

    if plot:
        if ax is None:
            fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(8,6), sharex=True)

        if ax[0] is not None:
            ax[0].set_title(f'Transit fit (error is from covariance matrix)')
            ax[0].scatter(phase, y*1e6)
            ax[0].plot(phase, yfit * 1e6, linewidth=2, color='tab:orange')
            ax[0].axvline(0)
            ax[0].axhline(-1*transit_depth, color='blue', linewidth=2, label=f'δ={np.round(transit_depth)}±{np.round(transit_depth_err)}ppm')
            ax[0].axhspan(-1*transit_depth-transit_depth_err, -1*transit_depth+transit_depth_err, color='blue', alpha=0.3, label='1σ')
            ax[0].legend()
            c_delay = 67.0191/3600/24

        if ax[1] is not None:
            ax[1].set_title(f'residuals')
            ax[1].scatter(phase, residuals)
            #within_5_sigma = np.abs(residuals) < 5 * np.std(residuals)
            #ax[1].set_title(f'residuals within 5σ ({fstr_ratio(np.sum(within_5_sigma), len(residuals))})')
            #ax[1].scatter(phase[within_5_sigma], residuals[within_5_sigma])

        if dirout is not None:
            plt.savefig(f'{dirout}/LS-transit-fit')

    if return_model:
        return transit_depth, transit_depth_err, transit_lmfit_params, yfit
    else:
        return transit_depth, transit_depth_err

def lmfit_secondary_eclipse(t, y, texp, t0, period, transit_lmfit_params, return_model=False, plot=True, ax=None, dirout=None, verbose=True):
    """Least squares fit for a secondary eclipse with batman. Takes in the transit params from `lmfit_transit`
     Do not input phase folded data"""
    import batman

    """Give it the fitted transit parameters"""
    from canislib.fitting import Fitter
    b_params = batman.TransitParams()       #planet to store transit parameters
    b_params.t0 = t0 + period/2
    b_params.t_secondary = t0                     #time of inferior conjunction
    b_params.per = period                       #orbital period
    b_params.rp = transit_lmfit_params['rp'].value                      #planet radius (in units of stellar radii)
    b_params.a = transit_lmfit_params['a'].value                       #semi-major axis (in units of stellar radii)
    b_params.inc = transit_lmfit_params['inc'].value                      #orbital inclination (in degrees)
    b_params.ecc = 0.                       #eccentricity
    b_params.w = 0.                        #longitude of periastron (in degrees) doesn't matter if e=0
    b_params.limb_dark = "quadratic"        #limb darkening model
    b_params.u = [transit_lmfit_params['u1'].value, transit_lmfit_params['u2'].value]      #limb darkening coefficients [u1, u2, u3, u4]
    b_params.fp = .01

    lm_params = {}
    lm_params['fp'] = .01

    """Model function"""
    supersample_factor=21
    m = batman.TransitModel(b_params, t, supersample_factor=supersample_factor, exp_time=texp)
    fac = m.fac
    def se_lc(t, lm_params):
        m = batman.TransitModel(b_params, t, transittype='secondary', fac=fac, supersample_factor=supersample_factor, exp_time=texp)
        b_params.fp = lm_params['fp']
        return m.light_curve(b_params) - np.max(m.light_curve(b_params))

    """Run fit and compute depth"""
    se_fitter = Fitter(t, y, lm_params, se_lc, verbose=verbose)
    se_fitter.set_vary(['fp'])

    se_fitter.run_minimizer()
    yfit = se_fitter.get_fit()
    residuals = (y - yfit) * 1e6
    se_lmfit_params = se_fitter.get_params()
    if verbose:
        print(se_fitter.get_result())

    phase = (t-t0)/period

    se_depth, se_depth_err = np.nan, np.nan
    if se_lmfit_params['fp'].value is not None:
        se_depth = se_lmfit_params['fp'].value * 1e6
    if se_lmfit_params['fp'].stderr is not None:
        se_depth_err = se_lmfit_params['fp'].stderr * 1e6

    if plot:
        if ax is None:
            fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(8,6), sharex=True)
        if ax[0] is not None:
            ax[0].set_title(f'SE fit (error is from covariance matrix)')
            ax[0].scatter(phase, y*1e6)
            ax[0].plot(phase, yfit * 1e6, linewidth=2, color='tab:orange')
            ax[0].axhline(-1*se_depth, linewidth=2, color='blue', label=f'δ={np.round(se_depth, 1)}±{np.round(se_depth_err,1)}ppm')
            ax[0].axhspan(-1*se_depth-se_depth_err, -1*se_depth+se_depth_err, alpha=0.3, color='blue', label='1σ')
            ax[0].legend()
        if ax[1] is not None:
            ax[1].set_title('residuals')
            ax[1].scatter(phase, residuals)
            #within_5_sigma = np.abs(residuals) < 5 * np.std(residuals)
            #ax[1].set_title(f'residuals within 5σ ({fstr_ratio(np.sum(within_5_sigma), len(residuals))})')
            #ax[1].scatter(x[within_5_sigma], residuals[within_5_sigma])

        if dirout is not None:
            plt.savefig(f'{dirout}/LS-SE-fit')

    # b_params_copy = copy.deepcopy(b_params)
    # b_params_copy.t_secondary = t0 + period/2
    # b_params_copy.fp = se_lmfit_params['fp']

    # yfit_unfolded = (batman.TransitModel(b_params_copy, t, transittype='secondary', fac=fac,
    # 						supersample_factor=supersample_factor, exp_time=texp).light_curve(b_params_copy) - 1)
    # yfit_unfolded -= np.max(yfit_unfolded)
    # residuals_unfolded = y-yfit_unfolded

    if return_model:
        return se_depth, se_depth_err, se_lmfit_params, yfit
    else:
        return se_depth, se_depth_err

def lmfit_joint_transit_and_SE(t, y, texp, t0, period, verbose=True, plot=False, ax=None, dirout=None):
    """Define model and params"""

    def minus_max(x):
        return x - np.max(x)

    from canislib.fitting import Fitter
    from copy import deepcopy
    import batman
    transit_params = batman.TransitParams()       #planet to store transit parameters
    transit_params.t0 = t0                     #time of inferior conjunction
    transit_params.per = period                       #orbital period
    transit_params.rp = 0.063                       #planet radius (in units of stellar radii)
    transit_params.a = 4.29                        #semi-major axis (in units of stellar radii)
    transit_params.inc = 87.                      #orbital inclination (in degrees)
    transit_params.ecc = 0.                       #eccentricity
    transit_params.w = 0.                        #longitude of periastron (in degrees) doesn't matter if e=0
    transit_params.limb_dark = "quadratic"        #limb darkening model
    transit_params.u = [0.3, 0.2]      #limb darkening coefficients [u1, u2, u3, u4]
    supersample_factor=21


    supersample_factor=21

    se_params = deepcopy(transit_params)
    c_delay=0
    se_params.t_secondary = period/2 + c_delay
    se_params.fp = 100e-6


    """Model function"""
    supersample_factor=21
    m = batman.TransitModel(transit_params, t, supersample_factor=supersample_factor, exp_time=texp)
    m2 = batman.TransitModel(se_params, t, supersample_factor=supersample_factor, exp_time=texp, transittype='secondary')
    fac = m.fac

    def joint_lc(t, lm_params):
        m = batman.TransitModel(transit_params, t, supersample_factor=supersample_factor, exp_time=texp, fac=fac)
        m2 = batman.TransitModel(se_params, t, supersample_factor=supersample_factor, exp_time=texp, transittype='secondary', fac=fac)
        transit_params.a = lm_params['a']
        transit_params.inc = lm_params['inc']
        transit_params.rp = lm_params['rp']
        transit_params.u = [lm_params['u1'], lm_params['u2']]

        se_params.t_secondary = period/2 + lm_params['c_delay']
        se_params.a = lm_params['a']
        se_params.inc = lm_params['inc']
        se_params.rp = lm_params['rp']
        se_params.u = [lm_params['u1'], lm_params['u2']]
        se_params.fp = lm_params['fp']
        return minus_max(m.light_curve(transit_params)) + minus_max(m2.light_curve(se_params))

    """Run fitter"""
    lm_params = {}
    lm_params['rp'] = 0.063
    lm_params['a'] = 4.29
    lm_params['u1'] = 0.3
    lm_params['u2'] = 0.2
    lm_params['inc'] = 87
    lm_params['fp'] = .01
    #lm_params['c_delay'] = 4.29 * 695700/3e5 /86400
    lm_params['c_delay'] = 0

    joint_fitter = Fitter(t, y, lm_params, joint_lc, verbose=verbose)
    joint_fitter.set_vary(['rp', 'a', 'inc'])
    joint_fitter.set_vary(['u1', 'u2'])
    #joint_fitter.set_vary(['c_delay'])
    joint_fitter.set_limits('rp', 0.01, 0.5)
    joint_fitter.set_limits('a', 1, 15)
    joint_fitter.set_limits('u1', -1.0,  1.0)
    joint_fitter.set_limits('u2', -1.0, 1.0)
    joint_fitter.set_limits('inc', 70, 90)
    joint_fitter.set_vary(['fp'])

    joint_fitter.run_minimizer()
    yfit = joint_fitter.get_fit()
    residuals = (y - yfit) * 1e6
    lmfit_params = joint_fitter.get_params()

    if verbose:
        print(joint_fitter.get_result())

    se_depth, se_depth_err = np.nan, np.nan
    if lmfit_params['fp'].value is not None:
        se_depth = lmfit_params['fp'].value * 1e6

    transit_depth = compute_transit_depth(lmfit_params['rp'], lmfit_params['a']*np.cos(lmfit_params['inc']*np.pi/180), (lmfit_params['u1'], lmfit_params['u2']))
    transit_depth *= 1e6

    if plot:
        phase = (t-t0)/period

        if ax is None:
            fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(8,6), sharex=True)
        if ax[0] is not None:
            ax[0].set_title(f'Least squares SE and transit joint fit')
            ax[0].scatter(phase, y*1e6)
            ax[0].plot(phase, yfit * 1e6, linewidth=2, color='tab:orange')
            ax[0].axhline(-1*se_depth, linewidth=2, color='blue', label=f'SE δ={np.round(se_depth, 1)}ppm')
            ax[0].axhline(-1*transit_depth, linewidth=2, color='blue', label=f'Transit δ={np.round(transit_depth, 0)}ppm')
            ax[0].legend()
        if ax[1] is not None:
            ax[1].set_title('residuals')
            ax[1].scatter(phase, residuals)

        if dirout is not None:
            plt.savefig(f'{dirout}/LS-SE-fit')
            plt.close(fig)

    return lmfit_params, se_depth, transit_depth, yfit


def joint_prayer_bead(t, y, texp, t0, period, sample_size=500, plot=False, dirout=None):
    _, _, _, yfit = lmfit_joint_transit_and_SE(t, y, texp, t0, period, verbose=False)
    pb_lmfit_params = []
    pb_transit_depths = []
    pb_se_depths = []

    residuals = y - yfit
    shifts = np.arange(len(t), step=max(int(len(t)/sample_size), 1))
    if plot:
        os.makedirs(f'{dirout}/prayer-bead-joint', exist_ok=True)
    for i, shift in tqdm(enumerate(shifts)):
        y_synthetic = yfit + np.roll(residuals, shift)
        plot_i = plot and i%30==0
        pbi_lmfit_params, se_depth, transit_depth, _ = lmfit_joint_transit_and_SE(t, y_synthetic, texp, t0, period, plot=plot_i, ax=None, verbose=False)
        pb_se_depths.append(se_depth)
        pb_transit_depths.append(transit_depth)
        if plot_i:
            plt.savefig(f'{dirout}/prayer-bead-joint/{i}.png')
            plt.close(plt.gcf())
        pb_lmfit_params.append(pbi_lmfit_params)

    return pb_lmfit_params, pb_se_depths, pb_transit_depths


def prayer_bead(t, y, period, texp, transit_lmfit_params, sample_size=500, plot=False, dirout=None):
    """Run prayer-bead algorithm to find errors as described in Kirkby-Kent et al. 2016
    Plot the first 10 for illustrative purposes"""
    t0=0
    se_depth, se_depth_err, se_lmfit_params, yfit = lmfit_secondary_eclipse(t, y, t0=t0, period=period, texp=texp, transit_lmfit_params=transit_lmfit_params,
                                                                                    return_model=True, plot=False, verbose=False)
    residuals = y-yfit

    shifts = np.arange(len(t), step=max(int(len(t)/sample_size), 1)) # include 0 as a shift

    if plot:
        os.makedirs(f'{dirout}/prayer-bead', exist_ok=True)

    se_depths = []
    se_depth_covar_errors = []
    for i, shift in tqdm(enumerate(shifts)):
        #print(i, end=' ')
        y_synthetic = yfit + np.roll(residuals, shift)
        plot_i =  plot and i%30==0
        synthetic_se_depth, synthetic_se_depth_err = lmfit_secondary_eclipse(t, y_synthetic, t0=t0, period=period, texp=texp, transit_lmfit_params=transit_lmfit_params,
                                                                                    return_model=False, plot=plot_i, verbose=False)
        if plot_i:
            plt.savefig(f'{dirout}/prayer-bead/{i}.png')
            plt.close(plt.gcf())
        se_depths.append(synthetic_se_depth)
        se_depth_covar_errors.append(synthetic_se_depth_err)

    se_depth_err = np.nanstd(se_depths)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        plt.suptitle(f'Prayer-bead (sample size={sample_size}, data size={len(t)}) on secondary eclipse depth')
        ax[0].set_title('SE depths')
        ax[0].hist(se_depths, bins=20)
        ax[1].set_title('Covariance-based SE depth errors\n(incorrect, including just for the sake of comparison)')
        ax[1].hist(se_depth_covar_errors, bins=20)
        ax[1].axvline(se_depth_err, label=f'Prayer-bead error: {np.round(se_depth_err,1)}')
        ax[1].legend()
        if dirout is not None:
            plt.savefig(f'{dirout}/5.1-prayer-bead_distribution.png')
            plt.close(plt.gcf())

    return se_depths, se_depth_err
