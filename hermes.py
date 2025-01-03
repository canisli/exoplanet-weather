#!/usr/bin/env python

import requests
import os, sys, argparse
import numpy as np
import lightkurve as lk
from astropy.table import Table
from inputimeout import inputimeout, TimeoutOccurred
from canislib.timeseriestools import lightkurve_to_dict
from canislib.data import psave, pload
from canislib.misc import print_bold, print_hline
from canislib.exoplanets import tildes_to_spaces, spaces_to_tildes, root

def main():
    parser = argparse.ArgumentParser(
                    prog='hermes',
                    description='use lightkurve to download light curves for Kepler and TESS')
    parser.add_argument('systems', metavar='systems', type=str, nargs='*')
    parser.add_argument('--pipeline', default='all')
    parser.add_argument('--mission', default=None)
    parser.add_argument('--bitmask', default='all')
    parser.add_argument('--input_file', default=None)
    parser.add_argument('--alias_to_search', default=None)
    parser.add_argument('--skip_confirmation', action='store_true', default=False)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    args = parser.parse_args()


    overwrite = args.overwrite
    bitmask = args.bitmask
    alias_to_search = args.alias_to_search
    mission = args.mission

    if mission is None:
        mission = input('> Specify mission: ')

    if args.pipeline == 'all':
        if mission == 'Kepler':
            pipelines = ['PDCSAP', 'SAP']
        elif mission == 'TESS':
            pipelines = ['PDCSAP', 'SAP'] # Manually run QLP.
    else:
        pipelines = [args.pipeline]

    if args.bitmask == 'all':
        bitmasks = ['default', 'none', 'hardest']
    else:
        bitmasks = [bitmask]

    if mission in ['Kepler']:
        exptime = 1800 # long cadence data
        author = 'Kepler'
    elif mission in ['TESS']: #SPOC
        if 'QLP' in pipelines and len(pipelines) > 1:
            raise ValueError('Run queries involving QLP by themselves')
        elif 'QLP' in pipelines:
            exptime = [1800, 600, 200]
            author = 'QLP'
        else:
            exptime = 120
            author = 'SPOC'
    
    if args.input_file is not None:
        from canislib.data import cat
        systems = cat(args.input_file).split()
    else:
        if len(args.systems) > 0:
            systems = args.systems
        else:
            raise ValueError('Specify at least one system to download')
    
    print_bold('\nConfiguration')
    print(f"""
    {mission = }, {author = } 
    {pipelines = } 
    {bitmasks = }
    {exptime = }
    {systems = }
    {overwrite = }
    """)
    if not args.skip_confirmation:
        try:
            inputimeout(prompt=f'Press enter to continue or wait {(timeout:=10)} seconds\n', timeout=timeout)
        except TimeoutOccurred:
            pass

    # -------------------------------------------------------------
        
    for i, system in enumerate(systems):
        print_hline(50)
        print_bold(f'[{tildes_to_spaces(system)}] [{i+1}/{len(systems)}]')
        system_name = system

        def already_have_downloaded():
            exists = []
            for pipeline in pipelines:
                for bitmask in bitmasks:
                    exists.append(os.path.exists(f'{root}/data/lightcurves{"/TESS" if mission == "TESS" else ""}/{system_name}.{pipeline}.{bitmask}.p'))
            return np.logical_and.reduce(exists)

        if not overwrite and already_have_downloaded():
            print('Already have all the light curves')
            continue
        print(f'Searching for light curve')
        
        # manual overrides
        if system in ['Kepler-76b', 'Kepler-76~b'] and mission == 'Kepler':
            system = 'kplr004570949'
        if system in ['Kepler-1989b', 'Kepler-1989~b'] and mission == 'Kepler':
            system = 'kplr008162789'

        def search_lk():
            nonlocal alias_to_search
            if alias_to_search is None:
                pl_name = system
            else:
                pl_name = alias_to_search
            if author == 'QLP': # QLP lightcurves have varying exptimes
                search_result = lk.search_lightcurve(tildes_to_spaces(pl_name), mission=mission, author=author)
            else:
                search_result = lk.search_lightcurve(tildes_to_spaces(pl_name), mission=mission, exptime=exptime, author=author)

            if mission == 'Kepler':
                assert len(search_result) <= 18

            return search_result
        
        try:
            search_result = search_lk()
        except requests.exceptions.ReadTimeout:
            print('Searching with lightkurve timed out. Trying again.')
            search_result = search_lk() # Try again

        print(f'Downloading {system} light curve')

        for pipeline in pipelines:
            for bitmask in bitmasks:
                fout = f'{root}/data/lightcurves{"/TESS" if mission == "TESS" else ""}/{system_name}.{pipeline}.{bitmask}.p'
                if pipeline in ['QLP', 'SAP']:
                    flux_column = 'sap_flux'
                elif pipeline in ['PDCSAP']:
                    flux_column = 'pdcsap_flux'
                else:
                    raise ValueError
                
                lcs = search_result.download_all(quality_bitmask=bitmask, flux_column=flux_column)
                print(f'Downloaded {system} light curve')   

                if mission == 'Kepler':
                    data = {'quarters': {}}
                elif mission == 'TESS':
                    data = {'sectors': {}}
                data['mission'] = mission
                data['exptime'] = exptime
                if lcs is None:
                    print_bold('no light curves found', color='red')
                    continue
                for lc in lcs:
                    if mission == 'Kepler':
                        data['quarters'][lc.quarter] = lightkurve_to_dict(lc)
                    elif mission == 'TESS':
                        data['sectors'][lc.sector] = lightkurve_to_dict(lc)
                if mission == 'Kepler':
                    quarters = data['quarters'].keys()
                elif mission =='TESS':
                    sectors = data['sectors'].keys()
                if mission == 'Kepler':
                    t_total = np.concatenate([data['quarters'][q]['time'] for q in quarters])
                    y_total = np.concatenate([data['quarters'][q]['flux'] for q in quarters])
                    yerr_total = np.concatenate([data['quarters'][q]['flux_err'] for q in quarters])
                    quality_total = np.concatenate([data['quarters'][q]['quality'] for q in quarters])
                elif mission == 'TESS':
                    t_total = np.concatenate([data['sectors'][q]['time'] for q in sectors])
                    y_total = np.concatenate([data['sectors'][q]['flux'] for q in sectors])
                    yerr_total = np.concatenate([data['sectors'][q]['flux_err'] for q in sectors])
                    quality_total = np.concatenate([data['sectors'][q]['quality'] for q in sectors])

                data['time_format'] = 'bkjd' if mission == 'Kepler' else 'btjd' if mission == 'TESS' else 'invalid'
                data['time'] = t_total
                data['flux'] = y_total
                data['flux_err'] = yerr_total
                data['quality'] = quality_total
                data['pipeline'] = pipeline
                data['bitmask'] = bitmask
                psave(data, fout)
        
    print_hline()
    print('Summary')
    print(f'mission: {mission}')
    print(f'exptime: {exptime}')
    print(f'systems: {systems}')

if __name__ == '__main__':
    main()