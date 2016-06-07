# -*- coding: utf-8 -*-
# %run pycfsr_nhf_dqdsst.py

'''
Script to generate nc file containing net heat flux
and dQdSST to be used by pycfsr2frc.py
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np
import scipy.interpolate as si
#import scipy.ndimage as nd
#import scipy.spatial as sp
#import matplotlib.nxutils as nx
#import time
#import scipy.interpolate.interpnd as interpnd
#import collections
#from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime
#import calendar as ca

#from pyroms2roms import horizInterp
#from pyroms2roms import ROMS, debug0, debug1
#from pyroms2roms import RomsGrid, RomsData
from pycfsr2frc import CfsrGrid, CfsrData


def create_nc(cfsrgrd):
    '''
    Create empty ncfile to contain CFSR net heat flux
    and dQdSST
    '''
    cfsrgrd.cfsrfile = cfsrgrd.romsfile.rpartition('/')[0] + '/CFSR_NHF_dQdSST.nc'
    
    
    # Global attributes
    nc = netcdf.Dataset(cfsrgrd.cfsrfile, 'w', format='NETCDF3_CLASSIC')
    nc.created  = datetime.now().isoformat()
    nc.type     = 'CFRS data file produced by pycfsr_nhf_dqdsst.py'
    #nc.start_date = sd
    #nc.end_date = ed
        
    # Dimensions
    nc.createDimension('lon', cfsrgrd.lon().shape[1])
    nc.createDimension('lat', cfsrgrd.lat().shape[0])
    nc.createDimension('time', None)

    cfsr_vars = OrderedDict()

    cfsr_vars['time'] =   ['time',
                           'time',
                           'net heat flux time',
                           'valid_date_time']
                         
    cfsr_vars['longrad'] = ['space',
                           'time',
                           'net heat flux',
                           'W m-2']

    for key, value in zip(cfsr_vars.keys(), cfsr_vars.values()):

        if 'time' in value[0]:
            dims = (value[1])

        elif 'space' in value[0]:
            dims = (value[1], 'lat', 'lon')
                
        else: error

        nc.createVariable(key, 'f8', dims)
        nc.variables[key].long_name = value[2]
        nc.variables[key].units     = value[3]
            
    nc.close()


def dQdSST(sst, sat, rho_atm, U, qsea):
    '''
    Compute the kinematic surface net heat flux sensitivity to the
    the sea surface temperature: dQdSST.
    Q_model ~ Q + dQdSST * (T_model - SST)
    dQdSST = - 4 * eps * stef * T^3  - rho_atm * Cp * CH * U
             - rho_atm * CE * L * U * 2353 * ln (10 * q_s / T^2)
    
    Input parameters:
    sst     : sea surface temperature (Celsius)
    sat     : sea surface atmospheric temperature (Celsius)
    rho_atm : atmospheric density (kilogram meter-3) 
    U       : wind speed (meter s-1)
    qsea    : sea level specific humidity
 
    Output:
    dqdsst  : kinematic surface net heat flux sensitivity to the
              the sea surface temperature (Watts meter-2 Celsius-1)
              
    From Roms_tools of Penven etal
    '''
    #  Specific heat of atmosphere.
    Cp = 1004.8
    # Sensible heat transfer coefficient (stable condition)
    Ch = 0.66e-3
    # Latent heat transfer coefficient (stable condition)
    Ce = 1.15e-3
    # Emissivity coefficient
    eps = 0.98
    # Stefan constant
    stef = 5.6697e-8
    # SST (Kelvin)
    SST = sst + 273.15
    #  Latent heat of vaporisation (J.kg-1)
    L = 2.5008e6 - 2.3e3 * sat
    # Infrared contribution
    q1 = -4. * stef * (SST**3)
    # Sensible heat contribution
    q2 = -rho_atm * Cp * Ch * U
    # Latent heat contribution
    dqsdt = 2353. * np.log(10.) * qsea / (SST**2)
    q3 = -rho_atm * Ce * L * U * dqsdt
    #  dQdSST
    dqdsst = q1 + q2 + q3
    return dqdsst
    
    
    


if __name__ == '__main__':
    
    '''
    pycfsr_nhf_dqdsst.py
    
    Prepare intermediate nc file of Net Heat Flux and dQdSST from CFSR data
        
      http://rda.ucar.edu/pub/cfsr.html
    
    CFSR surface data for ROMS forcing are global but subgrids can be
    selected. User must supply a list of the files available, pycfsr_nhf_dqdsst
    will loop through the list, reading and using each variable.

    
    
    Evan Mason, IMEDEA, 2012
    '''
    
    
    
    
    cfsr_dir = '/nas02/emason/NCEP-CFSR/'
    
    cfsr_files = OrderedDict()
    cfsr_files['DLWRF'] = 'flxf06.gdas.DLWRF.SFC.grb2.nc'
    cfsr_files['ULWRF'] = 'flxf06.gdas.ULWRF.SFC.grb2.nc'
    cfsr_files['SST'] = 'pgbh01.gdas.TMP.SFC.grb2.nc'
    cfsr_files['UWND'] = 'flxf01.gdas.UWND.10m.grb2.nc'
    cfsr_files['VWND'] = 'flxf01.gdas.VWND.10m.grb2.nc'
    
    
    cfsr_DLWRF = CfsrData(cfsr_dir + cfsr_files['DLWRF'])
    cfsr_ULWRF = CfsrData(cfsr_dir + cfsr_files['ULWRF'])
    cfsr_SST = CfsrData(cfsr_dir + cfsr_files['SST'])
    cfsr_UWND = CfsrData(cfsr_dir + cfsr_files['UWND'])
    cfsr_VWND = CfsrData(cfsr_dir + cfsr_files['VWND'])
    
    
    # boolean for datasets that need interpolation
    needs_interp = np.zeros(len(cfsr_files.keys()))
    
    
    start = True
    
    # loop over the CFSR files to check grid sizes 
    # are all the same
    ni = 0
    for key, value in zip(cfsr_files.keys(), cfsr_files.values()):
      
        print '---checking dimensions of variable %s' %key
        if start:
            cfsrgrd = CfsrGrid(cfsr_dir + value)
            cfsrdata = CfsrData(cfsr_dir + value)
            points = np.array([cfsrgrd.lon().ravel(),
                               cfsrgrd.lat().ravel()]).T
            cfsrtime = cfsrdata.time()
            start = False
        else:
            new_cfsrgrd = CfsrGrid(cfsr_dir + value)
            # if any grids not equal we need to interpolate...
            if np.any([cfsrgrd.lon().mean() != new_cfsrgrd.lon().mean(),
                       cfsrgrd.lon().size   != new_cfsrgrd.lon().size,
                       cfsrgrd.lat().mean() != new_cfsrgrd.lat().mean(),
                       cfsrgrd.lat().size   != new_cfsrgrd.lat().size]):
                print '------interpolation will be required for %s' %key
                # flag need for an interpolation of this product
                needs_interp[ni] = 1
                new_points = np.array([new_cfsrgrd.lon().ravel(),
                                       new_cfsrgrd.lat().ravel()]).T
                cfsr_files['%s_points' %key] = new_points
        ni += 1
    
    
    
    # create nc file
    create_nc(cfsrgrd)
    
    
    # loop over cfsrtime
    for cfsri, cfsrt in enumerate(cfsrtime):
        
        
        # check for timing mismatches greater than 6 hours
        # (6 hours is small for monthly means, but this may need to be
        #  changed for other timescales)
        assert np.logical_and(
                   cfsrt >= cfsr_DLWRF.time()[cfsri] - 6,
                   cfsrt <= cfsr_DLWRF.time()[cfsri] + 6), 'DLWRF time mismatch'
        assert np.logical_and(
                   cfsrt >= cfsr_ULWRF.time()[cfsri] - 6,
                   cfsrt <= cfsr_ULWRF.time()[cfsri] + 6), 'ULWRF time mismatch'
        assert np.logical_and(
                   cfsrt >= cfsr_SST.time()[cfsri] - 6,
                   cfsrt <= cfsr_SST.time()[cfsri] + 6), 'SST time mismatch'
        assert np.logical_and(
                   cfsrt >= cfsr_UWND.time()[cfsri] - 6,
                   cfsrt <= cfsr_UWND.time()[cfsri] + 6), 'UWND time mismatch'
        assert np.logical_and(
                   cfsrt >= cfsr_VWND.time()[cfsri] - 6,
                   cfsrt <= cfsr_VWND.time()[cfsri] + 6), 'VWND time mismatch'

        # downward longwave
        dlwrf = cfsr_DLWRF.frc_DLWRF(cfsri)
        # upward longwave
        ulwrf = cfsr_ULWRF.frc_ULWRF(cfsri)
        # net longwave
        nlw = ulwrf - dlwrf
        
        # SST
        sst = cfsr_SST.frc_sst(cfsri)
        if needs_interp[2]: # this would be better with a has_key...
            sst = si.griddata(cfsr_files['SST_points'], sst.ravel(),
                             (cfsrgrd.lon(), cfsrgrd.lat()), method='linear')
        
        # Wind speed
        uwnd = cfsr_UWND.frc_uspd(cfsri)
        vwnd = cfsr_VWND.frc_vspd(cfsri)
        U = np.hypot(uwnd, vwnd)
        
        #dqdsst = dQdSST(sst, sat, rho_atm, U, qsea)
        
        nc = netcdf.Dataset(cfsrgrd.cfsrfile, 'a')
        nc.variables['longrad'][cfsri] = nlw
        #nc.variables['dQdSST'][cfsri] = dqdsst

        nc.close()
        
        aaaaaaaa
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    