# -*- coding: utf-8 -*-
# %run scale_cfsr2coads.py

'''
Scale selected variables from surface forcing file created with
pycfsr2frc.py with the corresponding variables from a climatological
forcing file created by Roms_tools make_forcing.m, which uses COADS.

This correction is applied after a frc file has been created with 
pycfsr2frc.py.  Also, a standard COADS frc file created using Roms_tools
make_frc.m for the grid in use  must be available.

The motivation for this scaling was the observation of SST>35deg off the
coast of Tunisia in a CFSR-forced 5km Mediterranean solution.

It was found that COADS scalar winds (file w3.cdf in Roms_tools) were
consistently larger than the CFSR scalar winds computed from the CFSR
vector wind components.
Also swrad, need to check...
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np


if __name__ == '__main__':
    
    '''
    scale_cfsr2coads
    
    Apply a scaling factor to selected CFSR forcing variables
    '''
    
    # List of variables to be scaled
    #vars4scl = ['shflux', 'swrad', 'dQdSST', 'SST'] # used for first Med5 run
    vars4scl = ['dQdSST']
    
    #clim_frc = '/shared/emason/marula/emason/runs2012/MedSea5/frc_MedSea5.nc'
    #clim_frc = '/marula/emason/runs2012/MedSea5_OMP/frc_MedSea5.nc'
    #clim_frc = '/marula/emason/runs2012/MedSea5_OMP/frc_MedSea5.nc'
    #clim_frc = '/marula/emason/runs2012/na_7pt5km/roms_frc_NA2009_7pt5km_RIV.nc'
    clim_frc = '/marula/emason/runs2009/cb_2009_3km_42/cb_2009_3km_frc.nc'
    #clim_frc = '/marula/emason/runs2013/AlbSea_1pt5/frc_AlbSea_1.5km.nc'
    
    #cfsr_frc = '/home/emason/runs2012_tmp/MedSea5_R2.5/frc_intann_MedSea5.nc'
    #cfsr_frc = '/home/emason/runs2012_tmp/MedSea5_R2.5/frc_intann_MedSea5.nc'
    #cfsr_frc = '/marula/emason/runs2013/na_7pt5km_intann_5day/frc_CFSR_NA_7pt5km.nc'
    #cfsr_frc = '/marula/emason/runs2013/na_7pt5km_intann_5day/frc_CFSR_NA_7pt5km_UPDATE.nc'
    cfsr_frc = '/marula/emason/runs2013/cb_3km_2013_intann/frc_2013_cb3km_CFSR_UPDATE.nc'
    #cfsr_frc = '/marula/emason/runs2012/MedSea5_intann_monthly/frc_MedSea5_1985010100_64bit.nc'
    #cfsr_frc = '/marula/emason/runs2013/AlbSea_1pt25/frc_AlbSea_1pt25_CFSR_20030101.nc'
    #cfsr_frc = '/marula/emason/runs2013/cart500/frc_cart500.nc'
    
    downscaled = True
    if downscaled:
        #cfsr_par_frc = '/marula/emason/runs2012/MedSea5_intann_monthly/frc_MedSea5_test.nc'
        cfsr_par_frc = '/marula/emason/runs2013/na_7pt5km_intann_5day/frc_CFSR_NA_7pt5km_UPDATE.nc'
    
    # True to remove sub-annual variability (not really recommended)
    annual = False #True
    # True to remove sub-seasonal variability
    seasonal = True #False
    
    assert not np.alltrue([annual, seasonal]), '"annual" and "seasonal" cannot both be True'
    

    
    
    nccfsr = netcdf.Dataset(cfsr_frc, 'a')
    if downscaled:
        ncparfrc = netcdf.Dataset(cfsr_par_frc)
    else:
        ncclim = netcdf.Dataset(clim_frc)
    
    cfsr_months = np.array(nccfsr.variables['month'][:], dtype='int')
    
    mask = nccfsr.variables['SST'][0]
    mask[mask != 0] = 1.
    assert mask.sum(dtype='int') != mask.size, 'Check if SST == 0 over land?'
    n_p, m_p = mask.shape
    
    
    # First, loop over the variables
    for var in vars4scl:
        
        print '---Doing variable', var
        
        varout = np.zeros((12., n_p, m_p))
        varcount = np.zeros(12.)
        scaling = np.zeros(12.)
        
        
        if downscaled:
            if 'shflux' in var:
                scaling = ncparfrc.scl2COADS_shflux
            elif 'swrad' in var:
                scaling = ncparfrc.scl2COADS_swrad
            elif 'dQdSST' in var:
                scaling = ncparfrc.scl2COADS_dQdSST
            elif 'SST' in var:
                scaling = ncparfrc.scl2COADS_SST
            
        else:
            # Second, loop over months and get longterm monthly mean
            for climmon in np.arange(12):
            
                # Loop over cfsr records
                for cfsri, cfsrmon in enumerate(cfsr_months):
            
                    if (climmon + 1) == cfsrmon:
        
                        varout[climmon] += nccfsr.variables[var][cfsri]
                        varcount[climmon] += 1.
            
                varout[climmon] /= varcount[climmon]
        
            print '------got longterm monthly means'
        
        
            # Third, loop over months and get scaling
            for climmon in np.arange(12):
            
                coads_var = ncclim.variables[var][climmon]
                cfsr_var = varout[climmon]
            
                coads_var = np.ma.masked_where(mask==0, coads_var)
                cfsr_var = np.ma.masked_where(mask==0, cfsr_var)
            
                scaling[climmon] = cfsr_var.mean() - coads_var.mean()
        
            print '------got scalings for', var
        
        
            if annual:
                scaling.flat[:] = scaling.mean()
        
            elif seasonal:
                scaling.flat[:3] = scaling[:3].mean()
                scaling.flat[3:6] = scaling[3:6].mean()
                scaling.flat[6:9] = scaling[6:9].mean()
                scaling.flat[9:] = scaling[9:].mean()
        
        
        # Fifth, loop over cfsr records and apply scaling
        for cfsri, cfsrmon in enumerate(cfsr_months):
            
            to_be_scaled = nccfsr.variables[var][cfsri]
            try:
                to_be_scaled -= scaling[cfsrmon-1]
                to_be_scaled *= mask
                nccfsr.variables[var][cfsri] = to_be_scaled
            except Exception:
                pass
        
        # Add global variables with the scaling values
        # These will be needed if preparing frc files for
        # downscaled runs.
        if 'shflux' in var:
            nccfsr.scl2COADS_shflux = scaling
        elif 'swrad' in var:
            nccfsr.scl2COADS_swrad = scaling
        elif 'dQdSST' in var:
            nccfsr.scl2COADS_dQdSST = scaling
        elif 'SST' in var:
            nccfsr.scl2COADS_SST = scaling
            
        print '------scalings applied to', var
            

    nccfsr.scaled_variables = 'Variables %s scaled to COADS' %str(vars4scl)
    
    nccfsr.close()
    
    if downscaled:
        ncparfrc.close()
    else:
        ncclim.close()
        
    
    
    
    
    
