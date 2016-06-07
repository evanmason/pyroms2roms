# -*- coding: utf-8 -*-
# %run make_cfsr_clim.py

'''
===========================================================================
This file is part of py-roms2roms

    py-roms2roms is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    py-roms2roms is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with py-roms2roms.  If not, see <http://www.gnu.org/licenses/>.

Version 1.0.1

Copyright (c) 2014 by Evan Mason, IMEDEA
Email: emason@imedea.uib-csic.es
===========================================================================

Compute monthly mean frc or blk file from interannual frc/blk files

Options 'year_str' and 'year_end' allow individual years or
a range of years to specified, with output files named
accordingly, ie:

    monthly_1998-1998.nc or monthly_1998-2008.nc

===========================================================================
'''

import netCDF4 as netcdf
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
from scipy import io
from py_roms2roms import RomsGrid


def savethevar(create_file, savefile, thedims, var, thevar, grd):
    '''
    
    '''        
    if create_file: # Create new nc files
        xn = grd.maskr().shape[1]
        yn = grd.maskr().shape[0]
        nc = netcdf.Dataset(savefile, 'w', format='NETCDF4')
        nc.createDimension('time', 12)
        nc.createDimension('xi_rho', xn)
        nc.createDimension('xi_u', xn-1)
        nc.createDimension('eta_rho', yn)
        nc.createDimension('eta_v', yn-1)
        nc.close()

    nc = netcdf.Dataset(savefile, 'a')
    nc.createVariable(var, 'f8', dimensions=thedims)
    for t in np.arange(12):
        nc.variables[var][t] = thevar[t]
    nc.sync()
    nc.close()
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    #--Begin user defined options-----------------------------------
    

    sigma_params = dict(theta_s=None, theta_b=None, hc=None, N=None)

    
    #directory = '/marula/emason/runs2014/NA75_IA/'
    #frc_file = directory + 'blk_NA2009_2008-2010_6hr.nc'
    #grd = RomsGrid(directory + 'roms_grd_NA2009_7pt5km.nc', sigma_params, 'ROMS')
    
    directory = '/marula/emason/runs2014/NWMED2_unstable/'
    frc_file = directory + 'blk_nwmed_2km_2006-2006_6hr.nc'
    grd = RomsGrid(directory + 'grd_nwmed_2km.nc', sigma_params, 'ROMS')
    
    year_str = 2006
    year_end = 2006
    
    if 'blk' in frc_file:
        savefile = directory + 'cfsr_blk_monthly_%s-%s.nc' %(str(year_str), str(year_end))
    else:
        savefile = directory + 'cfsr_frc_monthly_%s-%s.nc' %(str(year_str), str(year_end))
    
    


    #--End user defined options-----------------------------------

    Mp, Np = grd.maskr().shape

    year_str = dt.datetime.datetime(year_str, 1, 1)
    year_end = dt.datetime.datetime(year_end, 12, 31)
    
    
    
    start_dic = io.loadmat(directory + 'start_date.mat')
    start_date = start_dic['start_date']

    create_file = True


    
    nc = netcdf.Dataset(frc_file)
    
    bulk_time = nc.variables['bulk_time'][:]
    
    #var = 'init'
    vars_rho = np.array(['tair', 'rhum', 'prate', 'wspd', 'radlw', 'radlw_in', 'radsw'])
    vars_u = np.array(['sustr', 'uwnd'])
    vars_v = np.array(['svstr', 'vwnd'])
    
    for var in nc.variables.keys():
        
        if np.any(var in vars_rho):
            thevar = np.zeros((12, Mp, Np))
            thedims = ('time', 'eta_rho', 'xi_rho')
        elif np.any(var in vars_u):
            thevar = np.zeros((12, Mp, Np-1))
            thedims = ('time', 'eta_rho', 'xi_u')
        elif np.any(var in vars_v):
            thevar = np.zeros((12, Mp-1, Np))
            thedims = ('time', 'eta_v', 'xi_rho')
        else:
            thevar = False  

        count = np.zeros((12.))

        if thevar is not False:
        
            print '--- doing %s' %var.upper()

            for ind, ot in enumerate(bulk_time):
        
                thetime = np.squeeze(ot + start_date)
        
                thedate = dt.num2date(thetime)
        
                if np.logical_and(thetime >= dt.date2num(year_str),
                                  thetime <= dt.date2num(year_end)):
        
                    print '------ doing %s, date' %var, thedate.year, thedate.month, thedate.day
            
                    thevar[thedate.month-1] += nc.variables[var][ind]
            
                    count[thedate.month-1] += 1.

            thevar = np.divide(thevar.T, count).T

            savethevar(create_file, savefile, thedims, var, thevar, grd)
            create_file = False
            
    nc.close()
    
    


    
