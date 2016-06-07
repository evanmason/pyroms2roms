# %run add_bry_flux_correction.py

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

Create a forcing file based on six hourly CFSR data

===========================================================================

'''


import netCDF4 as netcdf
import matplotlib.pyplot as plt
import numpy as np
import time as time

from py_roms2roms import RomsGrid





if __name__ == '__main__':
  
    # ROMS information
    roms_dir = '/marula/emason/runs2014/nwmed5km/'
    
    
    #bry_file = 'bry_nwmed5km_CORR_0pt2.nc'
    #bry_file = 'bry_nwmed5km_VARCORR_0pt2.nc'
    bry_file = 'bry_nwmed5km_weekly_clim_VARCORR_0pt2.nc'
    
    
    #roms_grd = 'grd_nwmed5km.nc'
    roms_grd = 'grd_nwmed5km_NARROW_STRAIT.nc'
    
    variable_correction = True

    flux_correction = 0.2 # Sv
    
    open_boundary = 'east'


    if 'roms_grd_NA2014_7pt5km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
        obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed
    
    elif 'grd_nwmed_2km.nc' in roms_grd:
        sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
        obc_dict = dict(south=1, east=1, north=1, west=1)
    
    elif 'grd_na6km.nc' in roms_grd:
        sigma_params = dict(theta_s=6., theta_b=0, hc=120, N=32)
        obc_dict = dict(south=1, east=1, north=1, west=1)
    
    elif 'grd_nwmed5km.nc' in roms_grd:
        sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
        obc_dict = dict(south=0, east=1, north=1, west=1)
    
    elif 'grd_nwmed5km_NARROW_STRAIT.nc' in roms_grd:
        sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
        obc_dict = dict(south=0, east=1, north=1, west=1)
    
    else:
        print 'No sigma parameters defined for grid: %s' %roms_grd
        raise Exception
    
    
    
    
    # Set up a RomsGrid object
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
    romsgrd.set_bry_dx()
    romsgrd.set_bry_maskr()
    romsgrd.set_bry_areas()
    
    if variable_correction:
        
        if 'west' in open_boundary:
            bry_surface_area  = romsgrd.area_west * romsgrd.maskr_west
        elif 'east' in open_boundary:
            bry_surface_area  = romsgrd.area_east * romsgrd.maskr_east
        elif 'south' in open_boundary:
            bry_surface_area  = romsgrd.area_south * romsgrd.maskr_south
        elif 'north' in open_boundary:
            bry_surface_area  = romsgrd.area_north * romsgrd.maskr_north
        
    else:
    
        if 'west' in open_boundary:
            bry_surface_area  = romsgrd.area_west.sum(axis=0) * romsgrd.maskr_west
        elif 'east' in open_boundary:
            bry_surface_area  = romsgrd.area_east.sum(axis=0) * romsgrd.maskr_east
        elif 'south' in open_boundary:
            bry_surface_area  = romsgrd.area_south.sum(axis=0) * romsgrd.maskr_south
        elif 'north' in open_boundary:
            bry_surface_area  = romsgrd.area_north.sum(axis=0) * romsgrd.maskr_north
    
        # Velocity correction
        correction = flux_correction * 1e6 / np.sum(bry_surface_area)
    
    
    with netcdf.Dataset(roms_dir + bry_file, 'a') as nc:
        
        bry_time = nc.variables['bry_time'][:]
        tsize = bry_time.size
        for tind in np.arange(tsize):
            print tind, tsize
            
            if variable_correction:
                
                u_mask = nc.variables['u_%s' %open_boundary][tind]
                u_mask = np.ma.masked_greater(u_mask, 0).mask
                
                correction = flux_correction * 1e6 / np.sum(bry_surface_area * u_mask)
                
                u = nc.variables['u_%s' %open_boundary][tind]
                print correction
                u[u_mask] += correction
                nc.variables['u_%s' %open_boundary][tind] = u
                ubar = np.sum(u *  bry_surface_area / romsgrd.dx_east, axis=0)
                ubar /= romsgrd.h()[:,-1]
                nc.variables['ubar_%s' %open_boundary][tind] = ubar
            
            else:
                
                nc.variables['u_%s' %open_boundary][tind] += correction
                nc.variables['ubar_%s' %open_boundary][tind] += correction
        
        nc.variables['u_%s' %open_boundary].variable_flux_correction = correction
        nc.variables['ubar_%s' %open_boundary].variable_flux_correction = correction
    
    print 'done'
    
    
    
    
    
    
    
    
    
  