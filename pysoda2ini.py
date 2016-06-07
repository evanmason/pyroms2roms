# -*- coding: utf-8 -*-
# %run pysoda2ini.py

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

Create a ROMS initial file based on SODA data

===========================================================================
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
import time
import scipy.interpolate.interpnd as interpnd
#import collections
from mpl_toolkits.basemap import Basemap
#from collections import OrderedDict

from py_roms2roms import vertInterp, horizInterp
from py_roms2roms import ROMS, debug0, debug1
from pysoda2roms import SodaData, SodaGrid, RomsGrid
from pyroms2ini import RomsData




if __name__ == '__main__':
    
    '''
    pysoda2ini (Python version of r2r_ini.m written in Matlab).


    
    Evan Mason, IMEDEA, 2012
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    # SODA information
    version = '2.1.6' # Note, uses meters for u, v, ssh
    #version = '1.4.3' # Note, uses centimeters for u, v, ssh
    #version = '2.2.8'
    
    #soda_dir     = '/Users/emason/data/SODA_2.1.6/'
    #soda_dir     = '/marula/emason/data/SODA/2.1.6/MONTHLY/'
    soda_dir     = '/marula/emason/data/SODA/2.1.6/5DAY/'
    #soda_dir     = '/shared/emason/SODA/2.2.8/MONTHLY/'
    #soda_dir     = '/Users/emason/data/SODA_2.2.8/'
    

    #soda_file    = 'SODA_2.1.6_199001.cdf'
    #soda_file    = 'SODA_2.1.6_198501.cdf'
    soda_file    = 'SODA_2.1.6_19850104-19850109.cdf'
    #soda_file    = 'SODA_2.2.8_198501.cdf'
    #soda_file    = 'SODA_2.2.8_198412.cdf'
    
    #roms_dir     = '/marula/emason/runs2012/MedSea5/'
    #roms_dir     = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir     = '/marula/emason/runs2014/MedCan5/'
    #roms_dir = '/shared/emason/runs2009/na_2009_7pt5km/'
    #roms_dir = '/Users/emason/runs2009/na_2009_7pt5/'
    roms_dir     = '/marula/emason/runs2014/NA75_IA/'
    
    #roms_grd     = 'grd_MedCan5.nc'
    #roms_grd     = 'grd_MedSea5.nc'
    roms_grd     = 'roms_grd_NA2009_7pt5km.nc'
    #roms_dir     = '/Users/emason/runs/runs2014/MedCan5/'
    #roms_grd     = 'grd_MedCan5.nc'
    
    if 'roms_grd_NA2009_7pt5km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
    elif 'grd_nwmed_2km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=250, N=40)
    else:
        print 'No sigma params defined for grid: %s' %roms_grd
        raise Exception
    
    
    # Initial file
    ini_time = 0.  # days
    tstart = 0. 
    tend = 0. 
    
    # Set ini filename
    #ini_filename = 'ini_MedSea5_%s.nc' %(soda_file.split('_')[-1].split('.')[0])
    #ini_filename = 'ini_SODA5day_NA7pt5km_%s.nc' %(soda_file.split('_')[-1].split('-')[0])
    #ini_filename = 'ini_MedCan5_%s.198501.nc' %(soda_file.split('_')[-1].split('-')[0])
    ini_filename = 'ini_NA2009_%s.198501.nc' %(soda_file.split('_')[-1].split('-')[0])
    
    
    balldist = 250000. # distance (m) for kde_ball (should be 2dx at least?)



    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    
    soda_file = soda_dir + soda_file
    
    fillval = 9999.99
    
    
    if '2.2.8' in version:
        ssh_str = 'SSH'
        temp_str = 'TEMP'
        salt_str = 'SALT'
        u_str = 'U'
        v_str = 'V'
    else:
        ssh_str = 'ssh'
        temp_str = 'temp'
        salt_str = 'salt'
        u_str = 'u'
        v_str = 'v'

    # Set up a RomsData object for the initial file
    romsini = RomsData(roms_dir + ini_filename, model_type='ROMS')
    
       
    # Initialise SodaGrid and RomsGrid objects for both parent and child grids
    sodagrd = SodaGrid(soda_file, model_type='SODA')
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, model_type='ROMS')

    
    
    # Activate flag for zero crossing trickery
    romsgrd.check_zero_crossing()
    if romsgrd.zero_crossing is True:
        sodagrd.zero_crossing = True
    else:
        sodagrd.zero_crossing = False

    # Set sodagrd indices (i0:i1, j0:j1) for minimal subgrid around chd
    sodagrd.set_subgrid(romsgrd, k=40)
    
    if True: # check the result of set_subgrid()
        debug0(sodagrd.lon(), sodagrd.lat(), sodagrd.mask(ssh_str),
                  romsgrd.boundary()[0], romsgrd.boundary()[1])
    
    
    
    # Create the initial file
    romsini.create_ini_nc(romsgrd, fillval)
    
    # Gnomonic projections for horizontal interpolations
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=True, M=romsgrd.M).make_kdetree()
    #roms_points = romsgrd.points # we only want data points
    sodagrd.proj2gnom(M=romsgrd.M).make_kdetree() # must use roms_M
    
    '''
    Check that no child data points lie outside of the
    parent domain. If this occurs pysoda2ini cannot function;
    the only solution is to make a new child grid ensuring the above
    condition is met.
    '''
    soda_tri = sp.Delaunay(sodagrd.points) # triangulate full parent 
    tn = soda_tri.find_simplex(romsgrd.points)
    assert not np.any(tn == -1), 'child data points outside parent domain detected'
    

    # Get depths
    #pzr = -sodagrd.depth_r()[::-1]
    #czr = romsgrd.scoord2z_r()
    
    # We want to iterate over smallest axis
    #ax = np.argmin(romsgrd.maskr().shape)
    #if ax == 0:
        #cind2d = '[ij]'
        #cind3d = '[:,ij]'
        #cind2d_self = '[self.ij]'
        #cind3d_self = '[:,self.ij]'
    #elif ax == 1:
        #cind2d = '[:,ij]'
        #cind3d = '[:,:,ij]'
        #cind2d_self = '[:,self.ij]'
        #cind3d_self = '[:,:,self.ij]'
    #else:
        #error
        
    
    
    # Update sodagrd for zero crossing
    sodagrd.fix_zero_crossing = True
    
    print 'Opening file', soda_file
    sfile = SodaData(soda_file, 'SODA', sodagrd)
    sfile.fix_zero_crossing = True
    
    # Tell SodaData object about precomputed i/j limits
    sfile.i0, sfile.i1 = sodagrd.i0, sodagrd.i1
    sfile.j0, sfile.j1 = sodagrd.j0, sodagrd.j1
    
    ptemp = sfile.temp(temp_str)
    psalt = sfile.salt(salt_str)
    pu = sfile.u(u_str)
    pv = sfile.v(v_str)
    
    # 1.4.3 uses cm
    if '1.4.3' in version: # cm to m
        pu *= 0.01
        pv *= 0.01
    
    
    # Get weights for fillmask
    fm_temp = np.array([])
    fm_salt = np.array([])
    fm_u = np.array([])
    fm_v = np.array([])
    for k in np.arange(sodagrd.depths().size):
        junk, t_wt = sfile.fillmask(ptemp[k], sodagrd.mask3d(temp_str)[k])
        junk, s_wt = sfile.fillmask(psalt[k], sodagrd.mask3d(salt_str)[k])
        junk, u_wt = sfile.fillmask(pu[k], sodagrd.mask3d(u_str)[k])
        junk, v_wt = sfile.fillmask(pv[k], sodagrd.mask3d(v_str)[k])
        fm_temp = np.append(fm_temp, t_wt)
        fm_salt = np.append(fm_salt, s_wt)
        fm_u = np.append(fm_u, u_wt)
        fm_v = np.append(fm_v, v_wt)
    fm_temp = fm_temp.reshape(fm_temp.size / 4., 4)
    fm_salt = fm_salt.reshape(fm_salt.size / 4., 4)
    fm_u = fm_u.reshape(fm_u.size / 4., 4)
    fm_v = fm_v.reshape(fm_v.size / 4., 4)
                
    junk, z_wt = sfile.fillmask(sfile.ssh(ssh_str), sodagrd.mask(ssh_str))
                
    for k in np.arange(sodagrd.depths().size):
        ptemp[k] = sfile.fillmask(ptemp[k], sodagrd.mask3d(temp_str)[k], fm_temp[k] )
        psalt[k] = sfile.fillmask(psalt[k], sodagrd.mask3d(salt_str)[k], fm_salt[k] )
        pu[k] = sfile.fillmask(pu[k], sodagrd.mask3d(u_str)[k], fm_u[k] )
        pv[k] = sfile.fillmask(pv[k], sodagrd.mask3d(v_str)[k], fm_v[k] )
            
    pzeta = sfile.fillmask(sfile.ssh(ssh_str), sodagrd.mask(ssh_str), z_wt)
    if '1.4.3' in version: # cm to m
        pzeta *= 0.01
    
    
        
    # Initialise prognostic variables
    temp = np.ma.zeros((romsgrd.mask3d().shape))
    salt = np.ma.zeros((romsgrd.mask3d().shape))
    u = np.ma.zeros((romsgrd.mask3d().shape))
    v = np.ma.zeros((romsgrd.mask3d().shape))
    zeta = np.ma.zeros((romsgrd.maskr().shape))
    
    
    sodagrd.set_2d_depths(romsgrd)
    sodagrd.set_3d_depths(romsgrd)
    
    # Loop over domain minimum axis
    for ij in np.arange(romsgrd.maskr().shape[0]):
        
        romsgrd.ij = ij
        
        if np.any(romsgrd.maskr()[ij]):
            
            #if ax == 0:
            mask = romsgrd.mask3d()[:,ij]
            #elif ax == 1:
                #mask = romsgrd.mask3d()[:,:,ij]
            
            
            # Prepare for horizontal interpolation with horizInterp
            roms_points = romsgrd.proj2gnom(M=romsgrd.M, index_str='[self.ij]').make_kdetree().points
            #soda_ini_indices = romsgrd.get_strip_indices(roms_M, soda_points,
                                #cind2d_self, cof=cof)
            
            print '--- compute a new kde ball (may take a while, but can be speeded up with cKDE if available)'
            soda_ini_indices = sodagrd.kdetree.query_ball_tree(romsgrd.kdetree, balldist)
            soda_ini_indices = np.array(soda_ini_indices).nonzero()[0]
            print '--- got kde ball'
            soda_ini_tri = sp.Delaunay(sodagrd.points[soda_ini_indices])
                                
            # Prepare for vertical interpolation with vertInterp
            #czr_ini = eval('czr' + cind3d)
            
            ## Do the interpolations to get pzr on the chd grid boundaries;
            ## these will be needed by vertInterp
            #pzr_ini = np.tile(pzr, (roms_points.shape[0], 1)).T
            
            sodagrd.set_map_coordinate_weights(romsgrd, j=ij)
            sodagrd.vert_interp()
                  
            ## Weights for map_coordinates (vertical interpolations)
            #vinterp_wghts = romsgrd.get_map_coordinate_weights(czr_ini, pzr_ini)
            #vinterp = vertInterp(vinterp_wghts)
            
            roms_ini_mask = np.tile(romsgrd.maskr()[ij] == 0., (np.int(romsgrd.N), 1))
                  
            if 1 and ij == 5:
                #if ax == 0:
                debug1(sodagrd.lon(), sodagrd.lat(), sodagrd.mask(ssh_str),
                           [soda_ini_indices], (romsgrd.lon()[ij], romsgrd.lat()[ij]))
                #elif ax == 1:
                    #debug1(sodagrd.lon(), sodagrd.lat(), sodagrd.mask(ssh_str),
                           #[soda_ini_indices], (romsgrd.lon()[:,ij], romsgrd.lat()[:,ij]))
            

            # Start the interpolations
            #before = time.time()
            czeta = horizInterp(soda_ini_tri, pzeta.flat[soda_ini_indices])(roms_points)
            htemp = np.ma.zeros((sodagrd.depths().size, czeta.size))
            hsalt = htemp.copy()
            hu = htemp.copy()
            hv = htemp.copy()
            
            for k in np.arange(sodagrd.depths().size):
                htemp[k] = horizInterp(soda_ini_tri, ptemp[k].flat[soda_ini_indices])(roms_points)
                hsalt[k] = horizInterp(soda_ini_tri, psalt[k].flat[soda_ini_indices])(roms_points)
                hu[k] = horizInterp(soda_ini_tri, pu[k].flat[soda_ini_indices])(roms_points)
                hv[k] = horizInterp(soda_ini_tri, pv[k].flat[soda_ini_indices])(roms_points)


            # Prepare for vertical interpolations
            htemp = np.vstack((htemp[0], htemp, htemp[-1]))
            hsalt = np.vstack((hsalt[0], hsalt, hsalt[-1]))
            hu = np.vstack((hu[0], hu, hu[-1]))
            hv = np.vstack((hv[0], hv, hv[-1]))
                        
            # Do vertical interpolations with map_coordinates
            ctemp = sodagrd.vinterp.vert_interp(htemp)
            csalt = sodagrd.vinterp.vert_interp(hsalt)
            cu = sodagrd.vinterp.vert_interp(hu)
            cv = sodagrd.vinterp.vert_interp(hv)
            

            temp[:,ij] = ctemp
            salt[:,ij] = csalt
            u[:,ij] = cu
            v[:,ij] = cv
            zeta[ij] = czeta

            
            print ij
        else:
            print eval('romsgrd.maskr()' + cind2d)

    # Rotate velocities to child angle
    for k in np.arange(romsgrd.N, dtype=np.int):
        u[k], v[k] = romsgrd.rotate(u[k],
                                 v[k], sign=1)
    u = romsgrd.rho2u_3d(u)
    v = romsgrd.rho2v_3d(v)
    
    # Calculate barotropic velocity
    H_u = np.sum(romsgrd.scoord2dz_u() * u, axis=0)
    D_u = np.sum(romsgrd.scoord2dz_u(), axis=0)
    ubar = H_u / D_u
    H_v = np.sum(romsgrd.scoord2dz_v() * v, axis=0)
    D_v = np.sum(romsgrd.scoord2dz_v(), axis=0)
    vbar = H_v / D_v
    
    # Do masking
    zeta = np.ma.masked_where(romsgrd.maskr() == False, zeta)
    temp = np.ma.masked_where(romsgrd.mask3d() == False, temp)
    salt = np.ma.masked_where(romsgrd.mask3d() == False, salt)
    u = np.ma.masked_where(romsgrd.umask3d() == False, u)
    v = np.ma.masked_where(romsgrd.vmask3d() == False, v)
    ubar = np.ma.masked_where(romsgrd.umask() == False, ubar)
    vbar = np.ma.masked_where(romsgrd.vmask() == False, vbar)
    
    # Fill masked areas with fillval
    np.ma.set_fill_value(zeta, 0.)
    np.ma.set_fill_value(temp, 0.)
    np.ma.set_fill_value(salt, 0.)
    np.ma.set_fill_value(u, 0.)
    np.ma.set_fill_value(v, 0.)
    np.ma.set_fill_value(ubar, 0.)
    np.ma.set_fill_value(vbar, 0.)
                    
    # Write to initial file
    nc = netcdf.Dataset(romsini.romsfile, 'a')
    nc.variables['zeta'][:] = zeta.data[np.newaxis]
    nc.variables['temp'][:] = temp.data[np.newaxis]
    nc.variables['salt'][:] = salt.data[np.newaxis]
    nc.variables['u'][:] = u.data[np.newaxis]
    nc.variables['v'][:] = v.data[np.newaxis]
    nc.variables['ubar'][:] = ubar.data[np.newaxis]
    nc.variables['vbar'][:] = vbar.data[np.newaxis]
    
    nc.variables['ocean_time'][:] = ini_time
    nc.variables['tstart'][:] = tstart
    nc.variables['tend'][:] = tend
    
    nc.SODA_file = soda_file
    
    nc.close()
    
    
    
