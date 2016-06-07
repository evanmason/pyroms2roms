# -*- coding: utf-8 -*-
# %run pysoda2roms.py

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

Create a ROMS boundary file based on SODA data

===========================================================================
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
import glob as glob
import time
import scipy.interpolate.interpnd as interpnd
import collections
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime
import calendar

from py_roms2roms import vertInterp, horizInterp, bry_flux_corr, debug0, debug1, debug2
from py_roms2roms import ROMS, RomsGrid, RomsData
from py_mercator2roms import EastGrid, WestGrid, SouthGrid, NorthGrid, prepare_romsgrd




class RomsGrid(RomsGrid):
    '''
    Modify the RomsGrid class
    '''
    
    def lon(self):
        '''
        We need to override 'lon' method of pyroms2roms RomsGrid to
        account for the SODA 0-360 degree grid.
        '''
        lon = self.read_nc('lon_rho',  indices=self.indices)
        
        # If all west of 0 degrees (e.g., for Canaries)
        # then it's easy...
        if np.all(lon < 0.):
            lon += 360.
            
        # Trickiest case (e.g. western Med)
        # no option but to activate a flag indicating
        # modification of the SODA grid and variables
        elif np.logical_and(np.any(lon < 0.),
                            np.any(lon >= 0.)):
            self.zero_crossing = True
        return lon


    def check_zero_crossing(self):
        if np.logical_and(np.any(self.lon() < 0.),
                          np.any(self.lon() >= 0.)):
            self.zero_crossing = True
        


class SodaData (RomsData):
    '''
    SODA data class (inherits from RomsData class)
    '''
    def __init__(self, filename, model_type, sodagrd):
        """
        soda_file, 'SODA', sodagrd
        """
        super(SodaData, self).__init__(filename, model_type)
        self._lon = sodagrd._lon
        self._lat = sodagrd._lat
        # Update sodagrd for zero crossing
        if sodagrd.zero_crossing is True:
                self.fix_zero_crossing = True
        # Tell SodaData object about i/j limits
        self.i0, self.i1 = sodagrd.i0, sodagrd.i1
        self.j0, self.j1 = sodagrd.j0, sodagrd.j1
    
    
    def lon(self):
        if self.fix_zero_crossing is True:
            lon1 = self._lon[self.j0:self.j1, :self.i0]
            lon0 = self._lon[self.j0:self.j1, self.i1:] - 360
            return np.concatenate((lon0, lon1), axis=1)
        else:
            return self._lon[self.j0:self.j1, self.i0:self.i1]
    
    
    def lat(self):
        if self.fix_zero_crossing is True:
            lat1 = self._lat[self.j0:self.j1, :self.i0]
            lat0 = self._lat[self.j0:self.j1, self.i1:]
            return np.concatenate((lat0, lat1), axis=1)
        else:
            return self._lat[self.j0:self.j1, self.i0:self.i1]
    
    
    def ssh(self, var):
        if self.fix_zero_crossing is True:
            ssh1 = self.read_nc(var, indices='[self.j0:self.j1, :self.i0]')
            ssh0 = self.read_nc(var, indices='[self.j0:self.j1, self.i1:]')
            return np.concatenate((ssh0, ssh1), axis=1)
        else:
            return self.read_nc(var, indices=self.indices)
    
    
    def _vars3d(self, var):
        var1 = self.read_nc(var, '[::-1, self.j0:self.j1,:self.i0]')
        var0 = self.read_nc(var, '[::-1, self.j0:self.j1,self.i1:]')
        return np.concatenate((var0, var1), axis=2)
    
    
    def temp(self, var):
        if self.fix_zero_crossing is True:
            return self._vars3d(var)
        else:
            return self.read_nc(var, '[::-1, self.j0:self.j1, self.i0:self.i1]')
    
    
    def salt(self, var):
        if self.fix_zero_crossing is True:
            return self._vars3d(var)
        else:
            return self.read_nc(var, '[::-1, self.j0:self.j1, self.i0:self.i1]')
    
    
    def u(self, var):
        if self.fix_zero_crossing is True:
            return self._vars3d(var)
        else:
            return self.read_nc(var, '[::-1, self.j0:self.j1, self.i0:self.i1]')
    
    
    def v(self, var):
        if self.fix_zero_crossing is True:
            return self._vars3d(var)
        else:
            return self.read_nc(var, '[::-1, self.j0:self.j1, self.i0:self.i1]')
            
    
    
    
class SodaGrid (ROMS):
    '''
    SODA grid class
    '''
    def __init__(self, filename, model_type):
        '''
        
        '''
        super(SodaGrid, self).__init__(filename, model_type)
        print 'Initialising SodaGrid', filename
        try:
            self._lon = self.read_nc('lon', indices='[:]')
            self._lat = self.read_nc('lat', indices='[:]')
        except Exception:
            self._lon = self.read_nc('LON', indices='[:]')
            self._lat = self.read_nc('LAT', indices='[:]')
        self._lon, self._lat = np.meshgrid(self._lon,
                                           self._lat)
        
        
    def lon(self):
        if self.fix_zero_crossing is True:
            lon1 = self._lon[self.j0:self.j1, :self.i0]
            lon0 = self._lon[self.j0:self.j1, self.i1:] - 360
            return np.concatenate((lon0, lon1), axis=1)
        else:
            return self._lon
    
    def lat(self):
        if self.fix_zero_crossing is True:
            lat1 = self._lat[self.j0:self.j1, :self.i0]
            lat0 = self._lat[self.j0:self.j1, self.i1:]
            return np.concatenate((lat0, lat1), axis=1)
        else:
            return self._lat
        
    #def _lonlat(self):
        #try:
            #lon = self.read_nc('lon', indices=''.join(('[', self.indices[18:])))
            #lat = self.read_nc('lat', indices=''.join((self.indices[:16], ']')))
        #except:
            #lon = self.read_nc('LON', indices=''.join(('[', self.indices[18:])))
            #lat = self.read_nc('LAT', indices=''.join((self.indices[:16], ']')))
        #lon, lat = np.meshgrid(lon, lat)
        #return lon, lat
        
    #def lon(self):
        #if self.fix_zero_crossing is True:
            #try:
                #lon1 = self.read_nc('lon', indices='[:self.i0]')
                #lon0 = self.read_nc('lon', indices='[self.i1:]') - 360.
                #lat = self.read_nc('lat', indices=''.join((self.indices[:16], ']')))
            #except:
                #lon1 = self.read_nc('LON', indices='[:self.i0]')
                #lon0 = self.read_nc('LON', indices='[self.i1:]') - 360.
                #lat = self.read_nc('LAT', indices=''.join((self.indices[:16], ']')))
            #lon = np.append(lon0, lon1)
            #return np.meshgrid(lon, lat)[0]
        #else:
            #return self._lonlat()[0]
        
    #def lat(self):
        #if self.fix_zero_crossing is True:
            #try:
                #lon1 = self.read_nc('lon', indices='[:self.i0]')
                #lon0 = self.read_nc('lon', indices='[self.i1:]') - 360.
                #lat = self.read_nc('lat', indices=''.join((self.indices[:16], ']')))
            #except:
                #lon1 = self.read_nc('LON', indices='[:self.i0]')
                #lon0 = self.read_nc('LON', indices='[self.i1:]') - 360.
                #lat = self.read_nc('LAT', indices=''.join((self.indices[:16], ']')))
            #lon = np.append(lon0, lon1)
            #return np.meshgrid(lon, lat)[1]
        #else:
            #return self._lonlat()[1]

    def mask(self, var):
        if self.fix_zero_crossing is True:
            mask1 = self.read_nc(var, indices='[self.j0:self.j1, :self.i0]')
            mask0 = self.read_nc(var, indices='[self.j0:self.j1, self.i1:]')
            mask = np.concatenate((mask0, mask1), axis=1)
        else:
            mask = self.read_nc(var, indices=self.indices)
        mask[mask > -999.] = 1.
        mask[mask < -999.] = 0.
        return mask
        
    def mask3d(self, var):
        '''
        3d mask
        var should be a string: 'temp', 'salt', etc...
        '''
        if self.fix_zero_crossing is True:
            mask1 = self.read_nc(var, indices='[::-1, self.j0:self.j1, :self.i0]')
            mask0 = self.read_nc(var, indices='[::-1, self.j0:self.j1, self.i1:]')
            mask3d = np.concatenate((mask0, mask1), axis=2)
        else:
            mask3d = self.read_nc(var, indices='[::-1, self.j0:self.j1, selfi0:self.i1]')
        mask3d[mask3d > -999.] = 1.
        mask3d[mask3d < -999.] = 0.
        return mask3d
        
        
    def depths(self):
        try:
            return self.read_nc('depth', indices='[:]')
        except:
            return self.read_nc('DEPTH', indices='[:]')
        
        
    def set_2d_depths(self, romsgrd):
        '''
        
        '''
        self._2d_depths = np.tile(-self.depths()[::-1],
                                  (romsgrd.h().size, 1)).T
        return self

    def set_3d_depths(self, romsgrd):
        '''
        
        '''
        self._3d_depths = np.tile(-self.depths()[::-1],
                                  (romsgrd.h().shape[1],
                                   romsgrd.h().shape[0], 1)).T
        return self
    
    def set_map_coordinate_weights(self, romsgrd, j=None):
        '''
        Order : set_2d_depths or set_3d_depths
        '''
        #self.set_2d_depths(romsgrd)
        if j is not None:
            soda_depths = self._3d_depths[:,j]
            roms_depths = romsgrd.scoord2z_r()[:,j]
        else:
            soda_depths = self._2d_depths
            roms_depths = romsgrd.scoord2z_r()
        self.mapcoord_weights = romsgrd.get_map_coordinate_weights(
                                          roms_depths, soda_depths)
        return self


    def vert_interp(self):
        '''
        Vertical interpolation using ndimage.map_coordinates()
          See vertInterp class in py_roms2roms.py
        Requires self.mapcoord_weights set by set_map_coordinate_weights()
        '''
        self.vinterp = vertInterp(self.mapcoord_weights)
        return self




def prepare_sodagrd(sodagrd, romsgrd):
    '''
    '''
    if sodagrd.zero_crossing is True: # Update sodagrd for zero crossing
        sodagrd.fix_zero_crossing = True
    #sodagrd.proj2gnom(ignore_land_points=False, M=romsgrd.M)
    #sodagrd.child_contained_by_parent(romsgrd)
    #sodagrd.make_kdetree()#.get_fillmask_cofs()
    return sodagrd
    
    
    
if __name__ == '__main__':
    
    '''
    pysoda2roms
    
    Information about SODA:
      http://www.atmos.umd.edu/~carton/index2_files/soda.htm
    
    SODA may be downloaded from:
      http://soda.tamu.edu/assim
    
    Evan Mason 2012
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # SODA information
    version = '2.1.6' # Note, uses m for u, v, ssh
    #version = '1.4.3' # Note, uses cm for u, v, ssh
    #version = '2.2.8'

    monthly = False # True for MONTHLY data, else 5DAY
    #soda_dir     = '/Users/emason/data/SODA_2.1.6/'
    #soda_dir     = '/marula/emason/data/SODA/2.1.6/'
    #soda_dir     = '/marula/emason/data/SODA/2.1.6/MONTHLY/'
<<<<<<< local
    #soda_dir     = '/marula/emason/data/SODA/2.1.6/5DAY/'
=======
    soda_dir     = '/marula/emason/data/SODA/2.1.6/5DAY/'
>>>>>>> other
    #soda_dir     = '/shared/emason/SODA/2.2.8/MONTHLY/'
<<<<<<< local
    soda_dir     = '/Users/emason/data/SODA_2.2.8/'
=======
    #soda_dir     = '/Users/emason/data/SODA_2.2.8/'
>>>>>>> other

    soda_grd = ''.join(('SODA_', version))


    # Child ROMS information
    #roms_dir     = '../'
    #roms_dir     = '/marula/emason/runs2012/MedSea5/'
    #roms_dir     = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir     = '/home/emason/'
    #roms_dir     = '/marula/emason/runs2014/MedCan5/'
    #roms_dir     = '/Users/emason/runs/runs2014/MedCan5/'
    roms_dir = '/Users/emason/runs2009/na_2009_7pt5/'
    #roms_dir = '/marula/emason/runs2014/NA75_IA/'
    
    #roms_grd     = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd     = 'grd_MedCan5.nc'
    roms_grd     = 'roms_grd_NA2009_7pt5km.nc'
    
    
    if 'roms_grd_NA2009_7pt5km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
    elif 'grd_nwmed_2km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=250, N=40)
    else:
        print 'No sigma params defined for grid: %s' %roms_grd
        raise Exception
    
    
    # Child ROMS boundary file information
    obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed
    bry_cycle    =  0.     # days, 0 means no cycle
    #bry_filename = 'bry_SODA5day_NA7pt5km.nc' # bry filename
    #bry_filename = 'bry_MedCan5_SODA_2.2.8_MONTHLY.nc'
<<<<<<< local
    bry_filename = 'bry_NA75_SODA_2.2.8_MONTHLY.nc.CORRECTED'
    first_file   = '198412' # first/last SODA file, 
    last_file    = '201112'
    #first_file   = '19850104-19850109' # first/last SODA file, 
    #last_file    = '20081224-20081229'
=======
    #bry_filename = 'bry_NA75_SODA_2.2.8_MONTHLY.nc.CORRECTED'
    bry_filename = 'bry_NA75_SODA_2.1.6_5DAY.nc'
    #first_file   = '198412' # first/last SODA file, 
    #last_file    = '201101'
    first_file   = '19850104-19850109' # first/last SODA file, 
    last_file    = '20081224-20081229'
>>>>>>> other
    
    day_zero = '19850101'
    
    
    balldist = 250000. # meters

    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    
    if monthly:
        assert len(first_file) == 6, 'first_file must be a length six string'
        assert len(last_file) == 6,  'last_file must be a length six string'
    else:
        assert len(first_file) == 17, 'first_file must be a length seventeen string'
        assert len(last_file) == 17,  'last_file must be a length seventeen string'

    fillval = 9999.

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

    day_zero = datetime(int(day_zero[:4]), int(day_zero[4:6]), int(day_zero[6:]))
    day_zero = plt.date2num(day_zero)

    

    # Initialise SodaGrid and RomsGrid objects for both parent and child grids
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
    
    romsgrd = prepare_romsgrd(romsgrd)
    
    romsgrd.set_bry_dx()
    romsgrd.set_bry_maskr()
    romsgrd.set_bry_areas()
    
    # Get surface areas of open boundaries and prepare boundary grids
    chd_bry_surface_areas = []
    boundary_grids = []
    for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):
        if 'west' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_west.sum(axis=0) * romsgrd.maskr_west)
            boundary_grids.append(WestGrid(roms_dir + roms_grd, sigma_params, 'ROMS'))
        elif 'east' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_east.sum(axis=0) * romsgrd.maskr_east)
            boundary_grids.append(EastGrid(roms_dir + roms_grd, sigma_params, 'ROMS'))
        elif 'south' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_south.sum(axis=0) * romsgrd.maskr_south)
            boundary_grids.append(SouthGrid(roms_dir + roms_grd, sigma_params, 'ROMS'))
        elif 'north' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_north.sum(axis=0) * romsgrd.maskr_north)
            boundary_grids.append(NorthGrid(roms_dir + roms_grd, sigma_params, 'ROMS'))
    # Get total surface of open boundaries
    chd_bry_total_surface_area = np.array([area.sum() for area in chd_bry_surface_areas]).sum()
    
    
    soda_grd = soda_grd + '_' + first_file + '.cdf'
    sodagrd = SodaGrid(''.join((soda_dir, soda_grd)), 'SODA')
    # Activate flag for SODA zero crossing trickery
    romsgrd.check_zero_crossing()
    if romsgrd.zero_crossing is True:
        print 'The ROMS domain straddles the zero-degree meridian'
        sodagrd.zero_crossing = True
    else:
        sodagrd.zero_crossing = False

    # Set sodagrd indices (i0:i1, j0:j1) for minimal subgrid around chd
    sodagrd.set_subgrid(romsgrd, k=40)
    sodagrd = prepare_sodagrd(sodagrd, romsgrd)
    
    if True: # check the result of set_subgrid()
        debug0(sodagrd.lon(), sodagrd.lat(), sodagrd.mask(ssh_str),
                  romsgrd.boundary()[0], romsgrd.boundary()[1])

    

    # Set up a RomsData object for the boundary file
    romsbry = RomsData(roms_dir + bry_filename, 'ROMS')
    romsbry.first_file = first_file
    romsbry.last_file = last_file


    # Create the boundary file
    romsbry.create_bry_nc(romsgrd, obc_dict, bry_cycle, fillval, 'pysoda2roms')


    # Gnomonic projections for horizontal interpolations
    #romsgrd.make_gnom_transform() # make Gnomonic basemap object, called as romsgrd.M
    #romsgrd.proj2gnom(ignore_land_points=True) # no_mask == True as we only want data points
    
    # Get depths
    #pzr = -sodagrd.depths()[::-1]
    #czr = romsgrd.scoord2z_r()
 
    ## debug1 shows results of get_strip_indices() at each open boundary
    #if True:
        #debug1(sodagrd.lon(), sodagrd.lat(), sodagrd.mask(ssh_str),
                                       #par_bry_indices, romsgrd.boundary())



    # Get list of roms parent data files
    if monthly:
        soda_files = sorted(glob.glob(soda_dir + 'SODA_' + version + '_??????.cdf'))
    else:
        soda_files = sorted(glob.glob(soda_dir + 'SODA_' + version +
                                      '_????????-????????.cdf'))
    
    tind       = 0     # index for writing records to bry file
    if monthly:
        bry_time = 0     # bry_time in days
    else:
        bry_time = -2.5
    active     = False # flag to soda_files for loop
    fm_weights = False # flag, if False then compute fillmask weights




    
    '''
    Start main loop over soda_files list here
    '''
    for soda_file in soda_files:
      
        if first_file in soda_file:
            active = True


        if active:
            
            print 'Opening file', soda_file
            sodadata = SodaData(soda_file, 'SODA', sodagrd)
            
            
            # Time loop over ocean_time
            ot_loop_start = time.time()

            # Get bry time (days)
            if monthly:
                # Monthly SODA
                stime = soda_file.split('/')[-1].split('_')[-1].split('.')[0]
                mid_month = 0.5 * calendar.monthrange(int(stime[:4]), int(stime[4:]))[1]
                # Note that the second half of the month (mid_month) is added to bry_time
                bry_time += mid_month # *after* bry_time has been written to bry file
                bry_time = datetime(int(stime[:4]), int(stime[4:]), int(np.round(mid_month)))
                bry_time = plt.date2num(bry_time) - day_zero
            else:
                # 5-day SODA
                stime = soda_file.split('/')[-1].split('_')[-1].split('.')[0].split('-')[0]
                bry_time = datetime(int(stime[:4]), int(stime[4:6]), int(stime[6:]))
                bry_time = plt.date2num(bry_time)
                bry_time += 2.5 # add 2.5 days
                bry_time -= day_zero

            # Read in variables
            ptemp = sodadata.temp(temp_str)
            psalt = sodadata.salt(salt_str)
            pu = sodadata.u(u_str) 
            pv = sodadata.v(v_str)
            
            
            if '1.4.3' in version: # cm to m
                pu *= 0.01
                pv *= 0.01

            
            # Get weights for fillmask
            if fm_weights is False:
                fm_temp = np.array([])
                fm_salt = np.array([])
                fm_u = np.array([])
                fm_v = np.array([])
                for k in np.arange(sodagrd.depths().size):
                    junk, t_wt = sodadata.fillmask(ptemp[k], sodagrd.mask3d(temp_str)[k])
                    junk, s_wt = sodadata.fillmask(psalt[k], sodagrd.mask3d(salt_str)[k])
                    junk, u_wt = sodadata.fillmask(pu[k], sodagrd.mask3d(u_str)[k])
                    junk, v_wt = sodadata.fillmask(pv[k], sodagrd.mask3d(v_str)[k])
                    fm_temp = np.append(fm_temp, t_wt)
                    fm_salt = np.append(fm_salt, s_wt)
                    fm_u = np.append(fm_u, u_wt)
                    fm_v = np.append(fm_v, v_wt)
                fm_temp = fm_temp.reshape(fm_temp.size / 4., 4)
                fm_salt = fm_salt.reshape(fm_salt.size / 4., 4)
                fm_u = fm_u.reshape(fm_u.size / 4., 4)
                fm_v = fm_v.reshape(fm_v.size / 4., 4)
                
                junk, z_wt = sodadata.fillmask(sodadata.ssh(ssh_str), sodagrd.mask(ssh_str))
                
                fm_weights = True
            
            for k in np.arange(sodagrd.depths().size):
                ptemp[k] = sodadata.fillmask( ptemp[k], sodagrd.mask3d(temp_str)[k],  fm_temp[k] )
                psalt[k] = sodadata.fillmask( psalt[k], sodagrd.mask3d(salt_str)[k],  fm_salt[k] )
                pu[k]    = sodadata.fillmask( pu[k],    sodagrd.mask3d(u_str)[k],  fm_u[k] )
                pv[k]    = sodadata.fillmask( pv[k],    sodagrd.mask3d(v_str)[k],  fm_v[k] )
            
            if '1.4.3' in version: # cm to m
                pzeta = 0.01 * sodadata.fillmask(sodadata.ssh(ssh_str), sodagrd.mask(ssh_str), z_wt)
            else:
                pzeta = sodadata.fillmask(sodadata.ssh(ssh_str), sodagrd.mask(ssh_str), z_wt)
            
            
            # Initialise lists to be used during bry flux correction
            boundarylist = []
            uvlist       = []
            uvbarlist    = []    
                
            # Loop over boundaries
            bryind1 = 0
            
            for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):
        
                print '------ processing %sern boundary' %open_boundary
                
                if 'west' in open_boundary and flag:
                     bry_romsgrd = boundary_grids[bryind1]
                elif 'east' in open_boundary and flag:
                     bry_romsgrd = boundary_grids[bryind1]
                elif 'north' in open_boundary and flag:
                     bry_romsgrd = boundary_grids[bryind1]
                elif 'south' in open_boundary and flag:
                     bry_romsgrd = boundary_grids[bryind1]
                
                bry_romsgrd = prepare_romsgrd(bry_romsgrd)
                
                sodagrd.set_2d_depths(bry_romsgrd).set_map_coordinate_weights(bry_romsgrd)
                sodagrd.vert_interp()
                
                sodagrd.proj2gnom(ignore_land_points=False, M=bry_romsgrd.M)
                if first_file in soda_file:
                    sodagrd.child_contained_by_parent(bry_romsgrd)
                sodagrd.make_kdetree()#.get_fillmask_cofs()
                ballpoints = sodagrd.kdetree.query_ball_tree(bry_romsgrd.kdetree, r=balldist)
                sodagrd.ball = np.array(np.array(ballpoints).nonzero()[0])
                sodagrd.tri = sp.Delaunay(sodagrd.points[sodagrd.ball])
                
                czeta  = horizInterp(sodagrd.tri, pzeta.flat[sodagrd.ball])(bry_romsgrd.points)
                htemp  = np.ma.zeros((sodagrd.depths().size, czeta.size))
                hsalt  = htemp.copy()
                hu     = htemp.copy()
                hv     = htemp.copy()
                
                
                for k in np.arange(sodagrd.depths().size):
                    htemp[k] = horizInterp(sodagrd.tri, ptemp[k].flat[sodagrd.ball])(bry_romsgrd.points)
                    hsalt[k] = horizInterp(sodagrd.tri, psalt[k].flat[sodagrd.ball])(bry_romsgrd.points)
                    hu[k]    = horizInterp(sodagrd.tri, pu[k].flat[sodagrd.ball])(bry_romsgrd.points)
                    hv[k]    = horizInterp(sodagrd.tri, pv[k].flat[sodagrd.ball])(bry_romsgrd.points)
                    hu[k], hv[k] = bry_romsgrd.rotate(hu[k],
                                                      hv[k], sign=1) # rotate to child angle

                # Prepare for vertical interpolations
                htemp = np.vstack((htemp[0], htemp, htemp[-1]))
                hsalt = np.vstack((hsalt[0], hsalt, hsalt[-1]))
                hu    = np.vstack((hu[0],    hu,    hu[-1]))
                hv    = np.vstack((hv[0],    hv,    hv[-1]))
                
                # Do vertical interpolations with map_coordinates
                ctemp = sodagrd.vinterp.vert_interp(htemp)
                csalt = sodagrd.vinterp.vert_interp(hsalt)
                cu    = sodagrd.vinterp.vert_interp(hu)
                cv    = sodagrd.vinterp.vert_interp(hv)
                
                if 0:
                    # debug_ind is index along boundary
                    debug_ind = 10
                    debug2(debug_ind, boundary)
                        

                        
                # Calculate barotropic velocity
                H_u   = np.sum((bry_romsgrd.scoord2dz() * cu), axis=0)
                D_u   = np.sum((bry_romsgrd.scoord2dz()), axis=0)
                cubar = H_u / D_u
                H_v   = np.sum((bry_romsgrd.scoord2dz() * cv), axis=0)
                D_v   = np.sum((bry_romsgrd.scoord2dz()), axis=0)
                cvbar = H_v / D_v
                
                # From horizontal rho to u, v points
                if open_boundary in ('north', 'south'):
                    cu = 0.5 * (cu[:,:-1] + cu[:,1:])
                    cubar = 0.5 * (cubar[:-1] + cubar[1:])
                elif open_boundary in ('east', 'west'):
                    cv = 0.5 * (cv[:,:-1] + cv[:,1:])
                    cvbar = 0.5 * (cvbar[:-1] + cvbar[1:])
                
                # Apply the masking
                #czeta = np.ma.masked_where(mask[0] == True, czeta)
                #ctemp = np.ma.masked_where(mask == True, ctemp)
                #csalt = np.ma.masked_where(mask == True, csalt)
                #if boundary in ('north', 'south'):
                    #cu = np.ma.masked_where(mask[:,:-1] == True, cu)
                    #cv = np.ma.masked_where(mask == True, cv)
                    #cubar = np.ma.masked_where(mask[0,:-1] == True, cubar)
                    #cvbar = np.ma.masked_where(mask[0] == True, cvbar)
                #elif boundary in ('east', 'west'):
                    #cu = np.ma.masked_where(mask == True, cu)
                    #cv = np.ma.masked_where(mask[:,:-1] == True, cv)
                    #cubar = np.ma.masked_where(mask[0] == True, cubar)
                    #cvbar = np.ma.masked_where(mask[0,:-1] == True, cvbar)
                    
                    
                # Write to boundary file
                nc = netcdf.Dataset(romsbry.romsfile, 'a')
                    
                if bryind1 == 0:
                    nc.variables['bry_time'][tind] = np.float(bry_time)
                    
                # Fill masked areas with 0
                czeta = np.ma.filled(czeta, 0.)
                ctemp = np.ma.filled(ctemp, 0.)
                csalt = np.ma.filled(csalt, 0.)
                    
                nc.variables['zeta_%s' % open_boundary][tind] = np.ma.squeeze(czeta)
                nc.variables['temp_%s' % open_boundary][tind] = np.ma.squeeze(ctemp)
                nc.variables['salt_%s' % open_boundary][tind] = np.ma.squeeze(csalt)
                    
                    
                # Save only tangential velocities for now because we
                # need to apply bry flux correction to normal velocities
                if open_boundary in ('north', 'south'):
                    cu = np.ma.filled(cu, 0.)
                    cubar = np.ma.filled(cubar, 0.)
                    nc.variables['u_%s' % open_boundary][tind] = cu
                    nc.variables['ubar_%s' % open_boundary][tind] = cubar
                        
                elif open_boundary in ('east', 'west'):
                    cv = np.ma.filled(cv, 0.)
                    cvbar = np.ma.filled(cvbar, 0.)
                    nc.variables['v_%s' % open_boundary][tind] = cv
                    nc.variables['vbar_%s' % open_boundary][tind] = cvbar
                    
                    
                '''
                Once all boundaries in obc_dict computed then apply volume 
                conservation to normal velocities and save
                '''
                if bryind1 == np.sum(obc_dict.values()) - 1: # all boundaries processed
                        
                    boundarylist.append(open_boundary)
                        
                    # Collect only normal velocities
                    if open_boundary in ('north', 'south'):
                        uvlist.append(cv)
                        uvbarlist.append(cvbar)
                            
                    elif open_boundary in ('east', 'west'):
                        uvlist.append(cu)
                        uvbarlist.append(cubar)

                    fc = bry_flux_corr(boundarylist,
                                           chd_bry_surface_areas,
                                           chd_bry_total_surface_area,
                                           uvbarlist)
                        
                    print '------ barotropic velocity correction:', fc, 'm/s'
                        
                    for fcind, fc_boundary in enumerate(boundarylist):
                            
                        # Correct and save the normal velocities
                        if 'north' in fc_boundary:
                            c_fc = np.ma.filled(uvlist[fcind] + fc, 0.)
                            cbar_fc = np.ma.filled(uvbarlist[fcind] + fc, 0.)
                            nc.variables['v_north'][tind] = c_fc
                            nc.variables['vbar_north'][tind] = cbar_fc
                            
                        elif  'south' in fc_boundary:
                            c_fc = np.ma.filled(uvlist[fcind] - fc, 0.)
                            cbar_fc = np.ma.filled(uvbarlist[fcind] - fc, 0.)
                            nc.variables['v_south'][tind] = c_fc
                            nc.variables['vbar_south'][tind] = cbar_fc
                            
                        elif 'west' in fc_boundary:
                            c_fc = np.ma.filled(uvlist[fcind] - fc, 0.)
                            cbar_fc = np.ma.filled(uvbarlist[fcind] - fc, 0.)
                            nc.variables['u_west'][tind] = c_fc
                            nc.variables['ubar_west'][tind] = cbar_fc
                                
                        elif 'east' in fc_boundary:
                            c_fc = np.ma.filled(uvlist[fcind] + fc, 0.)
                            cbar_fc = np.ma.filled(uvbarlist[fcind] + fc, 0.)
                            nc.variables['u_east'][tind] = c_fc
                            nc.variables['ubar_east'][tind] = cbar_fc
                    
                    
                else: # add to the list until all boundaries accounted for
                        
                    boundarylist.append(open_boundary)
                        
                    # Collect only normal velocities
                    if open_boundary in ('north', 'south'):
                        uvlist.append(cv)
                        uvbarlist.append(cvbar)
                            
                    elif open_boundary in ('east', 'west'):
                        uvlist.append(cu)
                        uvbarlist.append(cubar)

                # Update cycle length to be always 'mid_month' days greater
                # than bry_time
                if bry_cycle == 0.:
                    if monthly:
                        nc.variables['bry_time'].cycle_length = bry_time + mid_month # days
                    else:
                        nc.variables['bry_time'].cycle_length = bry_time
                nc.close()

                bryind1 += 1
                
        # Update tind for each record
        tind += 1
            
        # Update bry_time (brings it to end of current month)
        #bry_time += mid_month


        print '---time to process', time.time() - ot_loop_start, 'seconds'    
                

        if last_file in soda_file: active = False
        
    print 'all done'
            
            
      














