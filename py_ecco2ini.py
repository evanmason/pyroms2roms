# -*- coding: utf-8 -*-
# %run py_ecco2ini.py

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

Create a ROMS initial file based on ECCO2 data

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
from collections  import OrderedDict
from mpl_toolkits.basemap import Basemap
from datetime import datetime

from py_roms2roms import vertInterp, horizInterp
from py_roms2roms import ROMS, RomsGrid, debug0, debug1
#from pysoda2roms import EccoData, SodaGrid, RomsGrid
from py_ecco2roms import EccoData, get_list_of_ecco_files
from py_ecco2roms import prepare_romsgrd
from pyroms2ini import RomsData


class EccoDataIni (EccoData):
    '''
    EccoDataIni class (inherits from EccoData class)
    '''
    def __init__(self, filenames, model_type, ecco_var, romsgrd, **kwargs):
        """
        Creates a new Ecco ini data object.
        
        Parameters
        ----------
        
        *filenames* : list of Ecco nc files.
        *model_type* : string specifying Ecco model.
        *romsgrd* : a `RomsGrid` instance.
        
        """
        super(EccoDataIni, self).__init__(filenames, model_type, ecco_var, romsgrd, **kwargs)
        
        
    def interp2romsgrd(self):
        '''
        Modification of EccoData class definition *interp2romsgrd*
        '''
        if '3D' in self.dimtype:
            for k in np.arange(self._depths.size):
                self._interp2romsgrd(k)
        else:
            self._interp2romsgrd()
        return self


    def vert_interp(self, j):
        '''
        Modification of EccoData class definition *vert_interp*
        '''
        vinterp = vertInterp(self.mapcoord_weights)
        self.dataout[:,j] = vinterp.vert_interp(self.datatmp[::-1,j])
        return self


    def set_2d_depths(self):
        '''
        
        '''
        m, l = self.romsgrd.h().shape[0], self.romsgrd.h().shape[1]
        self._2d_depths = np.tile(-self.depths()[::-1],
                                  (l, m, 1)).T
        return self


def prepare_ecco(ecco):
    ecco.proj2gnom(ignore_land_points=False, M=ecco.romsgrd.M)
    #ecco.child_contained_by_parent(ecco.romsgrd)
    ecco.make_kdetree().get_fillmask_cofs()
    #ballpoints = ecco.kdetree.query_ball_tree(ecco.romsgrd.kdetree, r=balldist)
    #ecco.ball = np.array(np.array(ballpoints).nonzero()[0])
    #ecco.tri = sp.Delaunay(ecco.points[ecco.ball])
    if '3D' in ecco.dimtype:
        ecco.set_3d_depths()
        #ecco.vert_interp()
    return ecco



if __name__ == '__main__':
    
    '''
    py_ecco2ini (Python version of r2r_ini.m written in Matlab).


    
    Evan Mason, IMEDEA, 2014
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # ECCO2 information
    ecco_domain = 'ecco2.jpl.nasa.gov'
    ecco_url = 'http://ecco2.jpl.nasa.gov:80/opendap/'
    ecco_path = 'data1/cube/cube92/lat_lon/quart_90S_90N/'
    
    ecco_vars = OrderedDict([('SSH', 'SSH.nc'),
                             ('TEMP', 'THETA.nc'),
                             ('SALT', 'SALT.nc'),
                             ('U','UVEL.nc'),
                             ('V','VVEL.nc')])
    

    # Child ROMS information
    #roms_dir     = '../'
    #roms_dir     = '/marula/emason/runs2012/MedSea5/'
    #roms_dir     = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir     = '/marula/emason/runs2009/na_2009_7pt5km/'
    #roms_dir     = '/marula/emason/runs2014/na6km/'
    roms_dir     = '/marula/emason/runs2014/na75/'
    #roms_dir = '/marula/emason/runs2014/nwmed5km/'
    #roms_dir = '/Users/emason/runs2009/na_2009_7pt5/'
    #roms_dir     = '/marula/emason/runs2014/NWMED2/'
    
    #roms_grd     = 'roms_grd_NA2009_7pt5km.nc'
    roms_grd = 'roms_grd_NA2014_7pt5km.nc'
    #roms_grd     = 'grd_na6km.nc'
    #roms_grd = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd = 'grd_nwmed5km_NARROW_STRAIT.nc'

    if 'roms_grd_NA2014_7pt5km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
    elif 'grd_na6km.nc' in roms_grd:
        sigma_params = dict(theta_s=6., theta_b=0, hc=120, N=32)
    elif 'grd_nwmed_2km.nc' in roms_grd:
        sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
    elif 'grd_nwmed5km_NARROW_STRAIT.nc' in roms_grd:
        sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
    else:
        print 'No sigma parameters defined for grid: %s' %roms_grd
        raise Exception
    
    #ecco_ini_date = ecco_file.split('.')[-2]
    
    ini_date = '19920102'
    #ini_date = '20050102'
    day_zero = '19850101'
    
    # Initial file
    ini_time = 0.  # days
    tstart = 0. 
    tend = 0. 
    
    # Set ini filename
    #ini_filename = 'ini_MedSea5_%s.nc' %(ecco_file.split('_')[-1].split('.')[0])
    #ini_filename = 'ini_SODA5day_NA7pt5km_%s.nc' %(ecco_file.split('_')[-1].split('-')[0])
    #ini_filename = 'ini_MedCan5_%s.198501.nc' %(ecco_file.split('_')[-1].split('-')[0])
    ini_filename = 'ini_NA2015_ecco_198501.nc' 
    #ini_filename = 'ini_nwmed5km_ecco2_199201.nc' 
    #ini_filename = 'ini_nwmed5km_ecco2_TEST_CONSTANT.nc' 
    #ini_filename = 'ini_na6km_198501_Akima.nc' 
    #ini_filename = 'ini_na5_198501_BILINEAR.nc'
    
    
    #balldist = 250000. # distance (m) for kde_ball (should be 2dx at least?)



    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    
    fillval = 9999
    
    
    # Initialise RomsGrid object for child grid
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, model_type='ROMS')
    romsgrd = prepare_romsgrd(romsgrd)

    # Set up a RomsData object for the initial file
    romsini = RomsData(roms_dir + ini_filename, model_type='ROMS')

    # Create an initial file
    romsini.create_ini_nc(romsgrd, fillval, created_by='py_ecco2ini.py')
    
    ini_date_num = datetime(int(ini_date[:4]), int(ini_date[4:6]), int(ini_date[6:]))
    ini_date_num = plt.date2num(ini_date_num) + 0.5
    
    day_zero_num = datetime(int(day_zero[:4]), int(day_zero[4:6]), int(day_zero[6:]))
    day_zero_num = plt.date2num(day_zero_num)
    
    if ini_date_num != day_zero_num:
        ini_time = ini_date_num - day_zero_num # days
        
    
    for ecco_var, ecco_subdir in zip(ecco_vars.keys(), ecco_vars.values()):
        
        ecco_files = get_list_of_ecco_files(ecco_domain, ecco_url, ecco_path, ecco_subdir)
        ecco_file = [s for s in ecco_files if ini_date in s]
        
        assert len(ecco_file) > 0, 'Specified ecco_file not found in ecco_files.'
        
        print '\nProcessing variable *%s*' %ecco_var
        proceed = False
        
        ecco = EccoDataIni(ecco_file, 'Ecco', ecco_var, romsgrd)
        ecco = prepare_ecco(ecco)
        
    
        #if 1: # debug
            #plt.pcolormesh(ecco.lon(), ecco.lat(), ecco.maskr())
            #plt.scatter(ecco.lon().flat[ecco.ball], ecco.lat().flat[ecco.ball],
                        #s=10,c='g',edgecolors='none')
            #plt.plot(romsgrd.boundary()[0], romsgrd.boundary()[1], 'w')
            #plt.axis('image')
            #plt.show()
        
        # Read in variables and interpolate to romsgrd
        ecco.get_variable(ini_date_num).fillmask()
        ecco.interp2romsgrd()
        
        if '3D' in ecco.dimtype:
            
            for j in np.arange(romsgrd.maskr().shape[0]):
            
                if j / 100. == np.fix(j / 100.):
                    print '------ j = %s of %s' %(j+1, romsgrd.maskr().shape[0])
                
                ecco.set_map_coordinate_weights(j=j)
                ecco.vert_interp(j=j)
        
        # Calculate barotropic velocities
        if ecco.vartype in ('U', 'V'):
            ecco.set_barotropic()
        
        # Write to initial file
        print 'Saving *%s* to %s' %(ecco.vartype, romsini.romsfile)
        with netcdf.Dataset(romsini.romsfile, 'a') as nc:
                    
            if ecco.vartype in 'U':
                
                u = np.copy(ecco.dataout)
                ubar = np.copy(ecco.barotropic)
            
            elif ecco.vartype in 'V':
                
                ubar, vbar = romsgrd.rotate(ubar, ecco.barotropic, sign=1)
                ubar = romsgrd.rho2u_2d(ubar)
                ubar *= romsgrd.umask()
                nc.variables['ubar'][:] = ubar[np.newaxis]
                del ubar
                vbar = romsgrd.rho2v_2d(vbar)
                vbar *= romsgrd.vmask()
                nc.variables['vbar'][:] = vbar[np.newaxis]
                del vbar
                
                for k in np.arange(romsgrd.N.size).astype(np.int):
                    utmp, vtmp = u[k], ecco.dataout[k]
                    u[k], ecco.dataout[k] = romsgrd.rotate(utmp, vtmp, sign=1)
                
                u = romsgrd.rho2u_3d(u)
                u *= romsgrd.umask3d()
                nc.variables['u'][:] = u[np.newaxis]
                del u
                v = romsgrd.rho2v_3d(ecco.dataout)
                v *= romsgrd.vmask3d()
                nc.variables['v'][:] = v[np.newaxis]
                del v
                
            
            elif ecco.vartype in 'SSH':
                
                ecco.dataout *= romsgrd.maskr()#.ravel()
                nc.variables['zeta'][:] = ecco.dataout[np.newaxis]
    
                nc.variables['ocean_time'][:] = ini_time
                nc.variables['tstart'][:] = tstart
                nc.variables['tend'][:] = tend
                nc.ecco_start_date = ini_date
                nc.ecco_day_zero = day_zero
                
            else:    
                varname = ecco.vartype.lower()
                ecco.dataout *= romsgrd.mask3d()
                nc.variables[varname][:] = ecco.dataout[np.newaxis]
    
    print 'all done'
