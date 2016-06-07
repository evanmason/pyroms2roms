# -*- coding: utf-8 -*-
# %run py_mercator2ini.py

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

Create a ROMS initial file based on Mercator data

===========================================================================
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np
import numexpr as ne
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
from time import strptime
#import copy

from py_roms2roms import vertInterp, horizInterp, bry_flux_corr, debug0, debug1, debug2
#from py_roms2roms import ROMS, RomsGrid, RomsData
from py_mercator2roms import RomsGrid, MercatorData, prepare_romsgrd
from pyroms2ini import RomsData


class MercatorDataIni (MercatorData):
    '''
    MercatorDataIni class (inherits from MercatorData class)
    '''
    def __init__(self, filenames, model_type, mercator_var, romsgrd, **kwargs):
        """
        Creates a new Mercator ini data object.
        
        Parameters
        ----------
        
        *filenames* : list of Mercator nc files.
        *model_type* : string specifying Mercator model.
        *romsgrd* : a `RomsGrid` instance.
        
        """
        super(MercatorDataIni, self).__init__(filenames, model_type, mercator_var, romsgrd, **kwargs)
        
        
    def interp2romsgrd(self):
        '''
        Modification of MercatorData class definition *interp2romsgrd*
        '''
        if '3D' in self.dimtype:
            for k in np.arange(self._depths.size):
                self._interp2romsgrd(k)
        else:
            self._interp2romsgrd()
        return self


    def vert_interp(self, j):
        '''
        Modification of MercatorData class definition *vert_interp*
        '''
        vinterp = vertInterp(self.mapcoord_weights)
        self.dataout[:, j] = vinterp.vert_interp(self.datatmp[::-1, j])
        return self
      




def prepare_mercator(mercator, balldist):
    mercator.proj2gnom(ignore_land_points=False, M=mercator.romsgrd.M)
    mercator.child_contained_by_parent(mercator.romsgrd)
    mercator.make_kdetree().get_fillmask_cofs()
    ballpoints = mercator.kdetree.query_ball_tree(romsgrd.kdetree, r=balldist)
    #print 'dd', print(ballpoints)
    try:
        mercator.ball = np.array(ballpoints)
    except:
        print ballpoints, aaaaaaaa
    #print 'ee'
    #mercator.ball = np.array(ballpoints).nonzero()[0]
    mercator.ball = mercator.ball.nonzero()
    #print 'ff'
    mercator.ball = mercator.ball[0]
    #print 'gg'
    mercator.tri = sp.Delaunay(mercator.points[mercator.ball])
    if '3D' in mercator.dimtype:
        #print 'gg'
        mercator.set_3d_depths()
        #print 'hh'
    return mercator
    
    
    
if __name__ == '__main__':
    
    '''
    py_mercator2ini
    

    
    Evan Mason 2014
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # Mercator information
    #mercator_dir = '/marula/emason/data/mercator/nea_daily/2010/'
    #mercator_dir = '/marula/emason/data/mercator/nwmed/ORCA12/'
    mercator_dir = '/marula/emason/data/IBI_daily/'
    #mercator_dir = '/marula/emason/data/mercator/albsea/'
    
    # ROMS path and filename information
    #roms_dir     = '../'
    #roms_dir     = '/marula/emason/runs2012/MedSea5/'
    #roms_dir     = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir     = '/marula/emason/runs2009/na_2009_7pt5km/'
    #roms_dir     = '/marula/emason/runs2014/MedCan5/'
    #roms_dir     = '/Users/emason/runs/runs2014/MedCan5/'
    #roms_dir = '/Users/emason/runs2009/na_2009_7pt5/'
    #roms_dir     = '/marula/emason/runs2014/NWMED2/'
    #roms_dir     = '/marula/emason/runs2014/AlbSea175/'
    roms_dir     = '/marula/emason/runs2015/AlbSea500/'
    #roms_dir = '/marula/emason/runs2014/nwmed5km/'
    
    
    #roms_grd     = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd     = 'grd_MedCan5.nc'
    #roms_grd = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd = 'grd_nwmed_2km.nc'
    #roms_grd = 'grd_AlbSea175.nc'
    #roms_grd = 'grd_nwmed5km_NARROW_STRAIT.nc'
    roms_grd = 'grd_AlbSea500.nc'
    
    # Set sigma transformation parameters
    if 'roms_grd_NA2009_7pt5km.nc' in roms_grd:
        chd_sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
        
    elif 'grd_nwmed_2km.nc' in roms_grd:
        chd_sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
        ini_filename = 'ini_nwmed_2km_N40.nc'
        ini_date   = '20051230'
        
    elif 'grd_AlbSea175.nc' in roms_grd:
        chd_sigma_params = dict(theta_s=7., theta_b=0.25, hc=90., N=32)
        #ini_filename = 'ini_AlbSea175_BEFORE.nc'
        ini_filename = 'ini_AlbSea175_TEST-constant.nc'
        ini_date = '20131231'
        
    elif 'grd_AlbSea500.nc' in roms_grd:
        chd_sigma_params = dict(theta_s=6., theta_b=0.25, hc=100., N=32)
        #ini_filename = 'ini_AlbSea175_BEFORE.nc'
        ini_filename = 'ini_AlbSea500_NOPREFILTER.nc'
        ini_date = '20131231'
    
    elif 'grd_nwmed5km_NARROW_STRAIT.nc' in roms_grd:
        chd_sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
        ini_filename = 'ini_test_constant.nc'
        ini_date = '20131231'
    
    else:
        print 'No sigma params defined for grid: %s' %roms_grd
        raise Exception
    
    ini_time = 0.  # days
    tstart = 0.
    tend = 0. 

    if 'ORCA' in mercator_dir:
        mercator_ssh_file = mercator_dir + '*y%sm%sd%s_gridT*.nc' %(ini_date[:4], ini_date[4:6], ini_date[6:])
        #mercator_ssh_file = mercator_dir + '*y%sm%sd%s_grid2D*.nc' %(ini_date[:4], ini_date[4:6], ini_date[6:])
        mercator_temp_file = mercator_dir + '*y%sm%sd%s_gridT*.nc' %(ini_date[:4], ini_date[4:6], ini_date[6:])
        mercator_salt_file = mercator_dir + '*y%sm%sd%s_gridS*.nc' %(ini_date[:4], ini_date[4:6], ini_date[6:])
        mercator_u_file = mercator_dir + '*y%sm%sd%s_gridU*.nc' %(ini_date[:4], ini_date[4:6], ini_date[6:])
        mercator_v_file = mercator_dir + '*y%sm%sd%s_gridV*.nc' %(ini_date[:4], ini_date[4:6], ini_date[6:])
    
    elif 'PSY2V4R4' in mercator_dir:
        to_be_done
        
    elif 'IBI_daily' in mercator_dir:
        mercator_ssh_file = 'pde_ibi36v3r1_ibisr_01dav_%s_%s_R????????_HC01.nc' %(ini_date, ini_date)
        mercator_temp_file = mercator_dir + mercator_ssh_file
        mercator_salt_file = mercator_dir + mercator_ssh_file
        mercator_u_file = mercator_dir + mercator_ssh_file
        mercator_v_file = mercator_dir + mercator_ssh_file
        mercator_ssh_file = mercator_dir + mercator_ssh_file
    
    elif 'albsea' in mercator_dir:
        mercator_ssh_file = 'alb_sea.nc'
        mercator_temp_file = mercator_dir + mercator_ssh_file
        mercator_salt_file = mercator_dir + mercator_ssh_file
        mercator_u_file = mercator_dir + mercator_ssh_file
        mercator_v_file = mercator_dir + mercator_ssh_file
        mercator_ssh_file = mercator_dir + mercator_ssh_file
        
        
    else: Exception
    
    balldist = 40000. # meters

    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    fillval = 9999
  
    mercator_ssh_file = glob.glob(mercator_ssh_file)
    mercator_temp_file = glob.glob(mercator_temp_file)
    mercator_salt_file = glob.glob(mercator_salt_file)
    mercator_u_file = glob.glob(mercator_u_file)
    mercator_v_file = glob.glob(mercator_v_file)
    
    ini_filename = ini_filename.replace('.nc', '_%s.nc' %ini_date)

    day_zero = datetime(int(ini_date[:4]), int(ini_date[4:6]), int(ini_date[6:]))
    day_zero = plt.date2num(day_zero) + 0.5

    
    # Set up a RomsGrid object
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), chd_sigma_params, model_type='ROMS')
    #romsgrd_u = copy.deepcopy(romsgrd)
    #romsgrd_v = copy.copy(romsgrd)
    
    # Set up a RomsData object for the initial file
    romsini = RomsData(roms_dir + ini_filename, model_type='ROMS')
    
    # Create the initial file
    romsini.create_ini_nc(romsgrd, fillval)

    
    mercator_vars = OrderedDict([('SSH', mercator_ssh_file),
                                 ('TEMP', mercator_temp_file),
                                 ('SALT', mercator_salt_file),
                                 ('U', mercator_u_file),
                                 ('V', mercator_v_file)])
    
    
    
    # Set partial domain chunk sizes
    ndomx = 5
    ndomy = 4
    
    Mp, Lp = romsgrd.h().shape
    szx = np.floor(Lp / ndomx).astype(int)
    szy = np.floor(Mp / ndomy).astype(int)
    
    icmin = np.arange(0, ndomx) * szx
    icmax = np.arange(1, ndomx + 1) * szx
    jcmin = np.arange(0, ndomy) * szy
    jcmax = np.arange(1, ndomy + 1) * szy
    
    icmin[0] = 0
    icmax[-1] = Lp + 1
    jcmin[0] = 0
    jcmax[-1] = Mp + 1
    
    
    for domx in np.arange(ndomx):
        
        for domy in np.arange(ndomy):
    
            romsgrd.i0 = icmin[domx]
            I1 = romsgrd.i1 = icmax[domx]
            romsgrd.j0 = jcmin[domy]
            J1 = romsgrd.j1 = jcmax[domy]
            
            i0, i1, j0, j1 = romsgrd.i0, romsgrd.i1, romsgrd.j0, romsgrd.j1
            
            Mp, Lp = romsgrd.h().shape
            
            
            
            for mercator_var, mercator_file in zip(mercator_vars.keys(), mercator_vars.values()):
                
                try:
                    del mercator
                except:
                    pass
                
                print '\nProcessing variable *%s*' %mercator_var
                
                if mercator_var in 'U':
                    romsgrd.i1 += 1
                    i1 += 1
                    romsgrd.j1 += 1
                    j1 += 1
                
                elif mercator_var in 'V':
                    pass
                
                else:
                    romsgrd.i1 = i1 = I1
                    romsgrd.j1 = j1 = J1
                
                romsgrd = prepare_romsgrd(romsgrd)
                mercator = MercatorDataIni(mercator_file, 'Mercator', mercator_var, romsgrd)
                mercator = prepare_mercator(mercator, balldist)
        
        
                if 0: # debug
                    plt.pcolormesh(mercator.lon(), mercator.lat(),mercator.maskr())
                    plt.scatter(mercator.lon().flat[mercator.ball], mercator.lat().flat[mercator.ball],
                                s=10,c='g',edgecolors='none')
                    plt.plot(romsgrd.boundary()[0], romsgrd.boundary()[1], 'w')
                    plt.axis('image')
                    plt.show()
                    
        
            



        
                # Read in variables and interpolate to romsgrd
                mercator.get_variable(day_zero).fillmask()
                mercator.interp2romsgrd()
        
        
                if '3D' in mercator.dimtype:
                    #aaaaa
                    for j in np.arange(romsgrd.maskr().shape[0]):
            
                        if j / 100. == np.fix(j / 100.):
                            print '------ j = %s of %s' %(j + 1, romsgrd.maskr().shape[0])
                
                        mercator.set_map_coordinate_weights(j=j)
                        mercator.vert_interp(j=j)
                
                
                # Calculate barotropic velocities
                if mercator.vartype in ('U', 'V'):
                    mercator.set_barotropic()
                
                # Write to initial file
                print 'Saving *%s* to %s' %(mercator.vartype, romsini.romsfile)
                with netcdf.Dataset(romsini.romsfile, 'a') as nc:
                    
                    if mercator.vartype in 'U':
                
                        u = mercator.dataout.copy()
                        ubar = mercator.barotropic.copy()
            
                    elif mercator.vartype in 'V':
                
                        ubar, vbar = romsgrd.rotate(ubar, mercator.barotropic, sign=1)
                        ubar = romsgrd.rho2u_2d(ubar)
                        ubar *= romsgrd.umask()
                        nc.variables['ubar'][:, j0:j1, i0:i1 - 1] = ubar[np.newaxis]
                        del ubar
                        vbar = romsgrd.rho2v_2d(vbar)
                        vbar *= romsgrd.vmask()
                        nc.variables['vbar'][:, j0:j1 - 1, i0:i1] = vbar[np.newaxis]
                        del vbar
                
                        for k in np.arange(romsgrd.N.size).astype(np.int):
                            utmp, vtmp = u[k], mercator.dataout[k]
                            u[k], mercator.dataout[k] = romsgrd.rotate(utmp, vtmp, sign=1)
                
                        u = romsgrd.rho2u_3d(u)
                        u *= romsgrd.umask3d()
                        nc.variables['u'][:, :, j0:j1, i0:i1 - 1] = u[np.newaxis]
                        del u
                        v = romsgrd.rho2v_3d(mercator.dataout)
                        v *= romsgrd.vmask3d()
                        nc.variables['v'][:, :, j0:j1 - 1, i0:i1] = v[np.newaxis]
                        del v
                
            
                    elif mercator.vartype in 'SSH':
                
                
                        mercator.dataout = mercator.dataout.reshape(Mp, Lp)[np.newaxis]
                        mercator.dataout *= romsgrd.maskr()
                        
                        nc.variables['zeta'][:, j0:j1, i0:i1] = mercator.dataout
    
                        nc.variables['ocean_time'][:] = ini_time
                        nc.variables['tstart'][:] = tstart
                        nc.variables['tend'][:] = tend
                        nc.mercator_start_date = ini_date
            
                    elif mercator.vartype in 'TEMP' and mercator.dataout.max() > 100.:
                
                        mercator.dataout -= mercator.Kelvin
                        mercator.dataout *= romsgrd.mask3d()
                        nc.variables['temp'][:, :, j0:j1, i0:i1] = mercator.dataout[np.newaxis]
            
                    else:    
                
                        varname = mercator.vartype.lower()
                        mercator.dataout *= romsgrd.mask3d()
                        nc.variables[varname][:, :, j0:j1, i0:i1] = mercator.dataout[np.newaxis]
    
    print 'all done'
            
            
      














