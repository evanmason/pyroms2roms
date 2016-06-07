# -*- coding: utf-8 -*-
# %run pyroms2ini.py

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

Create a ROMS initial file based on ROMS data

===========================================================================
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
#import matplotlib.nxutils as nx
import time
import scipy.interpolate.interpnd as interpnd
#import collections
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime

from py_roms2roms import vertInterp, horizInterp
from py_roms2roms import ROMS, RomsGrid, RomsData



class RomsData(RomsData):
    '''
    Modify the RomsGrid class
    '''
    
    def create_ini_nc(self, grdobj, fillval, created_by='py_roms2ini.py'):
        '''
        Create a new initial file based on dimensions from
        grdobj
        NOTE the redundancy of some variables (hc, theta_s, theta_b, Tcline, sc_r, sc_w...),
             saved both as global attributes and as data variables.  This is to accomodate
             the requirements of different versions of the UCLA code that seem to be
             in existence.  Sasha is very clear on the use of global variables, see the
             bottom of his last post at:
             https://www.myroms.org/forum/viewtopic.php?f=20&t=2189&p=7897&hilit=shchepet#p7897
        '''
        # Global attributes
        nc = netcdf.Dataset(self.romsfile, 'w', format='NETCDF4')
        nc.created = datetime.now().isoformat()
        nc.type  = 'ROMS initial file produced by %s' % created_by
        nc.grd_file = grdobj.romsfile
        nc.hc = grdobj.hc
        nc.theta_s = grdobj.theta_s
        nc.theta_b = grdobj.theta_b
        nc.Tcline = grdobj.hc
        nc.Cs_r = grdobj.Cs_r()
        nc.Cs_w = grdobj.Cs_w()
        nc.VertCoordType = 'NEW'
        
        # Dimensions
        nc.createDimension('xi_rho', grdobj.lon().shape[1])
        nc.createDimension('xi_u', grdobj.lon().shape[1] - 1)
        nc.createDimension('eta_rho', grdobj.lon().shape[0])
        nc.createDimension('eta_v', grdobj.lon().shape[0] - 1)
        nc.createDimension('s_rho', grdobj.N)
        nc.createDimension('s_w', grdobj.N + 1)
        nc.createDimension('time', None)
        nc.createDimension('one', 1)
        
        # Create the variables and write...
        nc.createVariable('theta_s', 'f', ('one'), zlib=True)
        nc.variables['theta_s'].long_name = 'S-coordinate surface control parameter'
        nc.variables['theta_s'].units = 'nondimensional'
        nc.variables['theta_s'][:] = grdobj.theta_s
        
        nc.createVariable('theta_b', 'f', ('one'), zlib=True)
        nc.variables['theta_b'].long_name = 'S-coordinate bottom control parameter'
        nc.variables['theta_b'].units = 'nondimensional'
        nc.variables['theta_b'][:] = grdobj.theta_b
        
        nc.createVariable('Tcline', 'f', ('one'), zlib=True)
        nc.variables['Tcline'].long_name = 'S-coordinate surface/bottom layer width'
        nc.variables['Tcline'].units = 'meters'
        nc.variables['Tcline'][:] = grdobj.hc
        
        nc.createVariable('hc', 'f', ('one'), zlib=True)
        nc.variables['hc'].long_name = 'S-coordinate parameter, critical depth'
        nc.variables['hc'].units = 'meters'
        nc.variables['hc'][:] = grdobj.hc
        
        nc.createVariable('sc_r', 'f8', ('s_rho'))
        nc.variables['sc_r'].long_name = 'S-coordinate at RHO-points'
        nc.variables['sc_r'].units = 'nondimensional'
        nc.variables['sc_r'].valid_min = -1.
        nc.variables['sc_r'].valid_max = 0.
        
        nc.createVariable('Cs_r', 'f8', ('s_rho'), zlib=True)
        nc.variables['Cs_r'].long_name = 'S-coordinate stretching curves at RHO-points'
        nc.variables['Cs_r'].units = 'nondimensional'
        nc.variables['Cs_r'].valid_min = -1.
        nc.variables['Cs_r'].valid_max = 0.
        nc.variables['Cs_r'][:] = grdobj.Cs_r()
        
        nc.createVariable('Cs_w', 'f8', ('s_w'), zlib=True)
        nc.variables['Cs_w'].long_name = 'S-coordinate stretching curves at w-points'
        nc.variables['Cs_w'].units = 'nondimensional'
        nc.variables['Cs_w'].valid_min = -1.
        nc.variables['Cs_w'].valid_max = 0.
        nc.variables['Cs_w'][:] = grdobj.Cs_w()
        
        nc.createVariable('ocean_time', 'f8', ('time'), zlib=True)
        nc.variables['ocean_time'].long_name = 'time since initialization'
        nc.variables['ocean_time'].units     = 'seconds'
        
        nc.createVariable('tstart', 'f8', ('one'), zlib=True)
        nc.variables['tstart'].long_name = 'start processing day'
        nc.variables['tstart'].units     = 'days'
        
        nc.createVariable('tend', 'f8', ('one'), zlib=True)
        nc.variables['tend'].long_name = 'end processing day'
        nc.variables['tend'].units     = 'days'
        
        
        # dictionary for the prognostic variables
        prog_vars = OrderedDict()
        prog_vars['temp'] = ['rho3d',
                             'initial potential temperature',
                             'Celsius']
        prog_vars['salt'] = ['rho3d',
                             'initial salinity',
                             'psu']
        prog_vars['u']    = ['u3d',
                             'initial u-momentum component',
                             'meters second-1']
        prog_vars['v']    = ['v3d',
                             'initial v-momentum component',
                             'meters second-1']
        prog_vars['ubar'] = ['u2d',
                             'initial vertically integrated u-momentum component',
                             'meters second-1']
        prog_vars['vbar'] = ['v2d',
                             'initial vertically integrated v-momentum component',
                             'meters second-1']
        prog_vars['zeta'] = ['rho2d',
                             'initial sea surface height',
                             'meters']
                              
        for varname, value in zip(prog_vars.keys(), prog_vars.values()):

            if 'rho3d' in value[0]:
                dims = ('time', 's_rho', 'eta_rho', 'xi_rho')

            elif 'u3d' in value[0]:
                dims = ('time', 's_rho', 'eta_rho', 'xi_u')

            elif 'v3d' in value[0]:
                dims = ('time', 's_rho', 'eta_v', 'xi_rho')

            elif 'u2d' in value[0]:
                dims = ('time', 'eta_rho', 'xi_u')

            elif 'v2d' in value[0]:
                dims = ('time', 'eta_v', 'xi_rho')

            elif 'rho2d' in value[0]:
                dims = ('time', 'eta_rho', 'xi_rho')
            
            else: error

            nc.createVariable(varname, 'f8', dims,
                              fill_value=fillval, zlib=True)
            nc.variables[varname].long_name = value[1]
            nc.variables[varname].units     = value[2]

        nc.close()





if __name__ == '__main__':
    
    '''
    pyrom2ini (Python version of r2r_ini.m written in Matlab).
    
    Differences from r2r_ini:
        ......
    
    TO DO:
      Using a KDE tree it should be possible to compute all open
       boundaries together; ie treat them as a single boundary; this
       should speed things up significantly
      Start work on extra_variables
    
    Evan Mason 2012
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    #par_dir     = '../'
    #par_dir     = '/nas02/emason/runs2009/na_2009_7pt5km/'
    par_dir     = '/marula/emason/runs2009/gc_2009_1km_60/'
    #par_grd     = 'roms_grd_NA2009_7pt5km.nc'
    par_grd     = 'gc_2009_1km_grd_smooth.nc'
    par_theta_s = 6.
    par_theta_b = 2.
    par_hc      = 120.
    par_N       = 60.
    
    #chd_dir     = '../'
    chd_dir     = '/marula/emason/runs2015/GranCan250/'
    chd_grd     = 'frc_gc250_coast.nc'
    chd_theta_s = 6.
    chd_theta_b = 2.
    chd_hc      = 120.
    chd_N       = 60.
    
    
    # inital file
    ini_time    =  0     # days, 0 means no cycle
    ini_filename = 'ini_test.nc' # bry filename
    ini_type     = 'MM5-3/roms_rst.Y3M06' # parent file to read data from,
                              # usually one of 'roms_avg', 'roms_his' or 'roms_rst'
    par_file   = '0000' # an avg/his/rst file
    par_rec    =  1     # desired record no. from avg/his/rst file

    
    


    # dictionary with any additional variables and their dimensions
    # that we might want to process, these might be sediments, bio, etc...
    extra_variables = OrderedDict()


    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
