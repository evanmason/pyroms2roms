# -*- coding: utf-8 -*-
# %run pycfsr2frc.py

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

Create a ROMS forcing file based on CFSR monthly data

===========================================================================
'''

import netCDF4 as netcdf
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import numpy as np
import numexpr as ne
from scipy import io
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
#import matplotlib.nxutils as nx
import time
import scipy.interpolate.interpnd as interpnd
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
#from datetime import datetime
import calendar as ca

from py_roms2roms import horizInterp
from py_roms2roms import ROMS, debug0, debug1
from py_roms2roms import RomsGrid, RomsData



class RomsGrid(RomsGrid):
    '''
    Modify the RomsGrid class
    '''
    
    def create_frc_nc(self, frcfile, sd, ed, nr, cl, madeby, bulk):
        '''
        Create a new forcing file based on dimensions from grdobj
            frcfile : path and name of new frc file
            sd      : start date2num
            ed      : end date
            nr      : no. of records
            cl      : cycle length
            madeby  : name of this file
        '''
        if bulk:
            try:
                frcfile = frcfile.replace('frc_', 'blk_')
            except Exception:
                pass
        else:
            try:
                frcfile = frcfile.replace('blk_', 'frc_')
            except Exception:
                pass
                
        self.frcfile = frcfile
        print 'Creating new CFSR forcing file:', frcfile
        # Global attributes
        ''' The best choice should be format='NETCDF4', but it will not work with
        Sasha's 2008 code (not tested with Roms-Agrif).  Therefore I use
        format='NETCDF3_64BIT; the drawback is that it is very slow'
        '''
        #nc = netcdf.Dataset(frcfile, 'w', format='NETCDF3_64BIT')
        #nc = netcdf.Dataset(frcfile, 'w', format='NETCDF3_CLASSIC')
        nc = netcdf.Dataset(frcfile, 'w', format='NETCDF4')
        nc.created  = dt.datetime.datetime.now().isoformat()
        nc.type = 'ROMS interannual forcing file produced by %s.py' %madeby
        nc.grd_file = self.romsfile
        nc.start_date = sd
        nc.end_date = ed
        
        # Dimensions
        nc.createDimension('xi_rho', self.lon().shape[1])
        nc.createDimension('xi_u', self.lon().shape[1] - 1)
        nc.createDimension('eta_rho', self.lon().shape[0])
        nc.createDimension('eta_v', self.lon().shape[0] - 1)
        if bulk:
            nc.createDimension('bulk_time', nr)
        else:
            nc.createDimension('sms_time', nr)
            nc.createDimension('shf_time', nr)
            nc.createDimension('swf_time', nr)
            nc.createDimension('sst_time', nr)
            nc.createDimension('srf_time', nr)
            nc.createDimension('sss_time', nr)
            nc.createDimension('one', 1)
        

        # Dictionary for the variables
        frc_vars = OrderedDict()
        
        if bulk:
            frc_vars['bulk_time'] = ['time',
                                     'bulk_time',
                                     'bulk formulation execution time',
                                     'days']
        else:
            frc_vars['sms_time'] = ['time',
                                    'sms_time',
                                    'surface momentum stress time',
                                    'days']
            frc_vars['shf_time'] = ['time',
                                    'shf_time',
                                    'surface heat flux time',
                                    'days']
            frc_vars['swf_time'] = ['time',
                                    'swf_time',
                                    'surface freshwater flux time',
                                    'days']
            frc_vars['sst_time'] = ['time',
                                    'sst_time',
                                    'sea surface temperature time',
                                    'days']
            frc_vars['sss_time'] = ['time',
                                    'sss_time',
                                    'sea surface salinity time',
                                    'days']
            frc_vars['srf_time'] = ['time',
                                    'srf_time',
                                    'solar shortwave radiation time',
                                    'days']
        
            # Note this could be problematic if scale_cfsr_coads.py adjusts variables
            # with different frequencies...
            frc_vars['month']    = ['time',
                                    'sst_time',
                                    'not used by ROMS; useful for scale_cfsr_coads.py',
                                    'month of year']
        
        # Bulk formaulation
        if bulk:
            frc_vars['tair'] =   ['rho',
                                  'bulk_time',
                                  'surface air temperature',
                                  'Celsius']
            frc_vars['rhum'] =   ['rho',
                                  'bulk_time',
                                  'relative humidity',
                                  'fraction']
            frc_vars['prate'] =   ['rho',
                                  'bulk_time',
                                  'precipitation rate',
                                  'cm day-1']
            frc_vars['wspd'] =   ['rho',
                                  'bulk_time',
                                  'wind speed 10m',
                                  'm s-1']
            frc_vars['radlw'] =   ['rho',
                                   'bulk_time',
                                   'net outgoing longwave radiation',
                                   'Watts meter-2',
                                   'upward flux, cooling water']
            frc_vars['radlw_in'] = ['rho',
                                    'bulk_time',
                                    'downward longwave radiation',
                                    'Watts meter-2',
                                    'downward flux, warming water']
            frc_vars['radsw'] =    ['rho',
                                    'bulk_time',
                                    'shortwave radiation',
                                    'Watts meter-2',
                                    'downward flux, warming water']
            frc_vars['sustr'] =     ['u',
                                    'bulk_time',
                                    'surface u-momentum stress',
                                    'Newton meter-2']
            frc_vars['svstr'] =     ['v',
                                    'bulk_time',
                                    'surface v-momentum stress',
                                    'Newton meter-2']
            frc_vars['uwnd'] =     ['u',
                                    'bulk_time',
                                    '10m u-wind',
                                    'm s-1']
            frc_vars['vwnd'] =     ['v',
                                    'bulk_time',
                                    '10m v-wind',
                                    'm s-1']
            
        # dQdSST
        else:            
            frc_vars['sustr'] =  ['u',
                                  'sms_time',
                                  'surface u-momentum stress',
                                  'Newton meter-2']
            frc_vars['svstr'] =  ['v',
                                  'sms_time',
                                  'surface v-momentum stress',
                                  'Newton meter-2']	  
            frc_vars['shflux'] = ['rho',
                                  'shf_time',
                                  'surface net heat flux',
                                  'Watts meter-2']
            frc_vars['swflux'] = ['rho',
                                  'swf_time',
                                  'surface freshwater flux (E-P)',
                                  'centimeters day-1',
                                  'net evaporation',
                                  'net precipitation']
            frc_vars['SST'] =    ['rho',
                                  'sst_time',
                                  'sea surface temperature',
                                  'Celsius']
            frc_vars['SSS'] =    ['rho',
                                  'sss_time',
                                  'sea surface salinity',
                                  'PSU']
            frc_vars['dQdSST'] = ['rho',
                                  'sst_time',
                                  'surface net heat flux sensitivity to SST',
                                  'Watts meter-2 Celsius-1']
            frc_vars['swrad'] =  ['rho',
                                  'srf_time',
                                  'solar shortwave radiation',
                                  'Watts meter-2',
                                  'downward flux, heating',
                                  'upward flux, cooling']
        
        
        for key, value in zip(frc_vars.keys(), frc_vars.values()):

            #print key, value

            if 'time' in value[0]:
                dims = (value[1])

            elif 'rho' in value[0]:
                dims = (value[1], 'eta_rho', 'xi_rho')

            elif 'u' in value[0]:
                dims = (value[1], 'eta_rho', 'xi_u')
                    
            elif 'v' in value[0]:
                dims = (value[1], 'eta_v', 'xi_rho')
                
            else:
                error
            
            #print 'key, dims = ',key, dims
            nc.createVariable(key, 'f4', dims, zlib=True)
            nc.variables[key].long_name = value[2]
            nc.variables[key].units = value[3]
            
            if 'time' in key and nr is not None:
                nc.variables[key].cycle_length = cl
            
            if 'swrad' in key:
                nc.variables[key].positive = value[4]
                nc.variables[key].negative = value[5]
                
            if 'swflux' in key:
                nc.variables[key].positive = value[4]
                nc.variables[key].negative = value[5]
                
            if 'radlw' in key:
                nc.variables[key].positive = value[4]
                
            if 'radlw_in' in key:
                nc.variables[key].positive = value[4]
                
            if 'radsw' in key:
                nc.variables[key].positive = value[4]

        nc.close()


    def gc_dist(self, lon1, lat1, lon2, lat2):
        '''
        Use Haversine formula to calculate distance
        between one point and another
        !! lat and lon in radians !!
        '''
        r_earth = self.r_earth # Mean earth radius in metres (from scalars.h)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        dang = 2. * np.arcsin(np.sqrt(np.power(np.sin(0.5 * dlat), 2) + \
                    np.cos(lat2) * np.cos(lat1) * np.power(np.sin(0.5 * dlon), 2)))
        return r_earth * dang # distance
            
            
            



    #def get_runoff(self, swflux_data, dai_file, mon):
    def get_runoff(self, dai_file, mon):
        '''
        This needs to be speeded up.  Separate into 2 parts:
          1 def runoff_setup() to do one-time tasks
          2 get_runoff() to do variable tasks
        '''
        mon -= 1
        area = 1.
        area /= (self.pm() * self.pn())
        dx = 1.
        dx /= np.mean(self.pm())
        
        roms_dir = self.romsfile.replace(self.romsfile.split('/')[-1], '')
        cdist = io.loadmat(roms_dir + 'coast_distances.mat')
        mask = self.maskr()
        
        # Exponential decay, runoff forced towards coast.  75km ?
        #mult = np.exp(-cdist['cdist'] / 150.e4)
        mult = np.exp(-cdist['cdist'] / 60.e4)
        np.place(mult, mask > 1., 0)

        # Read in river data and set data trim
        lon0 = np.round(self.lon().min() - 1.0)
        lon1 = np.round(self.lon().max() + 1.0)
        lat0 = np.round(self.lat().min() - 1.0)
        lat1 = np.round(self.lat().max() + 1.0)

        nc = netcdf.Dataset(dai_file)
        lon_runoff = nc.variables['lon'][:]
        lat_runoff = nc.variables['lat'][:]

        i0 = np.nonzero(lon_runoff > lon0)[0].min()
        i1 = np.nonzero(lon_runoff < lon1)[0].max() + 1
        j0 = np.nonzero(lat_runoff > lat0)[0].min()
        j1 = np.nonzero(lat_runoff < lat1)[0].max() + 1

        flux_unit = 1.1574e-07 # (cm/day)

        surf_ro = np.zeros(self.lon().shape)

        lon_runoff = lon_runoff[i0:i1]
        lat_runoff = lat_runoff[j0:j1]

        runoff = nc.variables['runoff'][mon, j0:j1, i0:i1]
        nc.close()
        
        lon_runoff, lat_runoff = np.meshgrid(lon_runoff, lat_runoff)
        lon_runoff = lon_runoff[runoff >= 1e-5]
        lat_runoff = lat_runoff[runoff >= 1e-5]
        runoff = runoff[runoff >= 1e-5]
        n_runoff = runoff.size
   
        for idx in np.arange(n_runoff):
      
            #print idx, n_runoff

            # Find suitable unmasked grid points to distribute run-off
            dist = self.gc_dist(np.deg2rad(self.lon()),
                                np.deg2rad(self.lat()),
                                np.deg2rad(lon_runoff[idx]),
                                np.deg2rad(lat_runoff[idx]))
        
            if dist.min() <= 5. * dx:
                #scale = 150. * 1e3 # scale of Dai data is at 1 degree
                scale = 60. * 1e3 # scale of Dai data is at 1 degree
                bool_mask = (dist < scale) * (mask > 0)
                int_are = area[bool_mask].sum()
                ave_wgt = mult[bool_mask].mean()

                surface_flux = 1e6 * runoff[idx]
                #print 'surface_flux, int_are', surface_flux, int_are
                surface_flux /= np.array([int_are, np.spacing(2)]).max()
                #print 'surface_flux',surface_flux
                surf_ro[bool_mask] = (surf_ro[bool_mask] + surface_flux /
                                      flux_unit * mult[bool_mask] / ave_wgt)

        #return (swflux_data * mask) - (surf_ro * mask) comment Sep 2013
        return surf_ro * mask



    def get_runoff_index_weights(self, dtnum):
        '''
        Get indices and weights for Dai river climatology
        Input: dt - datenum for current month
        Output: dai_ind_min
                dai_ind_max
                weights
        '''
        dtdate = dt.num2date(dtnum)
        day_plus_hr = np.float(dtdate.day + dtdate.hour / 24.)
        x_monrange = ca.monthrange(dtdate.year, dtdate.month)[-1]
        x_half_monrange = 0.5 * x_monrange
        if dtdate.day > x_half_monrange:
            # Second half of month, use present and next months
            dai_ind_min = dtdate.month - 1
            dai_ind_max = dtdate.month
            if dai_ind_max == 12:
                dai_ind_max = 0
                y_monrange = ca.monthrange(dtdate.year + 1, 1)[-1]
            else:
                y_monrange = ca.monthrange(dtdate.year, dtdate.month)[-1]
            x = day_plus_hr - x_half_monrange            
            y = 0.5 * y_monrange
            y += x_monrange
            y -= day_plus_hr
            w1, w2 = y, x
        else:
            # First half of month, use previous and present months
            dai_ind_min = dtdate.month - 2
            dai_ind_max = dtdate.month - 1
            if dai_ind_min < 0:
                dai_ind_min = 11
                y_monrange = ca.monthrange(dtdate.year - 1, 12)[-1]
            else:
                y_monrange = ca.monthrange(dtdate.year, dtdate.month - 1)[-1]
            x = x_half_monrange - day_plus_hr
            y = 0.5 * y_monrange
            y += day_plus_hr
            w1, w2 = x, y
        xpy = x + y
        weights = [w1 / xpy, w2 / xpy]
        return dai_ind_min, dai_ind_max, weights







class CfsrGrid(ROMS):
    '''
    CFSR grid class (inherits from RomsGrid class)
    '''
    def __init__(self, filename, model_type):
        '''
        
        '''
        super(CfsrGrid, self).__init__(filename, model_type)
        print 'Initialising CfsrGrid', filename
        self._lon = self.read_nc('lon', indices='[:]')
        self._lat = self.read_nc('lat', indices='[:]')
        self._lon, self._lat = np.meshgrid(self._lon,
                                           self._lat[::-1])


        
        
    def lon(self):
        return self._lon
    
    def lat(self):
        return self._lat
    
    
    
    
    def metrics(self):
        '''
        Return array of metrics unique to this grid
          (lonmin, lonmax, lon_res, latmin, latmax, lat_res)
          where 'res' is resolution in degrees
        '''
        self_shape = self.lon().shape
        lon_range = self.read_nc_att('lon', 'valid_range')
        lon_mean = self.lon().mean().round(2)
        lat_mean = self.lat().mean().round(2)
        res = np.diff(lon_range) / self.lon().shape[1]
        met = np.hstack((self_shape[0], self_shape[1], lon_mean, lat_mean, res))
        return met
    
            
    
    #def get_mask(self):
        #'''
        #Return a land sea mask
        #'''
        #try:
            #result = self.read_nc('LAND_L1', '[0,::-1]')
        #except Exception:
            #try:
                #result = self.read_nc('LAND_L1_Avg', '[0,::-1]')
            #except Exception:
                #try:
                    #result = self.read_nc('SALTY_L160', '[0,::-1]')
                #except Exception:
                    #raise
        #return np.abs(result - 1.)
    
    
    #def select_mask(self, masks):
        #'''
        #Loop over list of masks to find which one
        #has same grid dimensions as self
        #'''
        #for each_ind, each_mask in enumerate(masks):
            #if np.alltrue(each_mask.metrics() == self.metrics()):
                #self.mask = each_mask.get_mask()
        #try:
            #self.mask
        #except Exception:
            ##print 'No suitable mask found masks'
            #raise
            ###to_do__make_routine_to_add_new_grid_on_the_fly
        #else:
            #return self
        
    def resolution(self):
        '''
        Return the resolution of the data in degrees
        '''
        return self.metrics()[-1]
    
    
    #def get_points(self, roms_M):
        #'''
        #get points
        #'''
        #return self.proj2gnom(roms_M)
                        
        


    #def get_points_kdetree(self, roms_points, roms_M):
        #'''
        #Check that no ROMS data points lie outside of the
        #CFSR domain. If this occurs pycfsr2frc cannot function;
        #the solution is to obtain a new CFSR file making sure that it
        #covers the child grid...
        #'''
        #cfsr_points_all = self.get_points(roms_M) # must use roms_M
        
        #cfsr_tri = sp.Delaunay(cfsr_points_all) # triangulate full parent 
        #tn = cfsr_tri.find_simplex(roms_points)
        #assert not np.any(tn == -1), 'ROMS data points outside CFSR domain detected'
        
        ## Create cfsr grid KDE tree...
        #cfsr_tree = sp.KDTree(cfsr_points_all)
        ## ... in order to get minimum no. indices for interpolation.
        
        #return cfsr_tree, cfsr_points_all



    def assert_resolution(self, cfsrgrd_tmp, key):
        '''
        '''
        assert self.metrics()[2] <= cfsrgrd_tmp.metrics()[2], \
            'Resolution of %s is lower than previous grids, move up in dict "cfsr_files"' %key





class CfsrData(RomsData):
    '''
    CFSR data class (inherits from RomsData class)
    '''
    def __init__(self, filename, varname, model_type, romsgrd, masks=None):
        '''
        
        '''
        super(CfsrData, self).__init__(filename, model_type)
        
        #print 'bbbbbbbbbbbbb'
        #classname = str(self.__class__).partition('.')[-1].strip("'>")
        #print '\n--- Instantiating *%s* instance from' %classname, filename
        self.varname = varname
        self._check_product_description()
        self.needs_time_averaging = False
        if 'mask' not in (str(type(self)).lower()):
            self._set_averaging_weights()
        self._set_start_end_dates('ref_date_time')
        self._lon = self.read_nc('lon', indices='[:]')
        self._lat = self.read_nc('lat', indices='[:]')
        self._lon, self._lat = np.meshgrid(self._lon,
                                           self._lat[::-1])
        self._get_metrics()
        if masks is not None:
            self._select_mask(masks)
            self._maskr = self.cfsrmsk._maskr
            self.fillmask_cof = self.cfsrmsk.fillmask_cof
        else:
            self.cfsrmsk = None
            self.fillmask_cof = None
        self.romsgrd = romsgrd
        self.datain = np.ma.empty(self.lon().shape)
        self.datatmp = np.ma.empty(self.lon().shape)
        self.dataout = np.ma.empty(self.romsgrd.lon().size)
        self._datenum = self._get_time_series()
        self.tind = None
        self.tri = None
        self.tri_all = None
        self.dt = None
        self.to_be_downscaled = False
        self.needs_all_point_tri = False
    
    def lon(self):
        return self._lon
    
    def lat(self):
        return self._lat
    
    def maskr(self):
        ''' Returns mask on rho points
        '''
        return self._maskr
    
    

    
    '''----------------------------------------------------------------------------------
    Methods to detect if CFSR instance data are forecasts, forecast averages or
    analyses. If a forecast of either type, a second field at tind-1 must be read and
    averaged appropriately to ensure all fields are at either 00, 06, 12, 18 hours.
    '''
    def _check_product_description(self):
        ''' Returns appropriate string
        '''
        self.product = self.read_nc_att(self.varname, 'product_description')
    
    
    def _set_averaging_weights(self):
        ''' 
        Order: call after self._check_product_description()
        '''
        def _calc_weights(frcst_hr, delta_hr):
            return np.array([frcst_hr, delta_hr - frcst_hr]) / delta_hr
            
        self.forecast_hour = self.read_nc('forecast_hour', indices='[0]').astype('float')
        if 'Forecast' in self.product:
            self.needs_time_averaging = True
            self.delta_hours = np.diff(self.read_nc('time', indices='[:2]')).astype('float')
            self.time_avg_weights = _calc_weights(self.forecast_hour, self.delta_hours)
        elif 'Average (reference date/time to valid date/time)' in self.product:
            self.needs_time_averaging = True
            self.delta_hours = np.diff(self.read_nc('time', indices='[:2]')).astype('float')
            self.time_avg_weights = _calc_weights(0.5 * self.forecast_hour, self.delta_hours)
        else:
            Exception, 'Undefined_product'
        return self
        
    def _get_data_time_average(self):
        '''
        Order: call after self._set_averaging_weights()
        '''
        np.add(self.time_avg_weights[0] * self.datatmp,
               self.time_avg_weights[1] * self.datain, out=self.datain)
        return self
    
    def print_weights(self):
        '''
        '''
        print '------ averaging weights for *%s* product: %s' %(self.product, self.time_avg_weights)
        
    '''----------------------------------------------------------------------------------
    '''
    
    def _select_mask(self, masks):
        '''Loop over list of masks to find which one
           has same grid dimensions as self
        '''
        for each_mask in masks:
            if np.alltrue(each_mask.metrics == self.metrics):
                self.cfsrmsk = each_mask
                #self._maskr = each_mask._get_landmask()
                return self
        return None
    
    
    def fillmask(self):
        '''Fill missing values in an array with an average of nearest  
           neighbours
           From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        Order:  call after self.get_fillmask_cof()
        '''
        dist, iquery, igood, ibad = self.fillmask_cof
        weight = dist / (dist.min(axis=1)[:,np.newaxis] * np.ones_like(dist))
        np.place(weight, weight > 1., 0.)
        xfill = weight * self.datain[igood[:,0][iquery], igood[:,1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)
        self.datain[ibad[:,0], ibad[:,1]] = xfill
        return self
    
    
    def _set_start_end_dates(self, varname):
        '''
        '''
        #print 'varname', varname
        self._start_date = self.read_nc(varname, '[0]')
        #print 'fff',self._start_date.dtype
        #print self._start_date
        #aaaa
        try:
            self._end_date = self.read_nc(varname, '[-1]')
        except Exception:
            self._end_date = self._start_date
        return self
        
        
    def datenum(self):
        return self._datenum

    def _date2num(self, date):
        '''
        Convert CFSR 'valid_date_time' to datenum
        Input: date : ndarray (e.g., "'2' '0' '1' '0' '1' '2' '3' '1' '1' '8'")
        '''
        #print 'dateee',date, date.size
        assert (isinstance(date, np.ndarray) and
                date.size == 10), 'date must be size 10 ndarray'
        return dt.date2num(dt.datetime.datetime(np.int(date.tostring()[:4]),
                                                  np.int(date.tostring()[4:6]),
                                                  np.int(date.tostring()[6:8]),
                                                  np.int(date.tostring()[8:10])))

    def _get_time_series(self):
        '''
        '''
        #print 'self._start_date', self._start_date
        date0 = self._date2num(self._start_date)
        date1 = self._date2num(self._end_date)
        datelen = self.read_dim_size('time')
        datenum = np.linspace(date0, date1, datelen)
        return datenum
        #if np.unique(np.diff(datenum)).size == 1:
            #return datenum
        #else:
            #raise 'dsdsdsd'




    def get_delaunay_tri(self):
        '''
        '''
        self.points_all = np.copy(self.points)
        self.tri_all = sp.Delaunay(self.points_all)
        self.points = np.array([self.points[:,0].flat[self.cfsrmsk.cfsr_ball],
                                self.points[:,1].flat[self.cfsrmsk.cfsr_ball]]).T            
        self.tri = sp.Delaunay(self.points)
        return self
    
    def reshape2roms(self):
        '''
        Following interpolation with horizInterp() we need to
        include land points and reshape
        '''
        self.dataout = self.dataout.reshape(self.romsgrd.lon().shape)
        return self
    
    def _check_for_nans(self):
        '''
        '''
        flat_mask = self.romsgrd.maskr().ravel()
        assert not np.any(np.isnan(self.dataout[np.nonzero(flat_mask)])
                          ), 'Nans in self.dataout sea points'
        self.dataout[:] = np.nan_to_num(self.dataout)
        return self
    
    
    def _interp2romsgrd(self):
        '''
        '''
        ball = self.cfsrmsk.cfsr_ball
        interp = horizInterp(self.tri, self.datain.flat[ball])
        self.dataout[self.romsgrd.idata()] = interp(self.romsgrd.points)
        return self
    
    def interp2romsgrd(self, fillmask=False):
        '''
        '''
        if fillmask:
            self.fillmask()
        self._interp2romsgrd()
        self._check_for_nans()
        return self
    
    
    def check_interp(self):
        '''
        '''
        fac = 5
        cmap = plt.cm.gist_ncar
        rlonmin, rlonmax = self.romsgrd.lon().min(), self.romsgrd.lon().max()
        rlatmin, rlatmax = self.romsgrd.lat().min(), self.romsgrd.lat().max()
        cfsri0, cfsri1 = (np.argmin(np.abs(self.lon()[0] - rlonmin)) - fac,
                          np.argmin(np.abs(self.lon()[0] - rlonmax)) + fac)
        cfsrj0, cfsrj1 = (np.argmin(np.abs(self.lat()[:,0] - rlatmin)) - fac,
                          np.argmin(np.abs(self.lat()[:,0] - rlatmax)) + fac)
        cfsr = np.ma.masked_where(self.maskr() == 0, self.datain.copy())
        try:
            roms = np.ma.masked_where(self.romsgrd.maskr() == 0,
                                      self.dataout.copy())
        except Exception:
            
            roms = np.ma.masked_where(self.romsgrd.maskr() == 0,
                                      self.dataout.reshape(self.romsgrd.maskr().shape))
        cmin, cmax = roms.min(), roms.max()
        plt.figure()
        plt.pcolormesh(self.lon()[cfsrj0:cfsrj1, cfsri0:cfsri1],
                       self.lat()[cfsrj0:cfsrj1, cfsri0:cfsri1],
                             cfsr[cfsrj0:cfsrj1, cfsri0:cfsri1], cmap=cmap, edgecolors='w')
        plt.clim(cmin, cmax)
        plt.pcolormesh(self.romsgrd.lon(), self.romsgrd.lat(), roms, cmap=cmap)
        plt.clim(cmin, cmax)
        plt.axis('image')
        plt.colorbar()
        plt.show()
        
        
    def get_date_index(self, other, ind):
        '''
        '''
        return np.nonzero(self.datenum() == other.datenum()[ind])[0][0]
        


    def _get_metrics(self):
        '''Return array of metrics unique to this grid
           (lonmin, lonmax, lon_res, latmin, latmax, lat_res)
           where 'res' is resolution in degrees
        '''
        self_shape = self._lon.shape
        lon_range = self.read_nc_att('lon', 'valid_range')
        lon_mean, lat_mean = (self._lon.mean().round(2),
                              self._lat.mean().round(2))
        res = np.diff(lon_range) / self._lon.shape[1]
        self.metrics = np.hstack((self_shape[0], self_shape[1], lon_mean, lat_mean, res))
        return self





    def _read_cfsr_frc(self, var, ind):
        ''' Read CFSR forcing variable (var) at record (ind)
        '''
        return self.read_nc(var, '[' + str(ind) + ']')[::-1]
        
    
    
    def _get_cfsr_data(self, varname):
        ''' Get CFSR data with explicit variable name
        '''
        # Fill the mask
        #outvar = self.fillmask(outvar, self.outmask, flags[cfsr_key])
        return self._read_cfsr_frc(varname, self.tind)
    
    def _get_cfsr_datatmp(self):
        ''' Get CFSR data with implicit variable name
        '''
        self.datatmp[:] = self._read_cfsr_frc(self.varname, self.tind-1)
        return self
    
    
    def get_cfsr_data(self):
        ''' Get CFSR data with implicit variable name
        '''
        self.datain[:] = self._read_cfsr_frc(self.varname, self.tind)
        if self.needs_time_averaging:
            self._get_cfsr_datatmp()
            self._get_data_time_average()
        return self
    
    
    def set_date_index(self, dt):
        try:
            self.tind = np.nonzero(self.datenum() == dt)[0][0]
        except Exception:
            raise # dt out of range in CFSR file
        else:
            return self
        
    
    def vd_time(self):
        '''
        Valid date and time as YYYYMMDDHH
        '''
        return self.read_nc('valid_date_time')
        
        
    def time(self):
        '''
        Hours since 1979-01-01 00:00:00.0 +0:00
        '''
        return self.read_nc('time')
        
    

    
    
    def check_vars_for_downscaling(self, var_instances):
        '''Loop over list of grids to find which have
           dimensions different from self
        '''
        for ind, each_grid in enumerate(var_instances):
            if np.any(each_grid.metrics != self.metrics):
                each_grid.to_be_downscaled = True
        return self
    
    

    
    
    
    def dQdSST(self, sst, sat, rho_air, U, qsea):
        '''
        Compute the kinematic surface net heat flux sensitivity to the
        the sea surface temperature: dQdSST.
        Q_model ~ Q + dQdSST * (T_model - SST)
        dQdSST = - 4 * eps * stef * T^3  - rho_air * Cp * CH * U
                 - rho_air * CE * L * U * 2353 * ln (10 * q_s / T^2)
    
        Input parameters:
        sst     : sea surface temperature (Celsius)
        sat     : sea surface atmospheric temperature (Celsius)
        rho_air : atmospheric density (kilogram meter-3) 
        U       : wind speed (meter s-1)
        qsea    : sea level specific humidity (kg/kg)
 
        Output:
        dqdsst  : kinematic surface net heat flux sensitivity to the
                  the sea surface temperature (Watts meter-2 Celsius-1)
        
        From Roms_tools of Penven etal
        '''
        # SST (Kelvin)
        sst = np.copy(sst)
        sst += self.Kelvin
        # Latent heat of vaporisation (J.kg-1)
        L = np.ones(sat.shape)
        L *= self.L1
        L -= (self.L2 * sat)
        
        # Infrared contribution
        q1 = -4.
        # Multiply by Stefan constant
        q1 *= self.Stefan
        q1 *= np.power(sst, 3)
        
        # Sensible heat contribution
        q2 = -rho_air
        q2 *= self.Cp
        q2 *= self.Ch
        q2 *= U
        
        # Latent heat contribution
        dqsdt = 2353.
        dqsdt *= np.log(10.)
        dqsdt *= qsea
        dqsdt /= np.power(sst, 2)
        q3 = -rho_air
        # Multiply by Ce (Latent heat transfer coefficient, stable condition)
        q3 *= self.Ce
        q3 *= L
        q3 *= U
        q3 *= dqsdt
        dQdSST = q1
        dQdSST += q2
        dQdSST += q3
        return dQdSST




class CfsrMask(CfsrData):
    '''Mask class (inherits from CfsrData class)
    '''
    def __init__(self, cfsr_dir, mask_file, romsgrd, balldist):
        super(CfsrMask, self).__init__(cfsr_dir + mask_file[0], mask_file[1], 'CFSR', romsgrd)
        self.tind = 0
        self.get_cfsr_data()
        self.balldist = balldist
        self.romsgrd = romsgrd
        self._maskr = np.abs(self.datain - 1)
        #self.maskr()
        self.proj2gnom(ignore_land_points=False, M=romsgrd.M)
        self.make_kdetree()
        self.has_ball = False
        self.get_kde_ball()
        self.cfsrmsk = None
        self.get_fillmask_cof(self.maskr())
        
    def get_kde_ball(self):
        '''
        '''
        self.cfsr_ball = self.kdetree.query_ball_tree(self.romsgrd.kdetree, self.balldist)
        self.cfsr_ball = np.array(self.cfsr_ball).nonzero()[0]
        self.has_ball = True
        return self
    
    def check_ball(self):
        plt.figure()
        ROMS = plt.scatter(self.romsgrd.lon()[self.romsgrd.maskr() == 1],
                           self.romsgrd.lat()[self.romsgrd.maskr() == 1],
                            s=10, c='g', edgecolor='none', label='ROMS points')
        CFSR = plt.scatter(self.lon().flat[self.cfsr_ball],
                           self.lat().flat[self.cfsr_ball],
                            s=20, c='r', edgecolor='none', label='CFSR points')
        lines = [CFSR, ROMS]
        plt.legend(lines, [l.get_label() for l in lines])
        plt.axis('image')
        plt.show()
      
      

class CfsrPrate(CfsrData):
    '''CFSR Precipitation rate class (inherits from CfsrData class)
       Responsible for one variable: 'prate'
    '''
    def __init__(self, cfsr_dir, prate_file, masks, romsgrd):
        super(CfsrPrate, self).__init__(cfsr_dir + prate_file[0], prate_file[1], 'CFSR',
                                        romsgrd, masks=masks)
        self.print_weights()
        #self.select_mask(masks)
        #self.get_fillmask_cof()
        #self._maskr = np.ones(self.lon().shape)

    def convert_cmday(self):
        self.datain *= 86400 # to days
        self.datain *= 0.1 # to cm
        return self
      



class CfsrSST(CfsrData):
    '''CFSR SST class (inherits from CfsrData class)
       Responsible for one variable: 'SST'
    '''
    def __init__(self, cfsr_dir, sst_file, masks, romsgrd):
        super(CfsrSST, self).__init__(cfsr_dir + sst_file[0], sst_file[1], 'CFSR',
                                      romsgrd, masks=masks)
        self.print_weights()
        #self.get_fillmask_cof()
    



class CfsrRadlw(CfsrData):
    '''CFSR Outgoing longwave radiation class (inherits from CfsrData class)
       Responsible for two ROMS bulk variables: 'radlw' and 'radlw_in'
    '''
    def __init__(self, cfsr_dir, radlw_file, masks, romsgrd):
        super(CfsrRadlw, self).__init__(cfsr_dir + radlw_file['shflux_LW_down'][0],
                                                   radlw_file['shflux_LW_down'][1], 'CFSR',
                                                   romsgrd, masks=masks)
        self.print_weights()
        self.down_varname = radlw_file['shflux_LW_down'][1]
        self.up_varname = radlw_file['shflux_LW_up'][1]
        #self.select_mask(masks)
        #self.get_fillmask_cof()
        self.radlw_datain = np.ma.empty(self.datain.shape)
        self.radlw_in_datain = np.ma.empty(self.datain.shape)
        self.radlw_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.radlw_in_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.numvars = 2
    
    def _get_radlw(self):
        # First, get radlw
        self.radlw_in_datain[:] = self._get_cfsr_data(self.down_varname)
        self.radlw_datain[:] = self._get_cfsr_data(self.up_varname)
        self.radlw_datain -= self.radlw_in_datain
        
    def _get_radlw_in(self, sst_datain):
        # Second, get radlw_in
        sst_datain **= 4
        sst_datain *= self.eps
        sst_datain *= self.Stefan
        self.radlw_in_datain[:] = -(self.radlw_datain - sst_datain)
    
    def get_cfsr_data(self, sst_datain):
        self._get_radlw()
        self._get_radlw_in(sst_datain)
        return self
    
    def interp2romsgrd(self, fillmask1=False, fillmask2=False):
        self.datain[:] = self.radlw_datain
        if fillmask1:
            self.fillmask()
        self._interp2romsgrd()._check_for_nans()
        self.radlw_dataout[:] = self.dataout.reshape(self.romsgrd.lon().shape)
        self.datain[:] = self.radlw_in_datain
        if fillmask2:
            self.fillmask()
        self._interp2romsgrd()._check_for_nans()
        self.radlw_in_dataout[:] = self.dataout.reshape(self.romsgrd.lon().shape)
        return self
        
      
class CfsrRadsw(CfsrData):
    '''CFSR Outgoing longwave radiation class (inherits from CfsrData class)
       Responsible for one ROMS bulk variable: 'radsw'
    '''
    def __init__(self, cfsr_dir, radsw_file, masks, romsgrd):
        super(CfsrRadsw, self).__init__(cfsr_dir + radsw_file['shflux_SW_down'][0],
                                                   radsw_file['shflux_SW_down'][1],'CFSR',
                                                   romsgrd, masks=masks)
        self.print_weights()
        self.down_varname = radsw_file['shflux_SW_down'][1]
        self.up_varname = radsw_file['shflux_SW_up'][1]

    def get_cfsr_data(self):
        ''' Overides CfsrData.get_cfsr_data()
        '''
        self.datain[:] = self._get_cfsr_data(self.down_varname)
        self.datain -= self._get_cfsr_data(self.up_varname) # downward - upward
        return self
        

class CfsrRhum(CfsrData):
    '''CFSR relative humidity class (inherits from CfsrData class)
       Responsible for one ROMS bulk variable: 'rhum'
    '''
    def __init__(self, cfsr_dir, rhum_file, masks, romsgrd):
        super(CfsrRhum, self).__init__(cfsr_dir + rhum_file[0], rhum_file[1], 'CFSR',
                                       romsgrd, masks=masks)
        self.print_weights()
        #self.get_fillmask_cof()
        

class CfsrQair(CfsrData):
    '''CFSR qair class (inherits from CfsrData class)
       Responsible for one variable: 'qair'
    '''
    def __init__(self, cfsr_dir, qair_file, masks, romsgrd):
        super(CfsrQair, self).__init__(cfsr_dir + qair_file[0], qair_file[1], 'CFSR',
                                       romsgrd, masks=masks)
        self.print_weights()
    
        

class CfsrSat(CfsrData):
    '''CFSR surface air temperature class (inherits from CfsrData class)
       Responsible for one ROMS bulk variable: 'tair'
    '''
    def __init__(self, cfsr_dir, sat_file, masks, romsgrd):
        super(CfsrSat, self).__init__(cfsr_dir + sat_file[0], sat_file[1], 'CFSR',
                                      romsgrd, masks=masks)
        self.print_weights()

    
        

class CfsrSap(CfsrData):
    '''CFSR surface air pressure class (inherits from CfsrData class)
       Responsible for one variable: 'qair'
    '''
    def __init__(self, cfsr_dir, sap_file, masks, romsgrd):
        super(CfsrSap, self).__init__(cfsr_dir + sap_file[0], sap_file[1], 'CFSR',
                                      romsgrd, masks=masks)
        self.print_weights()
        


class CfsrWspd(CfsrData):
    '''CFSR wind speed class (inherits from CfsrData class)
       Responsible for four variables: 'uspd', 'vspd', 'sustr' and 'svstr'
       Requires: 'qair', sap' and 'rhum'
    '''
    def __init__(self, cfsr_dir, wspd_file, masks, romsgrd):
        super(CfsrWspd, self).__init__(cfsr_dir + wspd_file['uspd'][0],
                                                  wspd_file['uspd'][1], 'CFSR',
                                                  romsgrd, masks=masks)
        self.print_weights()
        self.uwnd_varname = wspd_file['uspd'][1]
        self.vwnd_varname = wspd_file['vspd'][1]
        self.wspd_datain = np.ma.empty(self.lon().shape)
        self.uwnd_datain = np.ma.empty(self.lon().shape)
        self.vwnd_datain = np.ma.empty(self.lon().shape)
        self.ustrs_datain = np.ma.empty(self.lon().shape)
        self.vstrs_datain = np.ma.empty(self.lon().shape)
        self._rair_datain = np.ma.empty(self.lon().shape)
        self.wspd_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.uwnd_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.vwnd_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.ustrs_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.vstrs_dataout = np.ma.empty(self.romsgrd.lon().shape)

    def _get_wstrs(self, airsea, rhum_data, sat_data, sap_data, qair_data):
        sat_data -= self.Kelvin # convert from K to C
        sap_data *= 0.01 # convert from Pa to mb
        # Smith etal 1988
        self._rair_datain[:] = airsea.air_dens(sat_data, rhum_data, sap_data, qair_data)
        self.ustrs_datain[:], self.vstrs_datain[:] = airsea.stresstc(self.uwnd_datain,
                                                                     self.vwnd_datain,
                                                                           sat_data,
                                                                     self._rair_datain)

    def _get_wspd(self):
        self.uwnd_datain[:] = self._get_cfsr_data(self.uwnd_varname)
        self.vwnd_datain[:] = self._get_cfsr_data(self.vwnd_varname)
        self.wspd_datain[:] = np.hypot(self.uwnd_datain, self.vwnd_datain)

    def get_winds(self, airsea, rhum, sat, sap, qair):
        self._get_wspd()
        self._get_wstrs(airsea, rhum.datain, sat.datain, sap.datain, qair.datain)

    def interp2romsgrd(self):
        roms_shape = self.romsgrd.lon().shape
        self.datain[:] = self.wspd_datain
        self._interp2romsgrd()._check_for_nans()
        self.wspd_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.uwnd_datain
        self._interp2romsgrd()._check_for_nans()
        self.uwnd_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.vwnd_datain
        self._interp2romsgrd()._check_for_nans()
        self.vwnd_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.ustrs_datain
        self._interp2romsgrd()._check_for_nans()
        self.ustrs_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.vstrs_datain
        self._interp2romsgrd()._check_for_nans()
        self.vstrs_dataout[:] = self.dataout.reshape(roms_shape)
        return self




class AirSea(object):
    '''
    
    '''
    def __init__(self):
        '''
        Constants from:
          http://woodshole.er.usgs.gov/operations/sea-mat/air_sea-html/as_consts.html
        '''
        # Physical constants
        self.G = np.asarray(9.81) # acceleration due to gravity [m/s^2] (same ROMS as scalars.h)
        self.EPS_AIR = np.asarray(0.62197) # molecular weight ratio (water/air)
        self.C2K = np.asarray(273.15) # conversion factor for [C] to [K]
        self.GAS_CONST_R = np.asarray(287.04) # gas constant for dry air [J/kg/K]
        
        # Meteorological constants
        self.Z = 10. # Default measurement height [m]
        self.KAPPA = np.asarray(0.41) # von Karman's constant (same ROMS as scalars.h)
        self.CHARNOCK_ALPHA = np.asarray(0.011) # Charnock constant (for determining roughness length
        '''                            at sea given friction velocity), used in Smith
                                       formulas for drag coefficient and also in Fairall
                                       and Edson.  use alpha=0.011 for open-ocean and
                                       alpha=0.018 for fetch-limited (coastal) regions.'''
        self.R_ROUGHNESS = np.asarray(0.11) # Limiting roughness Reynolds # for aerodynamically
        '''                        smooth flow'''
        
        # Defaults suitable for boundary-layer studies  
        #self.cp = 1004.8 # heat capacity of air [J/kg/K] (same as Roms_tools dQdSST.m)
        #self.rho_air = 1.22 # air density (when required as constant) [kg/m^2]
        #self.Ta = 10. # default air temperature [C]
        #self.Pa = 1020. # default air pressure for Kinneret [mbars]
        #self.psych_default = 'screen' # default psychmometer type (see relhumid.m)
        # Saturation specific humidity coefficient reduced by 2% over salt water
        self.QSAT_COEFF = np.asarray(0.98)
        self.TOL = np.float128(0.00001)


    def viscair(self, Ta):
        '''
        vis=VISCAIR(Ta) computes the kinematic viscosity of dry air as a
        function of air temperature following Andreas (1989), CRREL Report
        89-11.

        INPUT: Ta - air temperature [C]
            1.326e-5*(1 + 6.542e-3*Ta + 8.301e-6*Ta.^2 - 4.84e-9*Ta.^3)
        OUTPUT: vis - air viscosity [m^2/s]
        '''
        Ta = np.asarray(Ta)
        #va = 6.542e-3 * Ta
        #va += 8.301e-6 * Ta**2
        #va -= 4.84e-9 * Ta**3
        #va += 1
        #va *= 1.326e-5
        va = ne.evaluate('1.326e-5 * (1 + 6.542e-3 * Ta + 8.301e-6 * \
                          Ta**2 - 4.84e-9 * Ta**3)')
        return va
    
    
    def cdntc(self, sp, Ta):
        '''
        cd, u10 = cdntc(sp, z , Ta) computes the neutral drag coefficient and
        wind speed at 10m given the wind speed and air temperature at height z
        following Smith (1988), J. Geophys. Res., 93, 311-326.
    
        INPUT:   sp - wind speed  [m/s]
             self.Z - measurement height [m]
            self.Ta - air temperature (optional)  [C]
    
        OUTPUT:  cd - neutral drag coefficient at 10m
                  u10 - wind speed at 10m  [m/s]
        ''' 
        R_roughness = self.R_ROUGHNESS
        z = self.Z
        kappa = self.KAPPA
        Charnock_alpha = self.CHARNOCK_ALPHA
        g = self.G
        
        # Iteration endpoint
        TOL = self.TOL
        
        # Get air viscosity
        visc = self.viscair(Ta)
        #print '----------visc',visc
        
        # Remove any sp==0 to prevent division by zero
        np.place(sp, sp == 0, 0.1)
        
        # Initial guess
        ustaro = np.zeros(sp.shape)
        ustarn = np.copy(sp)
        ustarn = ne.evaluate('ustarn * 0.036')
        #print 'ustarn1', ustarn
        
        # Iterate to find z0 and ustar
        while np.any(ne.evaluate('abs(ustarn - ustaro)') > TOL):
            ustaro = np.copy(ustarn)
            z0 = ne.evaluate('Charnock_alpha * ustaro**2 / g + \
                              R_roughness * visc / ustaro')
            ustarn = ne.evaluate('sp * (kappa / log(z / z0))')
     
        #print 'ustarn2', ustarn
        sqrcd = ne.evaluate('kappa / log(10. / z0)')
        cd = ne.evaluate('sqrcd**2') # np.power(sqrcd, 2)
        #u10 = ustarn / sqrcd
        u10 = ne.evaluate('ustarn / sqrcd')
        #print cd, u10
        return cd, u10
        
        
    def stresstc(self, sp, u, v, Ta, rho_air):
        '''
        taux, tauy = stresstc(u, v) computes the neutral wind stress given the
        wind speed and air temperature at height z following Smith (1988),
        J. Geophys. Res., 93, 311-326

        INPUT:  u, v  - wind speed components   [m/s]
                z     - measurement height  [m]
                Ta    - air temperature [C]
                rho_air  - air density  [kg/m^3]

        OUTPUT: taux, tauy - wind stress components  [N/m^2]
        Adapted from http://woodshole.er.usgs.gov/operations/sea-mat/air_sea-html/as_consts.html
        '''
        # NOTE: windspd is calculated in __main__ before call to self.stresstc, however if
        # windspd_fill_mask == True there can be issues at the coast in taux, tauy
        # Hence, we calculate it again without mask filling
        #sp = np.hypot(u, v) # this line commented Nov2014 cos redundant from _get_wspd
        cd, u10 = self.cdntc(sp, Ta)
        taux = np.copy(rho_air)
        tauy = np.copy(rho_air)
        taux *= u10
        taux *= u
        taux *= cd
        tauy *= u10
        tauy *= v
        tauy *= cd
        return taux, tauy



    def air_dens(self, Ta, RH, Pa, Q):
        '''
        rhoa=AIR_DENS(Ta,RH,Pa) computes the density of moist air.
        Air pressure is optional.

        INPUT:   Ta  - air temperature Ta  [C]
                 RH  - relative humidity  [%]
                 Pa  - air pressure (optional) [mb]
                 Q   - Q air
        OUTPUT:  rhoa - air density  [kg/m^3]
        '''
        #if Pa is not None:
            #self.Pa = Pa # pressure in mb
        
        #if Q is None:
            #Q = self.qsat(Ta, Pa)
            #Q *= 0.01
            #Q *= RH # specific humidity of air [kg/kg]
        
        o_six_one = 1. / self.EPS_AIR - 1. # 0.61 (moisture correction for temp.)
        o_six_one *= Q
        o_six_one += 1.
        Tv = np.copy(Ta)
        Tv += self.C2K
        Tv *= o_six_one # air virtual temperature
        Tv *= self.GAS_CONST_R
        rhoa = np.copy(Pa)
        rhoa *= 100.
        rhoa /= Tv # air density [kg/m^3]
        return rhoa


    def qsat(self, Ta, Pa=None):
        '''
        QSAT: computes specific humidity at saturation.
        q=QSAT(Ta) computes the specific humidity (kg/kg) at saturation at
        air temperature Ta (deg C). Dependence on air pressure, Pa, is small,
        but is included as an optional input.

          INPUT:   Ta - air temperature  [C]
                   Pa - (optional) pressure [mb]
          
           OUTPUT:  q  - saturation specific humidity  [kg/kg]

        Version 1.0 used Tetens' formula for saturation vapor pressure
        from Buck (1981), J. App. Meteor., 1527-1532.  This version
        follows the saturation specific humidity computation in the COARE
        Fortran code v2.5b.  This results in an increase of ~5% in
        latent heat flux compared to the calculation with version 1.0.
        '''

        if Pa is not None:
            self.Pa = Pa # pressure in mb

        # As in Fortran code v2.5b for COARE
        ew = 6.1121 * (1.0007 + 3.46e-6 * self.Pa) * np.exp((17.502 * Ta) / \
             (240.97 + Ta)) # in mb
        q  = 0.62197 * (ew / (self.Pa - 0.378 * ew)) # mb -> kg/kg
        return q
        
        
    def delq(self, Ts, Ta, rh, Pa=None):
        '''
        dq=DELQ(Ts,Ta,rh) computes the specific humidity (kg/kg) difference
        between the air (as determined by relative humidty rh and air
        temperature Ta measurements) and the sea surface (where q is
        assumed to be at 98% saturation at the sea surface temperature Ts).
        DELQ uses QSAT based on Tetens' formula for saturation vapor
        pressure from Buck (1981), J. App. Meteor., 1527-1532.  The
        dependence of QSAT on pressure is small (<0.5%) and has been
        removed using a mean pressure of 1020 mb.

            INPUT:   Ts - sea surface temperature  [C]
                     Ta - air temperature  [C]
                     rh - relative humidity  [%]
                     Pa - (optional) pressure [mb]

            OUTPUT:  dq - air-sea specific humidity difference  [kg/kg]
        '''
        if Pa is not None:
            self.Pa = Pa # pressure in mb
        
        dq = 0.01 * rh * self.qsat(Ta, self.Pa) - \
             self.QSAT_COEFF * self.qsat(Ts, self.Pa)
        return dq





if __name__ == '__main__':
    
    '''
    pycfsr2frc

    Prepare interannual ROMS surface forcing with CFSR data from
    
      http://rda.ucar.edu/pub/cfsr.html
    
    CFSR surface data for ROMS forcing are global but subgrids can be
    selected. User must supply a list of the files available, pycfsr2frc
    will loop through the list, sampling and interpolating each variable.
    ROMS needs the following variables:
      EP : evaporation - precipitation
      
      Net heat flux
          Qnet = SW - LW - LH - SH
      where SW denotes net downward shortwave radiation,
            LW net downward longwave radiation,
            LH latent heat flux,
        and SH sensible heat flux
    
    Note that there are dependencies for:
      dQdSS <- 
    
    CFSR grids, for info see http://rda.ucar.edu/datasets/ds093.2/docs/moly_filenames.html
        Horizontal resolution indicator, 4th character of filename:
        h - high (0.5-degree) resolution
        a - high (0.5-degree) resolution, spl type only
        f - full (1.0-degree) resolution
        l - low (2.5-degree) resolution
      But some files labelled 'l' are in fact 0.3-degree, eg, UWND, VWND...
    
    Notes about the data quality:
    1) The 0.3deg flxf06.gdas.DSWRF.SFC.grb2.nc is ugly
    
    
    Evan Mason, IMEDEA, 2012
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    domain = 'MedSea'
    #domain = 'NEA' 
    
    # CFSR information_________________________________
    if 'MedSea' in domain:
        cfsr_dir = '/shared/emason/NCEP-CFSR/'
    elif 'NEA' in domain:
        cfsr_dir = '/shared/emason/NCEP-CFSR/S-14_N56__W-63_E12/'
    

    
    # Filenames of needed CFSR variables
    SSS_file            = 'ocnh01.gdas.SALTY.5m.grb2.nc'
    swflux_file         = 'ocnh01.gdas.EMNP.SFC.grb2.nc'
    prate_file          = 'flxf01.gdas.PRATE.SFC.grb2.nc'
    shflux_SW_down_file = 'flxl01.gdas.DSWRF.SFC.grb2.nc'
    shflux_SW_up_file   = 'flxl01.gdas.USWRF.SFC.grb2.nc'
    shflux_LW_down_file = 'flxl01.gdas.DLWRF.SFC.grb2.nc'
    shflux_LW_up_file   = 'flxl01.gdas.ULWRF.SFC.grb2.nc'
    shflux_LH_file      = 'flxl01.gdas.LHTFL.SFC.grb2.nc'
    shflux_SH_file      = 'flxl01.gdas.SHTFL.SFC.grb2.nc'
    if 'MedSea' in domain:
        sustr_file      = 'flxf01.gdas.UWND.10m.grb2.nc' # use for MedSea
        svstr_file      = 'flxf01.gdas.VWND.10m.grb2.nc' # use for MedSea
        SST_file        = 'pgbh01.gdas.TMP.SFC.grb2.nc' # use for MedSea
    elif 'NEA' in domain:
        sustr_file      = 'flxf01.gdas.WND.10m.grb2.nc' # use for NEA
        svstr_file      = 'flxf01.gdas.WND.10m.grb2.nc' # use for NEA
        SST_file        = 'ocnh01.gdas.TMP.SFC.grb2.nc' # use for NEA
    
    # Surface air temperature
    # file 'flxf01.gdas.TMP.2m.grb2.nc' compares well with Roms_tools sat.cdf
    if 'MedSea' in domain:
        sat_file        = 'flxf01.gdas.TMP.2m.grb2.nc' # use for MedSea
        sap_file        = 'pgbhnl.gdas.PRES.SFC.grb2.nc' # use for MedSea
    elif 'NEA' in domain:
        sat_file        = 'pgbh01.gdas.TMP.2m.grb2.nc' # use for NEA
        sap_file        = 'flxf01.gdas.PRES.SFC.grb2.nc' # use for NEA
    
    # Relative humidity
    # file 'pgbh01.gdas.R_H.2m.grb2.nc' compares well with Roms_tools rh.cdf
    rel_hum_file        = 'pgbh01.gdas.R_H.2m.grb2.nc'
    qair_file           = 'flxf01.gdas.SPF_H.2m.grb2.nc'
    
    # Filenames of masks for the 1.0, 0.5 and 0.3 degree grids
    mask10_file  = 'pgbl01.gdas.LAND.SFC.grb2.nc'
    mask05_file  = 'pgbh01.gdas.LAND.SFC.grb2.nc'
    mask03_file  = 'flxf01.gdas.LAND.SFC.grb2.nc'
    mask1_8_file = 'flxl01.gdas.LAND.SFC.grb2.nc'
    maskocn_file = 'ocnh01.gdas.SALTY.5m.grb2.nc'

    
    
    # ROMS configuration information_________________________________
    
    #roms_dir = '/marula/emason/runs2012/MedSea15/'
    #roms_dir = '/shared/emason/marula/emason/runs2012/MedSea5/'
    #roms_dir = '/home/emason/runs2012_tmp/MedSea5_R2.5/'
    #roms_dir = '/marula/emason/runs2012/MedSea5_intann_monthly/'
    #roms_dir = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir = '/Users/emason/toto/'
    #roms_dir = '/marula/emason/runs2013/cb_3km_2013_intann/'
    #roms_dir  = '/marula/emason/runs2013/AlbSea_1pt25/'
    #roms_dir    = '/marula/emason/runs2013/cart500/'
    roms_dir = '/marula/emason/runs2012/MedSea_Romain/'
    
    #roms_grd = 'grd_MedSea5_R2.5.nc'
    #roms_grd    = 'grd_MedSea5.nc'
    #roms_grd = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd = 'cb_2009_3km_grd_smooth.nc'
    #roms_grd    = 'grd_AlbSea_1pt25.nc'
    #roms_grd    = 'grd_cart500.nc'
    roms_grd    = 'roms_grd_LongTerm_Romain_new3km.nc'
    
    # Forcing file
    #frc_filename = 'frc_MedSea5_test.nc' # ini filename
    #frc_filename = 'frc_MedSea5_1985010100_new.nc'
    #frc_filename = 'frc_MedSea5_1985010100_64bit.nc'
    #frc_filename = 'frc_CFSR_NA_7pt5km.nc'
    #frc_filename = 'frc_CFSR_NA_7pt5km_UPDATE.nc'
    #frc_filename = 'frc_2013_cb3km_CFSR_UPDATE.nc'
    #frc_filename = 'frc_AlbSea_1pt25_CFSR_20030101.nc'
    #frc_filename = 'frc_cart500.nc'
    #frc_filename = 'test_AlbSea_1pt25.nc'
    frc_filename = 'frc_romain_dqdsst.nc'

    # True for bulk forcing file, else standard dQdSTT forcing file
    bulk = False # variables ()

    # True if the frc file being prepared is for a downscaled simulation
    downscaled = True
    if downscaled:
        # Point to parent directory, where pyccmp2frc expects to find 
        # start_date.mat (created by set_ROMS_interannual_start.py)
        par_dir = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
        #par_dir = '/marula/emason/runs2012/MedSea5_intann_monthly/'

    # Start and end dates of the ROMS simulation
    # must be strings, format 'YYYYMMDDHH'
    #start_date = '1985010100'
    #end_date   = '1987102800'
    #end_date   = '2009123100'
    #end_date   = '2010120101'
    start_date = '1997010100'
    end_date   = '2000123123'
    #start_date = '1997010100'
    #end_date   = '2000123100'
    #start_date = '2002010100'
    #end_date   = '2004063000'
    #start_date = '2003010100'
    #end_date   = '2003123000'
    

    
    # Flag to make 360-day years
    make360 = False # False recommended
    
    
    cycle_length = 0


    # Option for river runoff climatology
    #   Note, a precomputed *coast_distances.mat* must be available
    #   in roms_dir; this is computed using XXXXXX.py
    add_dai_runoff = False # True of False
    if add_dai_runoff:
        dai_file = '/home/emason/matlab/runoff/dai_runoff_mon_-180+180.nc'
        #dai_file = '/home/emason/matlab/runoff/dai_runoff_mon_0_360.nc'
    
    
    # Interpolation / processing parameters_________________________________
    
    balldist = 250000. # distance (m) for kde_ball (should be 2dx at least?)
    
    # Filling of landmask options
    # Set to True to extrapolate sea data over land
    wind_fill_mask    = False # 10 m (False recommended)
    sat_fill_mask     = True # 2 m (True recommended)
    rel_hum_fill_mask = True # 2 m (True recommended)
    qair_fill_mask    = True # 2 m (True recommended)
    
    windspd_fill_mask = True # surface (True recommended)


    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    
    
    # This dictionary of CFSR files needs to supply some or all of the surface
    # forcing variables variables:
    if bulk:
        cfsr_files = OrderedDict([
                ('prate', prate_file),
                ('radlw', OrderedDict([
                  ('shflux_LW_down', shflux_LW_down_file),
                  ('shflux_LW_up', shflux_LW_up_file)])),
                ('radsw', OrderedDict([
                  ('shflux_SW_down', shflux_SW_down_file),
                  ('shflux_SW_up', shflux_SW_up_file)])),
                ('wspd', OrderedDict([
                  ('sat', sat_file),
                  ('rel_hum', rel_hum_file),
                  ('sap', sap_file),
                  ('qair', qair_file),
                  ('uspd', sustr_file),
                  ('vspd', svstr_file)]))])
    
    else:
        cfsr_files = OrderedDict([
                ('SSS', SSS_file),
                ('swflux', swflux_file),
                ('shflux', OrderedDict([
                  ('shflux_SW_down', shflux_SW_down_file),
                  ('shflux_SW_up', shflux_SW_up_file),
                  ('shflux_LW_down', shflux_LW_down_file),
                  ('shflux_LW_up', shflux_LW_up_file),
                  ('shflux_LH', shflux_LH_file),
                  ('shflux_SH', shflux_SH_file)])),
               ('dQdSST', OrderedDict([
                  ('uspd', sustr_file),
                  ('vspd', svstr_file),
                  ('SST', SST_file),
                  ('sat', sat_file),
                  ('rel_hum', rel_hum_file),
                  ('sap', sap_file),
                  ('qair', qair_file)]))])

        
        
        #cfsr_files = OrderedDict([
                 #('SSS', SSS),
          #('dQdSST', OrderedDict([
              #('sustr', sustr),
              #('svstr', svstr),
              #('SST', SST),
              #('sat', sat),
              #('rel_hum', rel_hum),
              #('sap', sap),
              #('qair', qair)]))])
        
    
    # Masks for the 1.0, 0.5 and 0.3 degree grids
    cfsr_masks = OrderedDict([
        ('mask10', mask10_file),
        ('mask05', mask05_file),
        ('mask03', mask03_file),
        ('maskocn', maskocn_file),
        ('mask1_8', mask1_8_file)])
    
    
    # Initialise an AirSea object
    airsea = AirSea()
    
    dtstrdt = plt.datetime.datetime(np.int(start_date[:4]),
                                    np.int(start_date[4:6]),
                                    np.int(start_date[6:8]),
                                    np.int(start_date[8:10]))
    
    dtenddt = plt.datetime.datetime(np.int(end_date[:4]),
                                    np.int(end_date[4:6]),
                                    np.int(end_date[6:8]),
                                    np.int(end_date[8:10]))
    
    # Number of records at monthly frequency
    numrec = (dtenddt.year - dtstrdt.year) * 12 + dtenddt.month - dtstrdt.month + 1
    
    dtstr, dtend = dt.date2num(dtstrdt), dt.date2num(dtenddt)
    
    if downscaled:
        inidate = io.loadmat(par_dir + 'start_date.mat')
        deltaday0 = dtstr - inidate['start_date']
        if make360:
            deltaday0 *= 360 / 365.
    
    # Initialise a RomsGrid object
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)))
    romsgrd.roms_dir = roms_dir
    
    # Create the forcing file
    romsgrd.create_frc_nc(''.join((roms_dir, frc_filename)),
                          start_date, end_date, numrec, cycle_length, 'pycfsr2frc')

    
    # Gnomonic projections for horizontal interpolations
    roms_M = romsgrd.get_gnom_trans()
    roms_points = romsgrd.proj2gnom(roms_M, ignore_land_points=True) # we only want data points

    # Create ROMS grid KDE tree
    roms_tree = sp.KDTree(roms_points)
    
    
    # Get all CFSR mask and grid sizes
    mask03 = CfsrData(''.join((cfsr_dir, cfsr_masks['mask03'])))
    grid03 = CfsrGrid(''.join((cfsr_dir, cfsr_masks['mask03'])))
    mask05 = CfsrData(''.join((cfsr_dir, cfsr_masks['mask05'])))
    grid05 = CfsrGrid(''.join((cfsr_dir, cfsr_masks['mask05'])))
    mask10 = CfsrData(''.join((cfsr_dir, cfsr_masks['mask10'])))
    grid10 = CfsrGrid(''.join((cfsr_dir, cfsr_masks['mask10'])))
    mask1_8 = CfsrData(''.join((cfsr_dir, cfsr_masks['mask1_8'])))
    grid1_8 = CfsrGrid(''.join((cfsr_dir, cfsr_masks['mask1_8'])))
    maskocn = CfsrData(''.join((cfsr_dir, cfsr_masks['maskocn'])))
    gridocn = CfsrGrid(''.join((cfsr_dir, cfsr_masks['maskocn'])))
    
    # Arrays of masks and grids used by 'metrics' method
    masks = np.array([mask03, mask05, mask10, maskocn, mask1_8])
    grids = np.array([grid03, grid05, grid10, gridocn, grid1_8])
    
    
    
    
    # Loop over the CFSR files
    # Each CFSR file contains a different variable
    for cfsr_key, cfsr_file in zip(cfsr_files.keys(), cfsr_files.values()):
        
        # The 'active' flag is True when looping through a desired time range
        flags = dict(active = False)
                
        # Used for dQdSST
        if cfsr_key in 'shflux':
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file['shflux_SW_down'])))
        
        # Used for dQdSST
        elif cfsr_key in 'dQdSST':
            # Using uspd/sustr here cos has highest resolution (0.3 deg.)
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file['sustr'])))
        
        # Used for bulk
        elif cfsr_key in 'wspd':
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file['uspd'])))
        
        # Used for bulk
        elif cfsr_key in 'radlw':
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file['shflux_LW_down'])))
        
        # Used for bulk
        elif cfsr_key in 'radsw':
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file['shflux_SW_down'])))
        
        # Used for both dQdSST and bulk
        else:
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file)))
        
        # Get KDE tree
        cfsr_tree, cfsr_points_all = cfsrgrd.get_points_kdetree(roms_points, roms_M)
            
        # Set starting metrics for grid comparisons
        if cfsr_key == cfsr_files.keys()[0]:
            metrics = cfsrgrd.metrics()
        
        try:
            cfsr = CfsrData(''.join((cfsr_dir, cfsr_file)))
            print 'Opening file:', ''.join((cfsr_dir, cfsr_file))
            
        except Exception:
            print 'Opening files at:', cfsr_dir
            for value in cfsr_file.values():
                print '  ', value
        
        # Note that, as some of the CFSR variables may be on different grids,
        # a new ball will be created each time a different grid
        # is detected...
        if cfsr_key == cfsr_files.keys()[0] or np.alltrue(metrics != cfsrgrd.metrics()):
            
            print '--- compute a new kde ball (may take a while)'
            cfsr_ball = np.array(cfsr_tree.query_ball_tree(roms_tree, balldist)).nonzero()[0]
            print '--- got kde ball'
            
            metrics = cfsrgrd.metrics()

            # Get cfsr_points covering subdomain only
            cfsr_points = np.array([cfsr_points_all[:,0].flat[cfsr_ball],
                                    cfsr_points_all[:,1].flat[cfsr_ball]]).T
            # Make new cfsr_tri that covers subdomain
            cfsr_tri = sp.Delaunay(cfsr_points)
    
            # Select appropriate mask for fillmask
            mask = cfsrgrd.select_mask(grids, masks)
        
        
        # Debug figure (this tests the query_ball_tree)
        if False:
            plt.figure()
            roms_M.scatter(cfsr_points_all[:,0], cfsr_points_all[:,1], c='g')
            roms_M.scatter(cfsr_points[:,0], cfsr_points[:,1], c='r')
            roms_M.drawcoastlines()
            rbx, rby = roms_M(romsgrd.boundary()[0], romsgrd.boundary()[1])
            roms_M.plot(rbx, rby)
            plt.title('KDE ball points for "%s"' %cfsr_key)
            plt.show()
        

        
        tind = 0       
    
        # Loop over the time records in the file
        for cfsri, cfsrt in enumerate(cfsr.vd_time()):

            
            if np.logical_and(
                  np.int(cfsr.vd_time()[cfsri].tostring()) >= np.int(start_date),
                  np.int(cfsr.vd_time()[cfsri].tostring()) <= np.int(end_date)):
                flags['active'] = True
            else:
                flags['active'] = False


            if flags['active']:
                
                cfsr_dt = plt.datetime.datetime(np.int(cfsrt.tostring()[:4]),
                                           np.int(cfsrt.tostring()[4:6]),
                                           np.int(cfsrt.tostring()[6:8]),
                                           np.int(cfsrt.tostring()[8:10]))
                #print 'cfsr_dt.month',cfsr_dt.month
                cfsr_num = dt.date2num(cfsr_dt)


                
                # Precipitation rate (bulk only)
                if 'prate' in cfsr_key:
                    
                    # Fill the mask
                    if flags.has_key('prate_wt'):
                        prate = cfsr.fillmask(cfsr.frc_prate(cfsri), prate_mask, flags['prate_wt'])
                    else:
                        prate_mask = cfsrgrd.select_mask(grids, masks)
                        prate, flags['prate_wt'] = cfsr.fillmask(cfsr.frc_prate(cfsri), prate_mask)
                    prate *= 86400 # to days
                    prate *= 0.1 # to cm
                    cfsrdata = prate
                    cfsr.out_var = 'prate'
                    cfsr.out_time = 'bulk_time'
                    
                
                # Outgoing longwave radiation (bulk only)
                elif 'radlw' in cfsr_key:
                
                    # Fill the mask
                    if flags.has_key('radlw_wt'):
                        lwd = lwd_data.fillmask(lwd_data.frc_shflux_LW_down(cfsri), lw_mask, flags['radlw_wt'])
                    else:
                        lw_mask = cfsrgrd.select_mask(grids, masks)
                        lwd_filename = ''.join((cfsr_dir, cfsr_file['shflux_LW_down']))
                        lwd_data = CfsrData(lwd_filename)
                        lwd = lwd_data.frc_shflux_LW_down(cfsri)
                        lwd, flags['radlw_wt'] = lwd_data.fillmask(lwd, lw_mask)
                        
                        lwu_filename = ''.join((cfsr_dir, cfsr_file['shflux_LW_up']))
                        lwu_data = CfsrData(lwu_filename)
                        
                    cfsrdata = lwu_data.fillmask(lwu_data.frc_shflux_LW_up(cfsri), lw_mask, flags['radlw_wt'])
                    
                    cfsrdata -= lwd
                    cfsr.out_var = 'radlw'
                    cfsr.out_time = 'bulk_time'
                
                    # Interpolate lwd to ROMS grid; this to be used for 'radlw_in'
                    lwd = horizInterp(cfsr_tri, lwd.flat[cfsr_ball])(roms_points)
                    lwd = lwd_data.reshape(lwd, romsgrd.maskr())
                
                
                # Shortwave radiation (bulk only)
                elif 'radsw' in cfsr_key:
                
                    # Fill the mask
                    if flags.has_key('radsw_wt'):
                        swd = swd_data.fillmask(swd_data.frc_shflux_SW_down(cfsri), sw_mask, flags['radsw_wt'])
                    else:
                        sw_mask = cfsrgrd.select_mask(grids, masks)
                        swd_filename = ''.join((cfsr_dir, cfsr_file['shflux_SW_down']))
                        swd_data = CfsrData(swd_filename)
                        swd = swd_data.frc_shflux_SW_down(cfsri)
                        swd, flags['radsw_wt'] = swd_data.fillmask(swd, lw_mask)
                        swu_filename = ''.join((cfsr_dir, cfsr_file['shflux_SW_up']))
                        swu_data = CfsrData(swu_filename)
                    
                    cfsrdata = swu_data.fillmask(swu_data.frc_shflux_SW_up(cfsri), sw_mask, flags['radsw_wt'])
                    #aaaaaaaaaa
                    cfsrdata = swd - cfsrdata
                    cfsr.out_var = 'radsw'
                    cfsr.out_time = 'bulk_time'
                
                
                # Evaporation minus precipitation (dQdSST only)
                elif 'swflux' in cfsr_key: # E - P
                    
                    # Fill the mask
                    if flags.has_key('swflux_wt'):
                        swflux = cfsr.fillmask(cfsr.frc_emp(cfsri), swflux_mask, flags['swflux_wt'])
                    else:
                        swflux_mask = cfsrgrd.select_mask(grids, masks)
                        swflux, flags['swflux_wt'] = cfsr.fillmask(cfsr.frc_emp(cfsri), swflux_mask)
                    
                    cfsrdata = swflux
                    cfsr.out_var = 'swflux'
                    cfsr.out_time = 'swf_time'
                    

                # Sea surface salinity (dQdSST only)
                elif 'SSS' in cfsr_key: # surface salinity (in fact 5m)
                    
                    # Fill the mask
                    if flags.has_key('sss_wt'):
                        sss = cfsr.fillmask(cfsr.frc_sss(cfsri), sss_mask, flags['sss_wt'])
                    else:
                        sss_mask = cfsrgrd.select_mask(grids, masks)
                        sss, flags['sss_wt'] = cfsr.fillmask(cfsr.frc_sss(cfsri), sss_mask)
                    
                    cfsrdata = sss
                    cfsrdata *= 1e3
                    cfsr.out_var = 'SSS'
                    cfsr.out_time = 'sss_time'
                    
                
                # Net heat flux (dQdSST only)
                elif 'shflux' in cfsr_key:
                    #---------------------------------------------
                    ''' Net Heat Flux (shflux)
                    Note, Net heat flux is computed as:
                      Qnet = SW - LW - LH - SH
                      where SW : net downward shortwave radiation
                                     (upward - downward)
                            LW : net downward longwave radiation
                                     (upward - downward)
                            LH : latent heat flux,
                            SH : sensible heat flux.
                    These should all be on the same grid'''
                    #---------------------------------------------
                    # Short wave down
                    if 'swd_filename' in locals():
                        swd = swd_data.fillmask(swd_data.frc_shflux_SW_down(cfsri), swd_mask, flags['swd_wt'])
                    else:
                        swd_filename = ''.join((cfsr_dir, cfsr_file['shflux_SW_down']))
                        swd_data = CfsrData(swd_filename)
                        swd = swd_data.frc_shflux_SW_down(cfsri)
                        swd_grid = CfsrGrid(swd_filename)
                        swd_mask = swd_grid.select_mask(grids, masks)
                        swd, flags['swd_wt'] = swd_data.fillmask(swd, swd_mask)
                        swd_points_all = swd_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        swd_tri = sp.Delaunay(swd_points_all)
                        swd_pts = swd_grid.get_points(roms_M)

                    
                    # Short wave up
                    if 'swu_filename' in locals():
                        swu = swu_data.fillmask(swu_data.frc_shflux_SW_up(cfsri), swu_mask, flags['swu_wt'])
                    else:
                        swu_filename = ''.join((cfsr_dir, cfsr_file['shflux_SW_up']))
                        swu_data = CfsrData(swu_filename)
                        swu = swu_data.frc_shflux_SW_up(cfsri)
                        swu_grid = CfsrGrid(swu_filename)
                        swu_mask = swu_grid.select_mask(grids, masks)
                        swu, flags['swu_wt'] = swu_data.fillmask(swu, swu_mask)
                        swu_points_all = swu_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        swu_tri = sp.Delaunay(swu_points_all)
                        swu_pts = swu_grid.get_points(roms_M)

                    
                    # Long wave down
                    if 'lwd_filename' in locals():
                        lwd = lwd_data.fillmask(lwd_data.frc_shflux_LW_down(cfsri), lwd_mask, flags['lwd_wt'])
                    else:
                        lwd_filename = ''.join((cfsr_dir, cfsr_file['shflux_LW_down']))
                        lwd_data = CfsrData(lwd_filename)
                        lwd = lwd_data.frc_shflux_LW_down(cfsri)
                        lwd_grid = CfsrGrid(lwd_filename)
                        lwd_mask = lwd_grid.select_mask(grids, masks)
                        lwd, flags['lwd_wt'] = lwd_data.fillmask(lwd, lwd_mask)
                        lwd_points_all = lwd_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        lwd_tri = sp.Delaunay(lwd_points_all)
                        lwd_pts = lwd_grid.get_points(roms_M)
                        

                    # Long wave up
                    if 'lwu_filename' in locals():
                        lwu = lwu_data.fillmask(lwu_data.frc_shflux_LW_up(cfsri), lwu_mask, flags['lwu_wt'])
                    else:
                        lwu_filename = ''.join((cfsr_dir, cfsr_file['shflux_LW_up']))
                        lwu_data = CfsrData(lwu_filename)
                        lwu = lwu_data.frc_shflux_LW_up(cfsri)
                        lwu_grid = CfsrGrid(lwu_filename)
                        lwu_mask = lwu_grid.select_mask(grids, masks)
                        lwu, flags['lwu_wt'] = lwu_data.fillmask(lwu, lwu_mask)
                        lwu_points_all = lwu_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        lwu_tri = sp.Delaunay(lwu_points_all)
                        lwu_pts = lwu_grid.get_points(roms_M)

                    # Latent heat
                    if 'lh_filename' in locals():
                        lh = lh_data.fillmask(lh_data.frc_shflux_LH(cfsri), lh_mask, flags['lh_wt'])
                    else:
                        lh_filename = ''.join((cfsr_dir, cfsr_file['shflux_LH']))
                        lh_data = CfsrData(lh_filename)
                        lh = lh_data.frc_shflux_LH(cfsri)
                        lh_grid = CfsrGrid(lh_filename)
                        lh_mask = lh_grid.select_mask(grids, masks)
                        lh, flags['lh_wt'] = lh_data.fillmask(lh, lh_mask)
                        lh_points_all = lh_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        lh_tri = sp.Delaunay(lh_points_all)
                        lh_pts = lh_grid.get_points(roms_M)
                        
                    # Sensible heat
                    if 'sh_filename' in locals():
                        sh = sh_data.fillmask(sh_data.frc_shflux_SH(cfsri), sh_mask, flags['sh_wt'])
                    else:
                        sh_filename = ''.join((cfsr_dir, cfsr_file['shflux_SH']))
                        sh_data = CfsrData(sh_filename)
                        sh = sh_data.frc_shflux_SH(cfsri)
                        sh_grid = CfsrGrid(sh_filename)
                        sh_mask = sh_grid.select_mask(grids, masks)
                        sh, flags['sh_wt'] = sh_data.fillmask(sh, sh_mask)
                        sh_points_all = sh_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        sh_tri = sp.Delaunay(sh_points_all)
                        sh_pts = sh_grid.get_points(roms_M)
                         

                    # Get Qnet
                    # net shortwave = swd - swu
                    # net longwave = lwu - lwd
                    cfsrdata = (swd - swu) - (lwu - lwd) - lh - sh
                    cfsr.out_var = cfsr_key
                    cfsr.out_time = 'shf_time'
                    
                    # Interpolate swd to ROMS grid
                    swd = horizInterp(cfsr_tri, swd.flat[cfsr_ball])(roms_points)
                    swd = cfsr.reshape(swd, romsgrd.maskr())


                # dQdSST
                elif cfsr_key in ('dQdSST', 'wspd'):
                    '''NOTE: 'wspd' corresponds to bulk option
                    '''
                    # U wind speed
                    if 'uspd_filename' in locals():
                        uspd = uspd_data.frc_uspd(cfsri)
                        if wind_fill_mask:
                            uspd = uspd_data.fillmask(uspd, uspd_mask, flags['uspd_wt'])
                        scalar_u = uspd.copy()
                    else:
                        uspd_filename = ''.join((cfsr_dir, cfsr_file['uspd']))
                        uspd_data = CfsrData(uspd_filename)
                        uspd = uspd_data.frc_uspd(cfsri)
                        scalar_u = uspd.copy()
                        uspd_grid = CfsrGrid(uspd_filename)
                        if wind_fill_mask:
                            uspd_mask = uspd_grid.select_mask(grids, masks)
                            uspd, flags['uspd_wt'] = uspd_data.fillmask(uspd, uspd_mask)
                        uspd_points_all = uspd_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        uspd_tri = sp.Delaunay(uspd_points_all)
                        uspd_pts = grid03.get_points(roms_M)
                    # Check if we need to interpolate to 0.3 deg grid
                    if not np.alltrue(uspd_grid.metrics() == grid03.metrics()):
                        print '--- interpolating USPD to 0.3 deg grid'
                        uspd = horizInterp(uspd_tri, uspd.ravel())(uspd_pts)
                        uspd = uspd.reshape(grid03.lon().shape)
                    
                    # V wind speed
                    if 'vspd_filename' in locals():
                        vspd = vspd_data.frc_vspd(cfsri)
                        if wind_fill_mask:
                            vspd = vspd_data.fillmask(vspd, vspd_mask, flags['vspd_wt'])
                        scalar_v = vspd.copy()
                    else:
                        vspd_filename = ''.join((cfsr_dir, cfsr_file['vspd']))
                        vspd_data = CfsrData(vspd_filename)
                        vspd = vspd_data.frc_vspd(cfsri)
                        scalar_v = vspd.copy()
                        vspd_grid = CfsrGrid(vspd_filename)
                        if wind_fill_mask:
                            vspd_mask = vspd_grid.select_mask(grids, masks)
                            vspd, flags['vspd_wt'] = vspd_data.fillmask(vspd, vspd_mask)
                        vspd_points_all = vspd_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        vspd_tri = sp.Delaunay(vspd_points_all)
                        vspd_pts = grid03.get_points(roms_M)
                    # Check if we need to interpolate to 0.3 deg grid
                    if not np.alltrue(vspd_grid.metrics() == grid03.metrics()):
                        print '--- interpolating VSPD to 0.3 deg grid'
                        vspd = horizInterp(vspd_tri, vspd.ravel())(vspd_pts)
                        vspd = vspd.reshape(grid03.lon().shape)
                    
                    
                    # Sea surface temperature
                    if 'dQdSST' in cfsr_key: # if bulk, no need for SST
                        if 'sst_filename' in locals():
                            sst = sst_data.fillmask(sst_data.frc_sst(cfsri), sst_mask, flags['sst_wt'])
                        else:
                            sst_filename = ''.join((cfsr_dir, cfsr_file['SST']))
                            sst_data = CfsrData(sst_filename)
                            sst = sst_data.frc_sst(cfsri)
                            sst_grid = CfsrGrid(sst_filename)
                            sst_mask = sst_grid.select_mask(grids, masks)
                            sst, flags['sst_wt'] = sst_data.fillmask(sst, sst_mask)
                            sst_points_all = sst_grid.get_points_kdetree(roms_points, roms_M)[-1]
                            sst_tri = sp.Delaunay(sst_points_all)
                            sst_pts = grid03.get_points(roms_M)
                        # Check if we need to interpolate to 0.3 deg grid
                        if not np.alltrue(sst_grid.metrics() == grid03.metrics()):
                            print '--- interpolating SST to 0.3 deg grid'
                            sst = horizInterp(sst_tri, sst.ravel())(sst_pts)
                            sst = sst.reshape(grid03.lon().shape)
                    
                    
                    # Specific humidity
                    if 'qair_filename' in locals():
                        qair = qair_data.frc_qair(cfsri)
                        if qair_fill_mask:
                            qair = qair_data.fillmask(qair_data.frc_qair(cfsri), qair_mask, flags['qair_wt'])
                    else:
                        qair_filename = ''.join((cfsr_dir, cfsr_file['qair']))
                        qair_data = CfsrData(qair_filename)
                        qair = qair_data.frc_qair(cfsri)
                        qair_grid = CfsrGrid(qair_filename)
                        if qair_fill_mask:
                            qair_mask = qair_grid.select_mask(grids, masks)
                            qair, flags['qair_wt'] = qair_data.fillmask(qair, qair_mask)
                        qair_points_all = qair_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        qair_tri = sp.Delaunay(qair_points_all)
                        qair_pts = grid03.get_points(roms_M)
                    # Check if we need to interpolate to 0.3 deg grid
                    if not np.alltrue(qair_grid.metrics() == grid03.metrics()):
                        print '--- interpolating QAIR to 0.3 deg grid'
                        qair = horizInterp(qair_tri, qair.ravel())(qair_pts)
                        qair = qair.reshape(grid03.lon().shape)
                    
                    
                    # Surface air temperature
                    if 'sat_filename' in locals():
                        sat = sat_data.frc_sat(cfsri)
                        if sat_fill_mask:
                            sat = sat_data.fillmask(sat_data.frc_sat(cfsri), sat_mask, flags['sat_wt'])
                    else:
                        sat_filename = ''.join((cfsr_dir, cfsr_file['sat']))
                        sat_data = CfsrData(sat_filename)
                        sat = sat_data.frc_sat(cfsri)
                        sat_grid = CfsrGrid(sat_filename)
                        if sat_fill_mask:
                            sat_mask = sat_grid.select_mask(grids, masks)
                            sat, flags['sat_wt'] = sat_data.fillmask(sat, sat_mask)
                        sat_points_all = sat_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        sat_tri = sp.Delaunay(sat_points_all)
                        sat_pts = grid03.get_points(roms_M)
                    # Check if we need to interpolate to 0.3 deg grid
                    if not np.alltrue(sat_grid.metrics() == grid03.metrics()):
                        print '--- interpolating SAT to 0.3 deg grid'
                        sat = horizInterp(sat_tri, sat.ravel())(sat_pts)
                        sat = sat.reshape(grid03.lon().shape)
                        

                    # Surface air pressure
                    if 'sap_filename' in locals():
                        sap = sap_data.fillmask(sap_data.frc_sap(cfsri), sap_mask, flags['sap_wt'])
                    else:
                        sap_filename = ''.join((cfsr_dir, cfsr_file['sap']))
                        sap_data = CfsrData(sap_filename)
                        sap = sap_data.frc_sap(cfsri)
                        sap_grid = CfsrGrid(sap_filename)
                        sap_mask = sap_grid.select_mask(grids, masks)
                        sap, flags['sap_wt'] = sap_data.fillmask(sap, sap_mask)
                        sap_points_all = sap_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        sap_tri = sp.Delaunay(sap_points_all)
                        sap_pts = grid03.get_points(roms_M)
                    # Check if we need to interpolate to 0.3 deg grid
                    if not np.alltrue(sap_grid.metrics() == grid03.metrics()):
                        print '--- interpolating SAP to 0.3 deg grid'
                        sap = horizInterp(sap_tri, sap.ravel())(sap_pts)
                        sap = sap.reshape(grid03.lon().shape)


                    # Relative humidity
                    if 'rel_hum_filename' in locals():
                        rel_hum = rel_hum_data.frc_rel_hum(cfsri)
                        if rel_hum_fill_mask:
                            rel_hum = rel_hum_data.fillmask(rel_hum_data.frc_rel_hum(cfsri), rel_hum_mask, flags['rel_hum_wt'])
                    else:
                        rel_hum_filename = ''.join((cfsr_dir, cfsr_file['rel_hum']))
                        rel_hum_data = CfsrData(rel_hum_filename)
                        rel_hum = rel_hum_data.frc_rel_hum(cfsri)
                        rel_hum_grid = CfsrGrid(rel_hum_filename)
                        if rel_hum_fill_mask:
                            rel_hum_mask = rel_hum_grid.select_mask(grids, masks)
                            rel_hum, flags['rel_hum_wt'] = rel_hum_data.fillmask(rel_hum, rel_hum_mask)
                        rel_hum_points_all = rel_hum_grid.get_points_kdetree(roms_points, roms_M)[-1]
                        rel_hum_tri = sp.Delaunay(rel_hum_points_all)
                        rel_hum_pts = grid03.get_points(roms_M)
                    # Check if we need to interpolate to 0.3 deg grid
                    if not np.alltrue(rel_hum_grid.metrics() == grid03.metrics()):
                        print '--- interpolating REL_HUM to 0.3 deg grid'
                        rel_hum = horizInterp(rel_hum_tri, rel_hum.ravel())(rel_hum_pts)
                        rel_hum = rel_hum.reshape(grid03.lon().shape)
                    
                    
                    # Prepare for dQdSST / bulk winds
                    if windspd_fill_mask:
                        windspd = np.hypot(scalar_u, scalar_v)
                    else:
                        windspd = np.hypot(uspd, vspd)
                    
                    sat -= cfsr.Kelvin # convert from K to C
                    if 'dQdSST' in cfsr_key:
                        sst -= cfsr.Kelvin # convert from K to C
                    sap /= 100. # convert from Pa to mb
                    
                    # Seems ok compared to Roms_tools airdens.cdf
                    rho_air = airsea.air_dens(sat, rel_hum, sap, qair)
                    
                    # Speed to stress (Smith etal 1988)
                    sustr, svstr = airsea.stresstc(uspd, vspd, sat, rho_air)
                    
                    if 'dQdSST' in cfsr_key:    
                        # Get qsea
                        dq = airsea.delq(sst, sat, rel_hum, sap)
                        qsea = np.copy(qair)
                        qsea -= dq
                    
                        cfsrdata = cfsr.dQdSST(sst, sat, rho_air, windspd, qsea)
                        cfsr.out_var = cfsr_key
                        cfsr.out_time = 'sst_time'
                        
                        # Interpolate SST to ROMS grid
                        sst = horizInterp(cfsr_tri, sst.flat[cfsr_ball])(roms_points)
                        sst = cfsr.reshape(sst, romsgrd.maskr())
                    
                    # Interpolate sustr, svstr to ROMS grid
                    sustr = horizInterp(cfsr_tri, sustr.flat[cfsr_ball])(roms_points)
                    sustr = cfsr.reshape(sustr, romsgrd.maskr())
                    svstr = horizInterp(cfsr_tri, svstr.flat[cfsr_ball])(roms_points)
                    svstr = cfsr.reshape(svstr, romsgrd.maskr())
                    
                    if 'wspd' in cfsr_key:
                        # Interpolate uspd, vspd, windspd to ROMS grid
                        uspd = horizInterp(cfsr_tri, uspd.flat[cfsr_ball])(roms_points)
                        uspd = cfsr.reshape(uspd, romsgrd.maskr())
                        vspd = horizInterp(cfsr_tri, vspd.flat[cfsr_ball])(roms_points)
                        vspd = cfsr.reshape(vspd, romsgrd.maskr())
                        sat = horizInterp(cfsr_tri, sat.flat[cfsr_ball])(roms_points)
                        sat = cfsr.reshape(sat, romsgrd.maskr())
                        rel_hum = horizInterp(cfsr_tri, rel_hum.flat[cfsr_ball])(roms_points)
                        rel_hum = cfsr.reshape(rel_hum, romsgrd.maskr())
                        rel_hum /= 100.
                        #cfsrdata_out = horizInterp(cfsr_tri, windspd.flat[cfsr_ball])(roms_points)
                        #cfsrdata_out = cfsr.reshape(cfsrdata_out, romsgrd.maskr())
                        cfsrdata = windspd
                        cfsr.out_var = cfsr_key
                
                else:
                    raise KeyError("Key %s in 'cfsr_files' is suspect" %cfsr_key)
                
                
                # Interpolate to ROMS grid
                cfsrdata_out = horizInterp(cfsr_tri, cfsrdata.flat[cfsr_ball])(roms_points)
                cfsrdata_out = cfsr.reshape(cfsrdata_out, romsgrd.maskr())
                
                # Figure for debugging
                if False and cfsri == 0:
                    cfsrx, cfsry = roms_M(cfsrgrd.lon(), cfsrgrd.lat())
                    plt.figure()
                    plt.subplot(121)
                    plt.title('Interpolated data (%s)' %cfsr_key)
                    rx, ry = roms_M(romsgrd.lon(), romsgrd.lat())
                    cfsrdata_out = np.ma.masked_where(romsgrd.maskr() == 0,
                                                   cfsrdata_out)
                    roms_M.pcolormesh(rx, ry, cfsrdata_out)
                    plt.clim(cfsrdatai.min(), cfsrdatai.max())
                    plt.subplot(122)
                    plt.title('Raw data (%s)' %cfsr_key)
                    cfsrdata = np.ma.masked_where(cfsrmask == 0,
                                               cfsrdata)
                    roms_M.pcolormesh(cfsrx, cfsry, cfsrdata)
                    plt.clim(cfsrdatai.min(), cfsrdatai.max())
                    plt.colorbar()
                    rbx, rby = roms_M(romsgrd.boundary()[0], romsgrd.boundary()[1])
                    roms_M.plot(rbx, rby, 'k')
                    plt.show()
                    
                    
                    
                
                # Write to forcing file
                nc = netcdf.Dataset(romsgrd.frcfile, 'a')
        
                # For dQdSST
                if 'dQdSST' in cfsr_key:

                    sustr, svstr = romsgrd.rotate(sustr, svstr, 1)
                    
                    sustr = romsgrd.rho2u_2d(sustr)
                    sustr *= romsgrd.umask()
                    svstr = romsgrd.rho2v_2d(svstr)
                    svstr *= romsgrd.vmask()
                    
                    nc.variables['sustr'][tind] = sustr
                    nc.variables['svstr'][tind] = svstr
                    nc.variables['SST'][tind] = sst
                    
                # For dQdSST
                elif 'shflux' in cfsr_key:
                    nc.variables['swrad'][tind] = swd
                
                # For bulk
                elif 'radlw' in cfsr_key:
                    nc.variables['radlw_in'][tind] = lwd
                
                # For bulk
                elif 'wspd' in cfsr_key:
                    
                    uspd, vspd = romsgrd.rotate(uspd, vspd, 1)
                    uspd = romsgrd.rho2u_2d(uspd)
                    uspd *= romsgrd.umask()
                    vspd = romsgrd.rho2v_2d(vspd)
                    vspd *= romsgrd.vmask()
                    sustr, svstr = romsgrd.rotate(sustr, svstr, 1)
                    sustr = romsgrd.rho2u_2d(sustr)
                    sustr *= romsgrd.umask()
                    svstr = romsgrd.rho2v_2d(svstr)
                    svstr *= romsgrd.vmask()
                    
                    nc.variables['sustr'][tind] = sustr
                    nc.variables['svstr'][tind] = svstr
                    nc.variables['uwnd'][tind] = uspd
                    nc.variables['vwnd'][tind] = vspd
                    nc.variables['rhum'][tind] = rel_hum
                    nc.variables['tair'][tind] = sat

                    
                # Add the Dai river climatology
                elif add_dai_runoff and 'swflux' in cfsr_key:
                    cfsrdata_out = romsgrd.get_runoff(cfsrdata_out, dai_file, cfsr_dt.month)
                    
                    
                nc.variables[cfsr.out_var][tind] = cfsrdata_out
                
                
                
                
                # Flag if we want 360-day years...
                if make360:
                    # Probably not necessary, but trying to account for leap years
                    if not ca.isleap(cfsr_dt.year):
                        mk360day = 360 / 365.
                    else:
                        mk360day = 360 / 366.
                    theday = (cfsr_num - dtstr) * mk360day
                    theday += 15. # set to middle of month
                else:
                    mid_month = 0.5 * ca.monthrange(cfsr_dt.year, cfsr_dt.month)[1]
                    theday = cfsr_num - dtstr
                    theday+= mid_month
                
                # If 'downscaled' is True this will add the number of days between
                # ini time of the parent and ini time of the new solution
                if downscaled:
                    theday += deltaday0
                
                nc.variables[cfsr.out_time][tind] = theday
                
                if 'dQdSST' in cfsr_key:
                    nc.variables['sms_time'][tind] = theday
                
                elif 'shflux' in cfsr_key:
                    nc.variables['srf_time'][tind] = theday
                
                if cycle_length == 0:
                    cyc_end = theday + 15.
                    if bulk:
                        nc.variables['bulk_time'].cycle_length = cyc_end
                    else:
                        nc.variables['sms_time'].cycle_length = cyc_end
                        nc.variables['shf_time'].cycle_length = cyc_end
                        nc.variables['swf_time'].cycle_length = cyc_end
                        nc.variables['sst_time'].cycle_length = cyc_end
                        nc.variables['srf_time'].cycle_length = cyc_end
                        nc.variables['sss_time'].cycle_length = cyc_end
                
                if not bulk:
                    nc.variables['month'][tind] = cfsr_dt.month
                
                nc.sync()
                
                nc.close()

                print '--date: %s, day: %s, record: %s' %(cfsrt.tostring(), np.float(theday), tind)
                tind += 1
                
                
                
            #if np.int(cfsrt.tostring()) > np.int(end_date):
                               
            #    flags['active'] = False
        print '--%s records written' %tind
    
    if not bulk:
        print '\nNow run scale_cfsr2coads.py to scale with COADS'
    
    
    
    
    
    
    
    
