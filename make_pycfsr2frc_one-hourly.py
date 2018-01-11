# %run make_pycfsr2frc_one-hourly.py

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

Create a forcing file based on hourly CFSR data

===========================================================================
'''
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import scipy.spatial as sp
import numpy as np
import numexpr as ne
import time as time
import matplotlib.pyplot as plt
from collections import OrderedDict
#from datetime import datetime

from pycfsr2frc import RomsGrid, RomsData, CfsrData, CfsrGrid, CfsrMask, AirSea
from py_roms2roms import horizInterp


class CfsrDataHourly(RomsData):
    '''
    CFSR data class (inherits from RomsData class)
    '''
    def __init__(self, filename, varname, model_type, romsgrd, time_array, masks=None):
        '''
        
        '''
        super(CfsrDataHourly, self).__init__(filename, model_type)
        self.varname = varname
        self.time_array = time_array
        self._check_product_description()
        self.needs_averaging_to_hour = False
        self.needs_averaging_to_the_half_hour = False
        self._set_averaging_approach()
        self._set_start_end_dates()
        self._datenum = self._get_time_series()
        self._forecast_hour = self.read_nc('forecast_hour', indices='[:]')
        if self.needs_averaging_to_the_half_hour:
            self._set_averaging_weights()
            #self._adjust_datenum()
        self._start_averaging = False
        
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
            #self.fillmask_cof = None
        self.romsgrd = romsgrd
        self.datain = np.ma.empty(self.lon().shape)
        if self.needs_averaging_to_hour:
            self.previous_in = np.ma.empty(self.lon().shape)
        elif self.needs_averaging_to_the_half_hour:
            self.datatmp = np.ma.empty(self.lon().shape)
        self.dataout = np.ma.empty(self.romsgrd.lon().size)
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
    
    
    def _set_averaging_approach(self):
        ''' 
        Order: call after self._check_product_description()
        '''
        option_1 = ('Forecast', 'Analysis')
        option_2 = ('1-hour Average', 'Average (reference date/time to valid date/time)')
        if self.product in option_1 :
            print '------ product "%s" hence using *valid_date_time*' %self.product
            self.needs_averaging_to_hour = False
            self.needs_averaging_to_the_half_hour = True
        elif self.product in option_2:
            print '------ product "%s" hence using *valid_date_time_range*' %self.product
            self.needs_averaging_to_hour = True
            self.needs_averaging_to_the_half_hour = False
        else:
            raise Exception, 'Undefined_product'
        return self
    
    
    def _set_averaging_weights(self):
        ''' 
        Order: call after self._check_product_description()
        '''
        basetime = dt.date2num(self.time_array[0])
        # Get nearest two indices
        inds = np.argsort(np.abs(self._datenum - basetime))[:2]
        delta = np.diff(self._datenum[inds.min():inds.max()+1])
        diff1 = np.diff([basetime, self._datenum[inds.max()]])
        #print diff1, delta
        self.time_avg_weights = np.array([diff1, delta - diff1])
        self.time_avg_weights /= delta
        return self
    
    
    def print_weights(self):
        '''
        '''
        try:
            print '------ averaging weights for *%s* product: %s' %(self.product,
                                                                    self.time_avg_weights)
        except Exception:
            print '------ no averaging weights for *%s* product' %self.product
        return self
    
    
    def _adjust_datenum(self):
        '''
        '''
        np.add(self._datenum[:-1] * self.time_avg_weights[0],
               self._datenum[1:]  * self.time_avg_weights[1], out=self._datenum[:-1])
        self._datenum[-1] = self._datenum[-2] + np.diff(self._datenum[:2])
    
    
    def _get_data_time_average(self):
        '''
        Order: call after self._set_averaging_weights()
        '''
        #print self.time_avg_weights[0], self.time_avg_weights[1]
        np.add(self.time_avg_weights[0] * self.datatmp,
               self.time_avg_weights[1] * self.datain, out=self.datain)
        return self
    
    
    '''----------------------------------------------------------------------------------
    '''
    
    def _read_cfsr_frc(self, var, ind):
        ''' Read CFSR forcing variable (var) at record (ind)
        '''
        return self.read_nc(var, '[' + str(ind) + ']')[::-1]
    
    def _get_cfsr_data(self, varname):
        ''' Get CFSR data with explicit variable name
        '''
        return self._read_cfsr_frc(varname, self.tind)
    
    def _get_cfsr_datatmp(self):
        ''' Get CFSR data with implicit variable name
        '''
        #print self.tind, self.tind-1
        self.datatmp[:] = self._read_cfsr_frc(self.varname, self.tind-1)
        return self
    
    
    def get_cfsr_data(self):
        ''' Get CFSR data with implicit variable name
        '''
        self.datain[:] = self._read_cfsr_frc(self.varname, self.tind)
        if self.needs_averaging_to_the_half_hour:
	    #print 'ssssss'
            self._get_cfsr_datatmp()
            self._get_data_time_average()
        return self
    
        
    def get_cfsr_data_previous(self):
        ''' Get CFSR data with implicit variable name
        '''
        self.previous_in[:] = self._read_cfsr_frc(self.varname, self.tind - 1)
        return self
    
    def cfsr_data_subtract_averages(self):
        ''' Extract one hour average from accumulated averages
        '''
        self.datain *= self.forecast_hour()
        self.previous_in *= (self.forecast_hour() - 1)
        self.datain -= self.previous_in
        return self
        
    
    def _check_for_nans(self, message=None):
        '''
        '''
        flat_mask = self.romsgrd.maskr().ravel()
        assert not np.any(np.isnan(self.dataout[np.nonzero(flat_mask)])
                          ), 'Nans in self.dataout sea points; hints: %s' %message
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
    
    
    
    
    
    
    
    
    def _set_start_end_dates(self):
        '''
        '''
        if self.needs_averaging_to_hour:
            self._start_date = self.read_nc('valid_date_time_range', '[0]')
            self._end_date = self.read_nc('valid_date_time_range', '[-1]')
        else:
            self._start_date = self.read_nc('valid_date_time', '[0]')
            self._end_date = self.read_nc('valid_date_time', '[-1]')
        return self
    
    
    def datenum(self):
        return self._datenum
    
    
    def set_date_index(self, dt):
        if self.needs_averaging_to_hour:
            try:
	        #print self.datenum().min(), self.datenum().max()
                dt_bool = np.isclose(self.datenum(), dt, rtol=1e-10, atol=1e-8)
                self.tind = np.nonzero(dt_bool)[0][0]
            except Exception:
                raise Exception, 'dt out of range in CFSR file'
            else:
                return self
        else:
            tind = np.argsort(np.abs(self._datenum - dt))[0]
            if self.tind == tind:
                self.tind += 1
            else:
                self.tind = tind
        return self
    
    
    def _date_from_string(self, date):
        '''
        Convert CFSR 'valid_date_time' to datenum
        Input: date : ndarray (e.g., "'2' '0' '1' '0' '1' '2' '3' '1' '1' '8'")
        '''
        assert (isinstance(date, np.ndarray) and
                date.size == 10), 'date must be size 10 ndarray'
        return dt.datetime.datetime(np.int(date.tostring()[:4]),
                                    np.int(date.tostring()[4:6]),
                                    np.int(date.tostring()[6:8]),
                                    np.int(date.tostring()[8:10]))
	
	
    def _get_time_series(self):
        '''
        '''
        if self.needs_averaging_to_hour:
            date_str = self._date_from_string(self._start_date[1])
            date_str -= dt.datetime.timedelta(minutes=30)
            date_end = self._date_from_string(self._end_date[1])
            date_end -= dt.datetime.timedelta(minutes=30)
        else:
            date_str = self._date_from_string(self._start_date)
            date_end = self._date_from_string(self._end_date)
        datelength = self.read_dim_size('time')
        #print 'date_str, date_end, datelen', date_str, date_end, datelen
        datenum = np.linspace(dt.date2num(date_str), dt.date2num(date_end), datelength)
        return datenum
        
        
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

    def get_delaunay_tri(self):
        '''
        '''
        self.points_all = np.copy(self.points)
        self.tri_all = sp.Delaunay(self.points_all)
        self.points = np.array([self.points[:,0].flat[self.cfsrmsk.cfsr_ball],
                                self.points[:,1].flat[self.cfsrmsk.cfsr_ball]]).T            
        self.tri = sp.Delaunay(self.points)
        return self
    
    
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







    def start_averaging(self):
        '''
        Ensure we start when *forecast_hour* == 1
        '''
        if self._forecast_hour[self.tind] == 1 or self._start_averaging:
            self._start_averaging = True
            return True
        else:
            return False


    def forecast_hour(self):
        ''' Return the current forecast hour
        '''
        return self._forecast_hour[self.tind]

    def check_vars_for_downscaling(self, var_instances):
        '''Loop over list of grids to find which have
           dimensions different from self
        '''
        for ind, each_grid in enumerate(var_instances):
            if np.any(each_grid.metrics != self.metrics):
                each_grid.to_be_downscaled = True
        return self












class CfsrPrate(CfsrDataHourly):
    '''CFSR Precipitation rate class (inherits from CfsrData class)
       Responsible for one variable: 'prate'
    '''
    def __init__(self, cfsr_dir, prate_file, masks, romsgrd, time_array):
        super(CfsrPrate, self).__init__(cfsr_dir + prate_file[0], prate_file[1], 'CFSR',
                                        romsgrd, time_array, masks=masks)
        #self._set_start_end_dates()
        #self.select_mask(masks)
        #self.get_fillmask_cof()
        #self._maskr = np.ones(self.lon().shape)

    def convert_cmday(self):
        datain = self.datain
        self.datain[:] = ne.evaluate('datain * 86400 * 0.1')
        #self.datain *= 86400 # to days
        #self.datain *= 0.1 # to cm
        return self


class CfsrSST(CfsrDataHourly):
    '''CFSR SST class (inherits from CfsrData class)
       Responsible for one variable: 'SST'
    '''
    def __init__(self, cfsr_dir, sst_file, masks, romsgrd, time_array):
        super(CfsrSST, self).__init__(cfsr_dir + sst_file[0], sst_file[1], 'CFSR',
                                      romsgrd, time_array, masks=masks)
        self.print_weights()
        #self.get_fillmask_cof()
       
       
class CfsrRadlw(CfsrDataHourly):
    '''CFSR Outgoing longwave radiation class (inherits from CfsrData class)
       Responsible for two ROMS bulk variables: 'radlw' and 'radlw_in'
    '''
    def __init__(self, cfsr_dir, radlw_file, masks, romsgrd, time_array):
        super(CfsrRadlw, self).__init__(cfsr_dir + radlw_file['shflux_LW_down'][0],
                                                   radlw_file['shflux_LW_down'][1], 'CFSR',
                                                   romsgrd, time_array, masks=masks)
        self.down_varname = radlw_file['shflux_LW_down'][1]
        self.up_varname = radlw_file['shflux_LW_up'][1]
        #self.select_mask(masks)
        #self.get_fillmask_cof()
        self.radlw_datain = np.ma.empty(self.datain.shape)
        self.radlw_in_datain = np.ma.empty(self.datain.shape)
        self.radlw_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.radlw_in_dataout = np.ma.empty(self.romsgrd.lon().shape)
        self.numvars = 2
    
    def _get_radlw(self, lw_up_datain):
        # First, get radlw
        radlw_in_datain = self.datain
        self.radlw_datain[:] = ne.evaluate('lw_up_datain - radlw_in_datain')
        #self.radlw_datain -= self.radlw_in_datain
        
    def _get_radlw_in(self, sst_datain):
        # Second, get radlw_in
        eps, stefan = self.eps, self.Stefan
        sst_datain[:] = ne.evaluate('sst_datain**4 * eps * stefan')
        #sst_datain **= 4
        #sst_datain *= self.eps
        #sst_datain *= self.Stefan
        radlw_datain = self.radlw_datain
        self.radlw_in_datain[:] = ne.evaluate('-(radlw_datain - sst_datain)')
    
    def get_cfsr_radlw_and_radlw_in(self, lw_up_datain, sst_datain):
        self._get_radlw(lw_up_datain)
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



class CfsrRadlwUp(CfsrDataHourly):
    '''
    '''
    def __init__(self, cfsr_dir, radlw_file, masks, romsgrd, time_array):
        super(CfsrRadlwUp, self).__init__(cfsr_dir + radlw_file['shflux_LW_up'][0],
                                                   radlw_file['shflux_LW_up'][1], 'CFSR',
                                                   romsgrd, time_array, masks=masks)
        


        
      
class CfsrRadSwDown(CfsrDataHourly):
    '''CFSR Outgoing longwave radiation class (inherits from CfsrDataHourly class)
       Responsible for one ROMS bulk variable: 'radsw'
    '''
    def __init__(self, cfsr_dir, radsw_file, masks, romsgrd, time_array):
        super(CfsrRadSwDown, self).__init__(cfsr_dir + radsw_file[0], radsw_file[1],'CFSR',
                                                   romsgrd, time_array, masks=masks)


class CfsrRadSwUp(CfsrDataHourly):
    '''
    '''
    def __init__(self, cfsr_dir, radlw_file, masks, romsgrd, time_array):
        super(CfsrRadSwUp, self).__init__(cfsr_dir + radlw_file[0], radlw_file[1],
                                          'CFSR', romsgrd, time_array, masks=masks)





class CfsrRhum(CfsrDataHourly):
    '''CFSR relative humidity class (inherits from CfsrData class)
       Responsible for one ROMS bulk variable: 'rhum'
    '''
    def __init__(self, cfsr_dir, rhum_file, masks, romsgrd, time_array):
        super(CfsrRhum, self).__init__(cfsr_dir + rhum_file[0], rhum_file[1], 'CFSR',
                                       romsgrd, time_array, masks=masks)
        self.print_weights()
        #self.get_fillmask_cof()
        

class CfsrQair(CfsrDataHourly):
    '''CFSR qair class (inherits from CfsrData class)
       Responsible for one variable: 'qair'
    '''
    def __init__(self, cfsr_dir, qair_file, masks, romsgrd, time_array):
        super(CfsrQair, self).__init__(cfsr_dir + qair_file[0], qair_file[1], 'CFSR',
                                       romsgrd, time_array, masks=masks)
        self.print_weights()
    
        

class CfsrSat(CfsrDataHourly):
    '''CFSR surface air temperature class (inherits from CfsrData class)
       Responsible for one ROMS bulk variable: 'tair'
    '''
    def __init__(self, cfsr_dir, sat_file, masks, romsgrd, time_array):
        super(CfsrSat, self).__init__(cfsr_dir + sat_file[0], sat_file[1], 'CFSR',
                                      romsgrd, time_array, masks=masks)
        self.print_weights()

    
        

class CfsrSap(CfsrDataHourly):
    '''CFSR surface air pressure class (inherits from CfsrData class)
       Responsible for one variable: 'qair'
    '''
    def __init__(self, cfsr_dir, sap_file, masks, romsgrd, time_array):
        super(CfsrSap, self).__init__(cfsr_dir + sap_file[0], sap_file[1], 'CFSR',
                                      romsgrd, time_array, masks=masks)
        self.print_weights()
        


class CfsrWspd(CfsrDataHourly):
    '''CFSR wind speed class (inherits from CfsrData class)
       Responsible for four variables: 'uspd', 'vspd', 'sustr' and 'svstr'
       Requires: 'qair', sap' and 'rhum'
    '''
    def __init__(self, cfsr_dir, wspd_file, masks, romsgrd, time_array):
        super(CfsrWspd, self).__init__(cfsr_dir + wspd_file['uspd'][0],
                                                  wspd_file['uspd'][1], 'CFSR',
                                                  romsgrd, time_array, masks=masks)
        #self.print_weights()
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
        kelvin = self.Kelvin # convert from K to C
        sat_data[:] = ne.evaluate('sat_data - kelvin')
        sap_data[:] = ne.evaluate('sap_data * 0.01') # convert from Pa to mb
        # Smith etal 1988
        self._rair_datain[:] = airsea.air_dens(sat_data, rhum_data, sap_data, qair_data)
        self.ustrs_datain[:], self.vstrs_datain[:] = airsea.stresstc(self.wspd_datain,
                                                                     self.uwnd_datain,
                                                                     self.vwnd_datain,
                                                                           sat_data,
                                                                     self._rair_datain)

    def _get_wspd(self):
        self.uwnd_datain[:] = self._get_cfsr_data(self.uwnd_varname)
        self.vwnd_datain[:] = self._get_cfsr_data(self.vwnd_varname)
        np.hypot(self.uwnd_datain, self.vwnd_datain, out=self.wspd_datain)

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
        self._interp2romsgrd()._check_for_nans('Nans here could indicate')
        self.ustrs_dataout[:] = self.dataout.reshape(roms_shape)
        self.datain[:] = self.vstrs_datain
        self._interp2romsgrd()._check_for_nans()
        self.vstrs_dataout[:] = self.dataout.reshape(roms_shape)
        return self






if __name__ == '__main__':
    
    '''
    make_pycfsr2frc_one-hourly

    Prepare one-hourly interannual ROMS surface forcing with, e.g., CFSv2 data (ds094.0) from
    
      http://rda.ucar.edu/pub/cfsr.html
      
      Concatenate with, e.g.:
          ncrcat rh2m.gdas.199912.grb2.nc rh2m.gdas.20????.grb2.nc tmp.nc
      and compress output with:
          nc3tonc4 tmp.nc REL_HUM_2000s_1hr.nc
    
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
    
    
    Evan Mason, IMEDEA, 2014
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # True for bulk forcing file, else standard dQdSTT forcing file
    bulk = True # variables ()

    
    # CFSR information_________________________________
    
    cfsr_version = 'ds094.1_1hourly'
    #cfsr_version = 'ds093.1_1hourly'
    
    #domain = 'S0_N50_W-50_E44'
    #domain = 'S0_N60_W-50_E44'
    domain = 'S27_N48_W-16_E20'
    #domain = 'S21_N51_W-30_E45'
    
    #cfsr_dir = '/shared/emason/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)
    #cfsr_dir = '/marula/emason/data/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)
    #cfsr_dir = '/shared/emason/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)
    #cfsr_dir = '/marula/emason/data/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)
    cfsr_dir = '/shared/emason/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)
    
    # Filenames and variable names of required CFSR variables
    # Note that these files have been prepared by concatenating the files 
    # dowloaded from CFSR using ncrcat
    if cfsr_version in 'ds093.1_1hourly':
        SSS_file = ('SALT_1hr.nc',  'SALTY_L160_Avg_1')
        swflux_file = ('EMP_6hr.nc',  'EMNP_L1_Avg_1')
        prate_file = ('PRATE_1hr.nc',  'PRATE_L1_Avg_1')
        shflux_SW_down_file = ('DOWN_SW_1hr.nc',  'DSWRF_L1_Avg_1')
        shflux_SW_up_file = ('UP_SW_1hr.nc',  'USWRF_L1_Avg_1')
        shflux_LW_down_file = ('DOWN_LW_1hr.nc',  'DLWRF_L1_Avg_1')
        shflux_LW_up_file = ('UP_LW_1hr.nc',  'ULWRF_L1_Avg_1')
        #shflux_LATENT_HEAT_file = ('HEAT_FLUXES/',  'LHTFL_L1_Avg_1')
        #shflux_SENSIBLE_HEAT_file = ('HEAT_FLUXES/',  'SHTFL_L1_Avg_1')
        sustr_file = ('WIND_1hr.nc',  'U_GRD_L103')
        svstr_file = ('WIND_1hr.nc',  'V_GRD_L103')
        SST_file = ('SST_1hr.nc',  'TMP_L1')
        sat_file = ('SAT_1hr.nc',  'TMP_L103')
        sap_file = ('PRESS_SFC_1hr.nc',  'PRES_L1')
        qair_file = ('SPEC_HUM_1hr.nc',  'SPF_H_L103')
        rel_hum_file = ('REL_HUM_1hr.nc',   'R_H_L103')
    
        # Filenames of masks for the different grids
        # Note: the filenames below are symbolics links to the relevant CFSv2 land cover files
        if domain in 'S0_N60_W-50_E44':
            mask_one = ('LANDMASK_9x15.nc', 'LAND_L1')
            mask_two = ('LANDMASK_11x19.nc', 'LAND_L1')
            mask_three = ('LANDMASK_43x73.nc', 'LAND_L1')
            mask_four = ('LANDMASK_68x116.nc', 'LAND_L1')
        elif domain in 'S21_N51_W-30_E45':
            mask_one = ('LANDMASK_12x31.nc', 'LAND_L1')
            mask_two = ('LANDMASK_16x41.nc', 'LAND_L1')
            mask_three = ('LANDMASK_61x151.nc', 'LAND_L1')
            mask_four = ('LANDMASK_96x241.nc', 'LAND_L1')
        else:
            raise Exception
        
        
    elif cfsr_version in 'ds094.1_1hourly':
        SSS_file = ('SALT_1hr.nc',  'SALTY_L160_Avg_1')
        swflux_file = ('EMP_6hr.nc',  'EMNP_L1_Avg_1')
        prate_file = ('PRATE_1hr.nc',  'PRATE_L1_Avg_1')
        shflux_SW_down_file = ('DOWN_SW_1hr.nc',  'DSWRF_L1_Avg_1')
        shflux_SW_up_file = ('UP_SW_1hr.nc',  'USWRF_L1_Avg_1')
        shflux_LW_down_file = ('DOWN_LW_1hr.nc',  'DLWRF_L1_Avg_1')
        shflux_LW_up_file = ('UP_LW_1hr.nc',  'ULWRF_L1_Avg_1')
        #shflux_LATENT_HEAT_file = ('ds094.0_heat_fluxes.nc',  'LHTFL_L1_Avg_1')
        #shflux_SENSIBLE_HEAT_file = ('ds094.0_heat_fluxes.nc',  'SHTFL_L1_Avg_1')
        sustr_file = ('WIND_1hr.nc',  'U_GRD_L103')
        svstr_file = ('WIND_1hr.nc',  'V_GRD_L103')
        SST_file = ('SST_1hr.nc',  'TMP_L1')
        sat_file = ('SAT_1hr.nc',  'TMP_L103')
        sap_file = ('PRESS_SFC_1hr.nc',  'PRES_L1')
        qair_file = ('SPEC_HUM_1hr.nc',  'SPF_H_L103')
        rel_hum_file = ('REL_HUM_1hr.nc',   'R_H_L103')
    
        # Filenames of masks for the different grids
        # Note: the filenames below are symbolics links to the relevant CFSv2 land cover files
        mask_one = ('LANDMASK_9x15.nc', 'LAND_L1')
        mask_two = ('LANDMASK_22x37.nc', 'LAND_L1')
        mask_three = ('LANDMASK_73x43.nc', 'LAND_L1')
        mask_four = ('LANDMASK_103x179.nc', 'LAND_L1')
    
    
    cfsr_masks = OrderedDict([('mask_one', mask_one),
                              ('mask_two', mask_two),
                              ('mask_three', mask_three),
                              ('mask_four', mask_four)])

    
    
    # ROMS configuration information_________________________________
    
    #roms_dir = '/home/emason/runs2012_tmp/MedSea5_R2.5/'
    #roms_dir = '/marula/emason/runs2012/MedSea5_intann_monthly/'
    #roms_dir = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir = '/Users/emason/toto/'
    #roms_dir = '/marula/emason/runs2013/cb_3km_2013_intann/'
    #roms_dir  = '/marula/emason/runs2013/AlbSea_1pt25/'
    #roms_dir    = '/marula/emason/runs2013/cart500/'
    #roms_dir = '/marula/emason/runs2012/MedSea_Romain/'
    #roms_dir     = '/marula/emason/runs2014/MedCan5/'
    #roms_dir     = '/marula/emason/runs2014/NA75_IA/'
    #roms_dir     = '/marula/emason/runs2014/nwmed5km/'
    #roms_dir     = '/marula/emason/runs2014/AlbSea175/'
    roms_dir     = '/marula/emason/runs2015/AlbSea500/'
    #roms_dir = './'
    
    #roms_grd = 'grd_MedSea5_R2.5.nc'
    #roms_grd    = 'grd_MedSea5.nc'
    #roms_grd = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd = 'grd_nwmed_2km.nc'
    #roms_grd = 'roms_grd.nc'
    #roms_grd = 'cb_2009_3km_grd_smooth.nc'
    #roms_grd    = 'grd_AlbSea_1pt25.nc'
    #roms_grd    = 'grd_cart500.nc'
    #roms_grd    = 'roms_grd_wmed_longterm.nc'
    #roms_grd     = 'grd_MedCan5.nc'
    #roms_grd = 'grd_AlbSea175.nc'
    #roms_grd = 'grd_nwmed5km_NARROW_STRAIT.nc'
    roms_grd = 'grd_AlbSea500.nc'
    
    # Forcing file
    #frc_filename = 'frc_MedSea5_test.nc' # ini filename
    #frc_filename = 'frc_MedSea5_1985010100_new.nc'
    #frc_filename = 'frc_MedSea5_1985010100_64bit.nc'
    #frc_filename = 'blk_MedSea5_1984123001.nc'
    #frc_filename = 'blk_NA75_1984123001.nc'
    #frc_filename = 'frc_CFSR_NA_7pt5km.nc'
    #frc_filename = 'frc_CFSR_NA_7pt5km_UPDATE.nc'
    #frc_filename = 'frc_2013_cb3km_CFSR_UPDATE.nc'
    #frc_filename = 'frc_AlbSea_1pt25_CFSR_20030101.nc'
    #frc_filename = 'frc_cart500.nc'
    #frc_filename = 'blk_nwmed5_1992-1993.nc'
    #frc_filename = 'blk_nwmed5_1994-1995.nc'
    #frc_filename = 'blk_nwmed5_1996-1997.nc' 
    frc_filename = 'blk_AlbSea500.nc'

    # Variable XXX_time/blk_time will be zero at this date
    day_zero = '19850101' # string with format YYYYMMDDHH
    #day_zero = '20060101' # string with format YYYYMMDDHH
    #day_zero = '20140101' # string with format YYYYMMDDHH
    
    # Modify filename
    if '_1hr' not in frc_filename:
        frc_filename = frc_filename.replace('.nc', '_1hr.nc')
    
    
    # True if the frc file being prepared is for a downscaled simulation
    #downscaled = False
    #if downscaled:
        # Point to parent directory, where make_pycfsr2frc_one-hourly expects to find 
        # start_date.mat (created by set_ROMS_interannual_start.py)
        #par_dir = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
        #par_dir = '/marula/emason/runs2012/MedSea5_intann_monthly/'

    
    # Start and end dates of the ROMS simulation
    # must be strings, format 'YYYYMMDDHH'
    #start_date = '1985010100'
    #end_date   = '1987102800'
    #start_date = '1992043000'
    #end_date   = '1992060123'
    #start_date = '2011010100'
    #end_date   = '2012113018'
    #start_date = '2006032400'
    #end_date   = '2006060600'
    #start_date = '2007022300'
    #end_date   = '2007022500'
    #start_date = '2007062300'
    #end_date   = '2007100818'
    #start_date = '1995111500'
    #end_date = '1991111600'
    #end_date   = '1998021500'
    
    start_date = '2013123000'
    
    #end_date   = '2014010100'
    end_date   = '2014053118'

    
    cycle_length = np.float(0)


    # Option for river runoff climatology
    #   Note, a precomputed *coast_distances.mat* must be available
    #   in roms_dir; this is computed using XXXXXX.py
    add_dai_runoff = True # True or False
    if add_dai_runoff:
        dai_file = '/home/emason/matlab/runoff/dai_runoff_mon_-180+180.nc'
        #dai_file = '/home/emason/matlab/runoff/dai_runoff_mon_0_360.nc'
        dai_ind_min, dai_ind_max = 999, 999 # intentionally give unrealistic initial values
    
    # Interpolation / processing parameters_________________________________
    
    #balldist = 100000. # distance (m) for kde_ball (should be 2dx at least?)
    
    # Filling of landmask options
    # Set to True to extrapolate sea data over land
    #winds_fillmask = False # 10 m (False recommended)
    #sat_fillmask  = True # 2 m (True recommended)
    #rhum_fillmask = True # 2 m (True recommended)
    #qair_fillmask = True # 2 m (True recommended)
    
    #windspd_fillmask = True # surface (True recommended)
    fillmask_radlw, fillmask_radlw_in = True, True
    
    
    sigma_params = dict(theta_s=None, theta_b=None, hc=None, N=None)

    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    timing_warning = 'bulk_time at tind is different from roms_day'
    
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
                ('SST', SST_file),
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
                  ('shflux_LH', shflux_LATENT_HEAT_file),
                  ('shflux_SH', shflux_SENSIBLE_HEAT_file)])),
               ('dQdSST', OrderedDict([
                  ('uspd', sustr_file),
                  ('vspd', svstr_file),
                  ('SST', SST_file),
                  ('sat', sat_file),
                  ('rel_hum', rel_hum_file),
                  ('sap', sap_file),
                  ('qair', qair_file)]))])
                  
    
    
    # Initialise an AirSea object
    airsea = AirSea()
    
    dtstrdt = dt.datetime.datetime(np.int(start_date[:4]),
                       np.int(start_date[4:6]),
                       np.int(start_date[6:8]),
                       np.int(start_date[8:10]), 30)
    
    dtenddt = dt.datetime.datetime(np.int(end_date[:4]),
                       np.int(end_date[4:6]),
                       np.int(end_date[6:8]),
                       np.int(end_date[8:10]), 30)
    
    # Number of records at six hourly frequency
    delta = dt.datetime.timedelta(hours=1)
    #numrec = dt.num2date(dt.drange(dtstrdt, dtenddt, delta))

    dtstr, dtend = dt.date2num(dtstrdt), dt.date2num(dtenddt)
        
    day_zero = dt.datetime.datetime(int(day_zero[:4]), int(day_zero[4:6]), int(day_zero[6:]))
    day_zero = dt.date2num(day_zero)
    
    time_array = dt.num2date(dt.drange(dtstrdt, dtenddt, delta))
    
    
    #if downscaled:
        #inidate = io.loadmat(par_dir + 'start_date.mat')
        #deltaday0 = dtstr - inidate['start_date']
    
    
    # Instantiate a RomsGrid object
    numrec = None
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
    romsgrd.create_frc_nc(''.join((roms_dir, frc_filename)), start_date, end_date, numrec,
                          cycle_length, 'make_pycfsr2frc_one-hourly', bulk)
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=True).make_kdetree()
    

    
    # Get all CFSR mask and grid sizes
    # Final argument is kde ball distance in metres (should be +/- 2.5 * dx)
    # but by calling mask_1.check_ball() to make a figure...
    if cfsr_version in 'ds093.1_1hourly':
        mask_1 = CfsrMask(cfsr_dir, cfsr_masks['mask_one'], romsgrd, 650000)
        mask_2 = CfsrMask(cfsr_dir, cfsr_masks['mask_two'], romsgrd, 110000)
        mask_3 = CfsrMask(cfsr_dir, cfsr_masks['mask_three'], romsgrd, 125000)
        mask_4 = CfsrMask(cfsr_dir, cfsr_masks['mask_four'], romsgrd, 120000)
        
    elif cfsr_version in 'ds094.1_1hourly':
        mask_1 = CfsrMask(cfsr_dir, cfsr_masks['mask_one'], romsgrd, 250000)
        mask_2 = CfsrMask(cfsr_dir, cfsr_masks['mask_two'], romsgrd, 650000)
        mask_3 = CfsrMask(cfsr_dir, cfsr_masks['mask_three'], romsgrd, 250000)
        mask_4 = CfsrMask(cfsr_dir, cfsr_masks['mask_four'], romsgrd, 150000)

    
    # List of masks
    masks = [mask_1, mask_2, mask_3, mask_4]
    

    

    

    for cfsr_key in cfsr_files.keys():
        
        print '\nProcessing variable key:', cfsr_key.upper()        
        

        # FIRST initialise *dQdSST* classes if defined
        
        if cfsr_key in 'shflux':
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file['shflux_SW_down'][0])), 'CFSR')
        
        elif cfsr_key in 'dQdSST': # used for dQdSST
            # Using uspd/sustr here cos has highest resolution (0.3 deg.)
            cfsrgrd = CfsrGrid(''.join((cfsr_dir, cfsr_file['sustr'][0])), 'CFSR')
        
        
        # SECOND initialise *bulk* classes if defined
        
        elif cfsr_key in 'prate': # Precipitation rate (used for bulk)
            cfsr_prate = CfsrPrate(cfsr_dir, cfsr_files['prate'], masks, romsgrd, time_array)
            
            cfsr_prate.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_prate.child_contained_by_parent(romsgrd)
            cfsr_prate.make_kdetree()
            cfsr_prate.get_delaunay_tri()
            
            
        elif cfsr_key in 'radlw': # Outgoing longwave radiation (used for bulk)
            cfsr_sst = CfsrSST(cfsr_dir, cfsr_files['wspd']['SST'], masks, romsgrd, time_array)
            cfsr_radlw = CfsrRadlw(cfsr_dir, cfsr_files['radlw'], masks, romsgrd, time_array)
            
            
            cfsr_radlw_up = CfsrRadlwUp(cfsr_dir, cfsr_files['radlw'], masks, romsgrd, time_array)
            
            supp_vars = [cfsr_radlw_up, cfsr_sst]
            cfsr_radlw.check_vars_for_downscaling(supp_vars)
            for supp_var in supp_vars:
                if supp_var.to_be_downscaled:
                    supp_var.proj2gnom(ignore_land_points=False, M=romsgrd.M)
                    supp_var.get_delaunay_tri()
                    cfsr_radlw.needs_all_point_tri = True
            
            cfsr_radlw.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_radlw.child_contained_by_parent(romsgrd)
            cfsr_radlw.make_kdetree()
            cfsr_radlw.get_delaunay_tri()
            
            
        elif cfsr_key in 'radsw': # used for bulk
            
            cfsr_radsw_down = CfsrRadSwDown(cfsr_dir, cfsr_files['radsw']['shflux_SW_down'], masks, romsgrd, time_array)
            cfsr_radsw_down.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_radsw_down.child_contained_by_parent(romsgrd)
            cfsr_radsw_down.make_kdetree()
            cfsr_radsw_down.get_delaunay_tri()
            
            cfsr_radsw_up = CfsrRadSwUp(cfsr_dir, cfsr_files['radsw']['shflux_SW_up'], masks, romsgrd, time_array)
            cfsr_radsw_up.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_radsw_up.child_contained_by_parent(romsgrd)
            cfsr_radsw_up.make_kdetree()
            cfsr_radsw_up.get_delaunay_tri()
        
        elif cfsr_key in 'wspd': # used for bulk
            cfsr_rhum = CfsrRhum(cfsr_dir, cfsr_files['wspd']['rel_hum'], masks, romsgrd, time_array)
            cfsr_qair = CfsrQair(cfsr_dir, cfsr_files['wspd']['qair'], masks, romsgrd, time_array)
            cfsr_sat = CfsrSat(cfsr_dir, cfsr_files['wspd']['sat'], masks, romsgrd, time_array)
            cfsr_sap = CfsrSap(cfsr_dir, cfsr_files['wspd']['sap'], masks, romsgrd, time_array)
            cfsr_wspd = CfsrWspd(cfsr_dir, cfsr_files['wspd'], masks, romsgrd, time_array)
            
            supp_vars = [cfsr_rhum, cfsr_sat, cfsr_sap, cfsr_qair]
            cfsr_wspd.check_vars_for_downscaling(supp_vars)
            for supp_var in supp_vars:
                if supp_var.to_be_downscaled:
                    supp_var.proj2gnom(ignore_land_points=False, M=romsgrd.M)
                    supp_var.get_delaunay_tri()
                    cfsr_wspd.needs_all_point_tri = True
            
            cfsr_sat.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_sat.child_contained_by_parent(romsgrd)
            cfsr_sat.make_kdetree()
            cfsr_sat.get_delaunay_tri()
            
            cfsr_wspd.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_wspd.child_contained_by_parent(romsgrd)
            cfsr_wspd.make_kdetree()
            cfsr_wspd.get_delaunay_tri()
            
            
        else:
            raise Exception, 'Unknown key %s' %cfsr_key
  
  
             
        tind = 0
        
        # Loop over time
        for cfsr_num in time_array:
                
            cfsr_dt = dt.date2num(cfsr_num)
            
            # Day count to be stored as bulk_time
            roms_day = np.float32(cfsr_dt - day_zero)
                
                
            # Precipitation rate (bulk only)
            if 'prate' in cfsr_key:
                    
                cfsr_prate.set_date_index(cfsr_dt)
                cfsr_prate.get_cfsr_data()
                
                # We filter for first occurrence of forecast_hour == 1
                if cfsr_prate.needs_averaging_to_hour and cfsr_prate.forecast_hour() > 1:
                    
                    # For averages longer than one hour we need to get
                    # the hourly average by using the previous record
                    cfsr_prate.get_cfsr_data_previous()
                    cfsr_prate.cfsr_data_subtract_averages()
                    
                cfsr_prate.convert_cmday()
                cfsr_prate.interp2romsgrd()
                #cfsr_prate.check_interp()
                    
                # Get indices and weights for Dai river climatology
                if add_dai_runoff:
                    ind_min, ind_max, weights = romsgrd.get_runoff_index_weights(cfsr_dt)
                    # If condition so we don't read runoff data every iteration
                    if np.logical_or(ind_min != dai_ind_min, ind_max != dai_ind_max):
                        dai_ind_min, dai_ind_max = ind_min, ind_max
                        runoff1 = romsgrd.get_runoff(dai_file, dai_ind_min+1)
                        runoff2 = romsgrd.get_runoff(dai_file, dai_ind_max+1)
                    runoff = np.average([runoff1, runoff2], weights=weights, axis=0)
                    cfsr_prate.dataout += runoff.ravel()
                    
                cfsr_prate.dataout *= romsgrd.maskr().ravel()
                np.place(cfsr_prate.dataout, cfsr_prate.dataout < 0., 0)
                
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['prate'][tind] = cfsr_prate.dataout
                    nc.variables['bulk_time'][tind] = roms_day
                    if numrec is not None:
                        nc.variables['bulk_time'].cycle_length = np.float(numrec)
                    nc.sync()
            
            
            # Outgoing longwave radiation (bulk only)
            elif 'radlw' in cfsr_key:
                
                cfsr_sst.set_date_index(cfsr_dt)
                cfsr_radlw_up.set_date_index(cfsr_dt)
                cfsr_radlw.set_date_index(cfsr_dt)
                cfsr_sst.get_cfsr_data().fillmask()
                cfsr_radlw_up.get_cfsr_data()
                cfsr_radlw.get_cfsr_data()
                
                # We filter for first occurrence of forecast_hour == 1
                if cfsr_radlw.needs_averaging_to_hour and cfsr_radlw.forecast_hour() > 1:
                    
                    # For averages longer than one hour we need to get
                    # the hourly average by using the previous record
                    cfsr_radlw_up.get_cfsr_data_previous()
                    cfsr_radlw_up.cfsr_data_subtract_averages()
                    cfsr_radlw.get_cfsr_data_previous()
                    cfsr_radlw.cfsr_data_subtract_averages()
                    
                # Note that fillmask is called by cfsr_radlw.interp2romsgrd
                cfsr_radlw.get_cfsr_radlw_and_radlw_in(cfsr_radlw_up.datain,
                                                       cfsr_sst.datain)
                    
                cfsr_radlw.interp2romsgrd(fillmask_radlw, fillmask_radlw_in)
                
                cfsr_radlw.radlw_dataout *= romsgrd.maskr()
                cfsr_radlw.radlw_in_dataout *= romsgrd.maskr()
                    
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['radlw'][tind] = cfsr_radlw.radlw_dataout
                    nc.variables['radlw_in'][tind] = cfsr_radlw.radlw_in_dataout
                    assert nc.variables['bulk_time'][tind] == roms_day, timing_warning
                    nc.sync()
                #tind += 1; print 'remove me'

            
            # Net shortwave radiation (bulk only)
            elif 'radsw' in cfsr_key:
                    
                cfsr_radsw_down.set_date_index(cfsr_dt)
                cfsr_radsw_up.set_date_index(cfsr_dt)
                cfsr_radsw_down.get_cfsr_data()
                cfsr_radsw_up.get_cfsr_data()
                
                # We filter for first occurrence of forecast_hour == 1
                if cfsr_radsw_down.needs_averaging_to_hour and cfsr_radsw_down.forecast_hour() > 1:
                    
                    # For averages longer than one hour we need to get
                    # the hourly average by using the previous record
                    cfsr_radsw_down.get_cfsr_data_previous()
                    cfsr_radsw_down.cfsr_data_subtract_averages()
                    cfsr_radsw_up.get_cfsr_data_previous()
                    cfsr_radsw_up.cfsr_data_subtract_averages()

                cfsr_radsw_down.fillmask()
                cfsr_radsw_down.interp2romsgrd()
                cfsr_radsw_up.fillmask()
                cfsr_radsw_up.interp2romsgrd()
                #cfsr_radsw_down.check_interp()
                cfsr_radsw_down.dataout -= cfsr_radsw_up.dataout
                cfsr_radsw_down.dataout *= romsgrd.maskr().ravel()
                    
                np.place(cfsr_radsw_down.dataout, cfsr_radsw_down.dataout < 0., 0)
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['radsw'][tind] = cfsr_radsw_down.dataout
                    assert nc.variables['bulk_time'][tind] == roms_day, timing_warning
                    nc.sync()
                
                #    tind += 1; print 'remove me'
                

            # Wind speed (wspd, uwnd, vwnd) and stress (sustr, svstr) (bulk only)
            elif 'wspd' in cfsr_key:
                    
                cfsr_rhum.set_date_index(cfsr_dt)
                cfsr_sat.set_date_index(cfsr_dt)
                cfsr_sap.set_date_index(cfsr_dt)
                cfsr_qair.set_date_index(cfsr_dt)
                cfsr_wspd.set_date_index(cfsr_dt)
                
                # We filter for first occurrence of forecast_hour == 1
                assert not np.any([cfsr_rhum.needs_averaging_to_hour,
                                   cfsr_sat.needs_averaging_to_hour,
                                   cfsr_sap.needs_averaging_to_hour,
                                   cfsr_qair.needs_averaging_to_hour,
                                   cfsr_wspd.needs_averaging_to_hour]), 'These should all be forecasts'
                  
                cfsr_rhum.get_cfsr_data()
                cfsr_sat.get_cfsr_data()
                cfsr_sap.get_cfsr_data()
                cfsr_qair.get_cfsr_data()
                cfsr_wspd.get_cfsr_data()
                    
                cfsr_rhum.get_cfsr_data().fillmask()
                cfsr_sat.get_cfsr_data().fillmask()
                cfsr_sap.get_cfsr_data().fillmask()
                cfsr_qair.get_cfsr_data().fillmask()
                    
                for supp_var in supp_vars:
                    if supp_var.to_be_downscaled:
                        supp_var.data = horizInterp(supp_var.tri_all, supp_var.datain.ravel())(cfsr_wspd.points_all)
                        supp_var.data = supp_var.data.reshape(cfsr_wspd.lon().shape)
                            
                cfsr_wspd.get_winds(airsea, cfsr_rhum, cfsr_sat, cfsr_sap, cfsr_qair)
                cfsr_wspd.interp2romsgrd()
                cfsr_wspd.wspd_dataout *= romsgrd.maskr()
                    
                cfsr_wspd.uwnd_dataout[:], \
                cfsr_wspd.vwnd_dataout[:] = romsgrd.rotate(cfsr_wspd.uwnd_dataout,
                                                           cfsr_wspd.vwnd_dataout, sign=1)
                cfsr_wspd.uwnd_dataout[:,:-1] = romsgrd.rho2u_2d(cfsr_wspd.uwnd_dataout)
                cfsr_wspd.uwnd_dataout[:,:-1] *= romsgrd.umask()
                cfsr_wspd.vwnd_dataout[:-1] = romsgrd.rho2v_2d(cfsr_wspd.vwnd_dataout)
                cfsr_wspd.vwnd_dataout[:-1] *= romsgrd.vmask()
                    
                cfsr_wspd.ustrs_dataout[:], \
                cfsr_wspd.vstrs_dataout[:] = romsgrd.rotate(cfsr_wspd.ustrs_dataout,
                                                            cfsr_wspd.vstrs_dataout, sign=1)
                cfsr_wspd.ustrs_dataout[:,:-1] = romsgrd.rho2u_2d(cfsr_wspd.ustrs_dataout)
                cfsr_wspd.ustrs_dataout[:,:-1] *= romsgrd.umask()
                cfsr_wspd.vstrs_dataout[:-1] = romsgrd.rho2v_2d(cfsr_wspd.vstrs_dataout)
                cfsr_wspd.vstrs_dataout[:-1] *= romsgrd.vmask()
                    
                cfsr_rhum.datain *= 0.01
                cfsr_rhum.interp2romsgrd()
                cfsr_rhum.dataout *= romsgrd.maskr().ravel()
                    
                cfsr_sat.interp2romsgrd()
                cfsr_sat.dataout *= romsgrd.maskr().ravel()
                    
                with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                    nc.variables['rhum'][tind] = cfsr_rhum.dataout.reshape(romsgrd.lon().shape)
                    nc.variables['tair'][tind] = cfsr_sat.dataout.reshape(romsgrd.lon().shape)
                    nc.variables['wspd'][tind] = cfsr_wspd.wspd_dataout
                    nc.variables['uwnd'][tind] = cfsr_wspd.uwnd_dataout[:,:-1]
                    nc.variables['vwnd'][tind] = cfsr_wspd.vwnd_dataout[:-1]
                    nc.variables['sustr'][tind] = cfsr_wspd.ustrs_dataout[:,:-1]
                    nc.variables['svstr'][tind] = cfsr_wspd.vstrs_dataout[:-1]
                    assert nc.variables['bulk_time'][tind] == roms_day, timing_warning
                    nc.sync()
            
            #print tind
            tind += 1

                
                
                
                
                
                
                
                
