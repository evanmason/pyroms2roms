# -*- coding: utf-8 -*-
# %run py_mercator2roms.py

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

Create a ROMS boundary file based on Mercator data

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
from time import strptime

from py_roms2roms import vertInterp, horizInterp, bry_flux_corr, debug0, debug1, debug2
from py_roms2roms import ROMS, RomsGrid, WestGrid, EastGrid, NorthGrid, SouthGrid, RomsData





class MercatorData (RomsData):
    '''
    MercatorData class (inherits from RomsData class)
    '''
    def __init__(self, filenames, model_type, mercator_var, romsgrd, **kwargs):
        """
        Creates a new Mercator data object.
        
        Parameters
        ----------
        
        *filenames* : list of Mercator nc files.
        *model_type* : string specifying Mercator model.
        *romsgrd* : a `RomsGrid` instance.
        
        """
        super(MercatorData, self).__init__(filenames[0], model_type)
        self.romsfile = filenames
        self.vartype = mercator_var
        self.romsgrd = romsgrd
        self.var_dic = {'sossheig':'SSH',
                        'votemper':'TEMP', 'vosaline':'SALT',
                        'vozocrtx':'U', 'vomecrty':'V',
                        'ssh':'SSH',
                        'temperature':'TEMP', 'salinity':'SALT',
                        'u':'U', 'v':'V'}
        #print self.var_dic
        self._set_variable_type()
        try:
            self._lon = self.read_nc('nav_lon', indices='[:]')
            self._lat = self.read_nc('nav_lat', indices='[:]')
        except Exception:
            self._lon = self.read_nc('longitude', indices='[:]')
            self._lat = self.read_nc('latitude', indices='[:]')
            self._lon, self._lat = np.meshgrid(self._lon, self._lat)
        #print self._lon.shape
        #print self._lat.shape
        if kwargs.has_key('i0'):
            self.i0, self.i1 = kwargs['i0'], kwargs['i1']
            self.j0, self.j1 = kwargs['j0'], kwargs['j1']
        else:
            self.set_subgrid(romsgrd, k=50)
        
        try:
            self._depths = self.read_nc('deptht', indices='[:]')
        except Exception:
            self._depths = self.read_nc('depth', indices='[:]')
        try:
            self._time_count = self.read_nc('time_counter', indices='[:]')
        except Exception:
            self._time_count = self.read_nc('time', indices='[:]')
            
        self._set_data_shape()
        self._set_dates()
        self.datain = np.empty(self._data_shape, dtype=np.float64)
        if len(self._data_shape) == 2:
            self.dataout = np.ma.empty(self.romsgrd.lon().shape)
        else:
            tmp_shape = (self._depths.size, self.romsgrd.lon().shape[0],
                         self.romsgrd.lon().shape[1])
            self.datatmp = np.ma.empty(tmp_shape, dtype=np.float64).squeeze()
            self.dataout = np.ma.empty(self.romsgrd.mask3d().shape,
                                       dtype=np.float64).squeeze()
        self._set_maskr()
        self._set_angle()

        
    def lon(self):
        return self._lon[self.j0:self.j1, self.i0:self.i1]
    
    def lat(self):
        return self._lat[self.j0:self.j1, self.i0:self.i1]
    
    def maskr(self):
        return self._maskr[self.j0:self.j1, self.i0:self.i1]
    
    def maskr3d(self):
        return self._maskr3d[:, self.j0:self.j1, self.i0:self.i1]
    
    def angle(self):
        return self._angle[self.j0:self.j1, self.i0:self.i1]
    
    def depths(self):
        return self._depths
    
    def dates(self):
        return self._time_count
    
    
    def _set_data_shape(self):
        ssh_varnames = ('sossheig') # clumsy; needs to be added to if they invent more names
        #if kwargs.has_key('predefined_shp'):
            #self._data_shape = kwargs['predefined_shp']
            #if len(self._data_shape ) > 2:
                #self.dimtype = '3D'
            #else:
                #self.dimtype = '2D'
        #print 'self.varname', self.varname
        if self._depths.size == 1 or self.varname in ssh_varnames:
            _shp = (self.j1 - self.j0, self.i1 - self.i0)
            self._data_shape = _shp
            self.dimtype = '2D'
        else:
            _shp = (self._depths.size, self.j1 - self.j0, self.i1 - self.i0)
            self._data_shape = _shp
            self.dimtype = '3D'
        return self
    
    
    def _set_variable_type(self):
        self.varname = None
        for varname, vartype in zip(self.var_dic.keys(), self.var_dic.values()):
            #print self.romsfile
            #print varname, vartype
            #print self.list_of_variables()
            #print self.vartype
            if varname in self.list_of_variables() and vartype in self.vartype:
                self.varname = varname
                return self
        if self.varname is None:
            raise Exception # no candidate variable identified
    
    
    def _get_maskr(self, k=None):
        """
        Called by _set_maskr()
        """
        if '3D' in self.dimtype:
            self.k = k
            indices = '[0, self.k]'
        else:
            indices = '[0]'
        try:
            _mask = self.read_nc(self.varname, indices=indices).mask
        except Exception: # triggered when all points are masked
            _mask = np.ones(self.read_nc(self.varname, indices=indices).shape)
        _mask = np.asarray(_mask, dtype=np.int)
        _mask *= -1
        _mask += 1
        return _mask
    
        
    def _set_maskr(self):
        """
        Set the landsea mask (*self.maskr*) with same shape
        as input Mercator nc file. If a 3D variable, then additional
        attribute, self._maskr3d is set.
        """
        if '3D' in self.dimtype:
            self._maskr = self._get_maskr(k=0)
            self._set_maskr3d()
        else:
            self._maskr = self._get_maskr()
        return self
    
    
    def _set_maskr3d(self):
        """
        Called by _set_maskr()
        """
        _3dshp = (self._depths.size, self._lon.shape[0], self._lon.shape[1])
        #print _3dshp
        self._maskr3d = np.empty(_3dshp)
        for k in xrange(self._depths.size):
            self._maskr3d[k] = self._get_maskr(k=k)
        return self
        
        
    def _set_angle(self):
        '''
        Compute angles of local grid positive x-axis relative to east
        '''
        latu = np.deg2rad(0.5 * (self._lat[:,1:] + self._lat[:,:-1]))
        lonu = np.deg2rad(0.5 * (self._lon[:,1:] + self._lon[:,:-1]))
        dellat = latu[:,1:] - latu[:,:-1]
        dellon = lonu[:,1:] - lonu[:,:-1]
        dellon[dellon >  np.pi] = dellon[dellon >  np.pi] - (2. * np.pi)
        dellon[dellon < -np.pi] = dellon[dellon < -np.pi] + (2. * np.pi)
        dellon = dellon * np.cos(0.5 * (latu[:,1:] + latu[:,:-1]))

        self._angle = np.zeros_like(self._lat)
        ang_s = np.arctan(dellat / (dellon + np.spacing(0.4)))
        deli = np.logical_and(dellon < 0., dellat < 0.)
        ang_s[deli] = ang_s[deli] - np.pi

        deli = np.logical_and(dellon < 0., dellat >= 0.)
        ang_s[deli] = ang_s[deli] + np.pi
        ang_s[ang_s >  np.pi] = ang_s[ang_s >  np.pi] - np.pi
        ang_s[ang_s < -np.pi] = ang_s[ang_s <- np.pi] + np.pi

        self._angle[:,1:-1] = ang_s
        self._angle[:,0] = self._angle[:,1]
        self._angle[:,-1] = self._angle[:,-2]
        return self
    
    
    def get_variable(self, date):
        ind = (self._time_count == date).nonzero()[0][0]
        #print 'ind', ind
        try:
            with netcdf.Dataset(self.romsfile[ind]) as nc:
                if '3D' in self.dimtype:
                    self.datain[:] = np.ma.masked_array(
                        nc.variables[self.varname][0, :, self.j0:self.j1,
                                                         self.i0:self.i1].astype(np.float64)).data
                else:
                    self.datain[:] = np.ma.masked_array(
                        nc.variables[self.varname][0, self.j0:self.j1,
                                                      self.i0:self.i1].astype(np.float64)).data
        except Exception:
            with netcdf.Dataset(self.romsfile[0]) as nc:
                if '3D' in self.dimtype:
                    self.datain[:] = np.ma.masked_array(
                        nc.variables[self.varname][ind, :, self.j0:self.j1,
                                                           self.i0:self.i1].astype(np.float64)).data
                else:
                    self.datain[:] = np.ma.masked_array(
                        nc.variables[self.varname][ind, self.j0:self.j1,
                                                        self.i0:self.i1].astype(np.float64)).data
        return self
    
    
    def _set_time_origin(self):
        try:
            self._time_counter_origin = self.read_nc_att('time_counter', 'time_origin')
            ymd, hms = self._time_counter_origin.split(' ')
        except Exception:
            try:
                self._time_counter_origin = self.read_nc_att('time_counter', 'units')
                junk, junk, ymd, hms = self._time_counter_origin.split(' ')
                #print self._time_counter_origin
            except Exception:
                self._time_counter_origin = self.read_nc_att('time', 'units')
                junk, junk, ymd, hms = self._time_counter_origin.split(' ')
        #else: Exception
        #print self._time_counter_origin
        y, mo, d = ymd.split('-')
        h, mi, s = hms.split(':')
        
        #print y, mo, d, h, mi, s
        try:
            time_origin = datetime(int(y), strptime(mo,'%b').tm_mon, int(d),
                                   int(h), int(mi), int(s))
	except Exception:
	    time_origin = datetime(int(y), int(mo), int(d),
                                   int(h), int(mi), int(s))
        self.time_origin = plt.date2num(time_origin)
        return self
    
    
    def _set_dates(self):
        self._set_time_origin()
        #self._time_count = self.read_nc_mf('time_counter')
        
        with netcdf.Dataset(self.romsfile[0]) as nc:
            try:
                date_start = nc.variables['time_counter'][:]
            except Exception:
                date_start = nc.variables['time'][:]
        with netcdf.Dataset(self.romsfile[-1]) as nc:
            try:
                date_end = nc.variables['time_counter'][:]
            except Exception:
                date_end = nc.variables['time'][:]
        if 'second' in self._time_counter_origin:
            self._time_interval = 86400.
        elif 'hour' in self._time_counter_origin:
            self._time_interval = 24.
        #print 'date_start',date_start
        #print 'date_end',date_end
        #print 'self._time_interval',self._time_interval
        try:
            #print 'here 1'
            self._time_count = np.arange(date_start, date_end +
                                  self._time_interval, self._time_interval)
        except:
            #print 'here 2'
            self._time_count = np.arange(date_start[0], date_end[-1] + 
                                  self._time_interval, self._time_interval)
        #print 'date_start', date_start
        #print 'date_end', date_end
        #print '',
        #self._time_count = self.read_nc_mf('TIME')
        self._time_count /= self._time_interval #86400.
        self._time_count += self.time_origin
        return self
    
    
    def get_date(self, ind):
        return self._time_count[ind]
        
        
    def get_fillmask_cofs(self):
        if '3D' in self.dimtype:
            self.fillmask_cof_tmp = []
            for k in xrange(self._depths.size):
                try:
                    self.get_fillmask_cof(self.maskr3d()[k])
                    self.fillmask_cof_tmp.append(self.fillmask_cof)
                except Exception:
                    self.fillmask_cof_tmp.append(None)
            self.fillmask_cof = self.fillmask_cof_tmp
        else:
            self.get_fillmask_cof(self.maskr())
        return self
    
    
    def _fillmask(self, dist, iquery, igood, ibad, k=None):
        '''
        '''
        weight = dist / (dist.min(axis=1)[:,np.newaxis] * np.ones_like(dist))
        np.place(weight, weight > 1., 0.)
        if k is not None:
	    #print self.datain[k].shape
	    #print igood.shape
	    #print 'igood', igood
	    #print 'iquery', iquery
	    #print 'k', k
	    #print igood[:,0][iquery]
	    #print igood[:,1][iquery]
            #aaaaaa
            xfill = weight * self.datain[k][igood[:,0][iquery], igood[:,1][iquery]]
            xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)
            self.datain[k][ibad[:,0], ibad[:,1]] = xfill
        else:
            xfill = weight * self.datain[igood[:,0][iquery], igood[:,1][iquery]]
            xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)
            self.datain[ibad[:,0], ibad[:,1]] = xfill
        return self
    
    def fillmask(self):
        '''Fill missing values in an array with an average of nearest  
           neighbours
           From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        Order:  call after self.get_fillmask_cof()
        '''
        if '3D' in self.dimtype:
            for k in xrange(len(self.fillmask_cof)):
                if self.fillmask_cof[k] is not None:
                    dist, iquery, igood, ibad = self.fillmask_cof[k]
                    #print dist, iquery, igood, ibad
                    try:
                        self._fillmask(dist, iquery, igood, ibad, k=k)
                    except:
                        self.maskr3d()[k] = 0
            self._check_and_fix_deep_levels()
        else:
            dist, iquery, igood, ibad = self.fillmask_cof
            self._fillmask(dist, iquery, igood, ibad)
        return self
    
    
    def _check_and_fix_deep_levels(self):
        for k in xrange(self._depths.size):
            if np.sum(self.maskr3d()[k]) == 0:
                self.datain[k] = self.datain[k-1]
        return self
    
    
    def set_2d_depths(self):
        '''
        
        '''
        self._2d_depths = np.tile(-self.depths()[::-1],
                                  (self.romsgrd.h().size, 1)).T
        return self
    
    
    def set_3d_depths(self):
        '''
        
        '''
        self._3d_depths = np.tile(-self.depths()[::-1],
                                  (self.romsgrd.h().shape[1],
                                   self.romsgrd.h().shape[0], 1)).T
        return self
    
    
    def set_map_coordinate_weights(self, j=None):
        '''
        Order : set_2d_depths or set_3d_depths
        '''
        if j is not None:
            mercator_depths = self._3d_depths[:,j]
            roms_depths = self.romsgrd.scoord2z_r()[:,j]
        else:
            mercator_depths = self._2d_depths
            roms_depths = self.romsgrd.scoord2z_r()
        self.mapcoord_weights = self.romsgrd.get_map_coordinate_weights(
                                             roms_depths, mercator_depths)
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
    
    
    def vert_interp(self):
        '''
        Vertical interpolation using ndimage.map_coordinates()
          See vertInterp class in py_roms2roms.py
        Requires self.mapcoord_weights set by set_map_coordinate_weights()
        '''
        self._vert_interp = vertInterp(self.mapcoord_weights)
        return self
    
    
    def _interp2romsgrd(self, k=None):
        '''
        '''
        if k is not None:
            interp = horizInterp(self.tri, self.datain[k].flat[self.ball])
            interp = interp(self.romsgrd.points)
            try:
                self.datatmp[k] = interp
            except Exception:
                self.datatmp[k] = interp.reshape(self.romsgrd.maskr().shape)
        else:
            interp = horizInterp(self.tri, self.datain.flat[self.ball])
            self.dataout = interp(self.romsgrd.points)
        return self
    
    
    def interp2romsgrd(self, j=None):
        '''
        '''
        if '3D' in self.dimtype:
            for k in np.arange(self._depths.size):
                self._interp2romsgrd(k)
            self.dataout[:] = self._vert_interp.vert_interp(self.datatmp[::-1])
        else:
            self._interp2romsgrd()
        return self
    
    
    #def _get_barotropic_velocity(self, baroclinic_velocity, cell_depths):
        #sum_baroclinic = np.sum((baroclinic_velocity * cell_depths), axis=0)
        #total_depth = np.sum((cell_depths), axis=0)
        #sum_baroclinic /= total_depth
        #return sum_baroclinic
        
        
    #def set_barotropic(self): #, boundary):
        #'''
        #'''
        #self.barotropic = self._get_barotropic_velocity(self.dataout, self.romsgrd.scoord2dz())
        #return self
    
    
    def debug_figure(self):
        if self.var_dic[self.varname] in 'SALT':
            cmin, cmax = 34, 38
        elif self.var_dic[self.varname] in 'TEMP':
            cmin, cmax = 5, 25
        elif self.var_dic[self.varname] in ('U', 'V'):
            cmin, cmax = -.4, .4
        else:
            raise Exception
        
        plt.figure()
        ax = plt.subplot(211)
        ax.set_title('Mercator vertical grid')
        pcm = ax.pcolormesh(self.romsgrd.lat(), -self.depths(), self.datatmp)
        pcm.set_clim(cmin, cmax)
        ax.plot(np.squeeze(self.romsgrd.lat()), -self.romsgrd.h(), c='gray', lw=3)
        plt.colorbar(pcm)
        
        ax = plt.subplot(212)
        ax.set_title('ROMS vertical grid')
        pcm = ax.pcolormesh(np.tile(self.romsgrd.lat(), (self.romsgrd.N, 1)),
                                   self.romsgrd.scoord2z_r(),
                                   self.dataout)
        pcm.set_clim(cmin, cmax)
        ax.plot(np.squeeze(self.romsgrd.lat()), -self.romsgrd.h(), c='gray', lw=2)
        plt.colorbar(pcm)
        
        plt.figure()
        ax = plt.subplot(211)
        ax.scatter(self._vert_interp.hweights, self._vert_interp.vweights, s=10, edgecolors='none')
        
        ax = plt.subplot(212)
        lats1 = np.tile(self.romsgrd.lat(), (self.romsgrd.N, 1))
        lats2 = np.tile(self.romsgrd.lat(), (self._depths.size, 1))
        ax.scatter(lats1, self.romsgrd.scoord2z_r(), s=10, c='r', edgecolors='none', label='ROMS')
        ax.scatter(lats2, self._2d_depths, s=10, c='g', edgecolors='none', label='Mercator')
                                     
        plt.show()
        return
        
        
    def check_angle(self):
        '''
        Check angle computation
        '''
        plt.pcolormesh(self.lon(), self.lat(), self.angle())
        plt.axis('image')
        plt.colorbar()
        plt.show()
        return


def prepare_romsgrd(romsgrd):
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=False)
    romsgrd.make_kdetree()
    return romsgrd


def prepare_mercator(mercator, balldist):
    assert mercator._time_count.size == len(mercator.romsfile), "it's possible some files are missing"
    mercator.proj2gnom(ignore_land_points=False, M=mercator.romsgrd.M)
    mercator.child_contained_by_parent(mercator.romsgrd)
    mercator.make_kdetree().get_fillmask_cofs()
    ballpoints = mercator.kdetree.query_ball_tree(mercator.romsgrd.kdetree, r=balldist)
    mercator.ball = np.array(np.array(ballpoints).nonzero()[0])
    mercator.tri = sp.Delaunay(mercator.points[mercator.ball])
    if '3D' in mercator.dimtype:
        mercator.set_2d_depths().set_map_coordinate_weights()
        mercator.vert_interp()  
    proceed = True
    return mercator, proceed
    
    
    
if __name__ == '__main__':
    
    '''
    py_mercator2roms
    
    
    Evan Mason 2014
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # Mercator information
    #mercator_dir = '/marula/emason/data/mercator/nea_daily/'
    #mercator_dir = '/marula/emason/data/mercator/nwmed/ORCA12/'
    #mercator_dir = '/marula/emason/data/IBI_daily/'
    mercator_dir = '/data_cmems/ibi_phys005001b/'
    

    # Child ROMS information
    #roms_dir     = '../'
    #roms_dir     = '/marula/emason/runs2012/MedSea5/'
    #roms_dir     = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir     = '/marula/emason/runs2009/na_2009_7pt5km/'
    #roms_dir     = '/marula/emason/runs2014/na75/'
    #roms_dir     = '/Users/emason/runs/runs2014/MedCan5/'
    #roms_dir = '/Users/emason/runs2009/na_2009_7pt5/'
    #roms_dir     = '/marula/emason/runs2014/NWMED2/'
    #roms_dir     = '/marula/emason/runs2014/AlbSea175/'
    #roms_dir     = '/marula/emason/runs2015/AlbSea500/'
    roms_dir     = '/marula/emason/runs2016/meddies/'
    
    #roms_grd     = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd     = 'grd_MedCan5.nc'
    #roms_grd = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd = 'grd_nwmed_2km.nc'
    #roms_grd = 'grd_AlbSea175.nc'
    #roms_grd = 'grd_AlbSea500.nc'
    roms_grd = 'grd_meddies_1km.nc'

    if 'roms_grd_NA2009_7pt5km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
        bry_filename = 'bry_na75_mercator.nc'
        start_date = '20100106'
        end_date   = '20130531'
        day_zero = '????????'
        obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed
    
    elif 'grd_nwmed_2km.nc' in roms_grd:
        sigma_params = dict(theta_s=6.5, theta_b=0, hc=110, N=36)
        bry_filename = 'bry_nwmed_2km.nc'
        start_date = '20100106'
        end_date   = '20130531'
        day_zero = '19850101'
        obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed
    
    elif 'grd_AlbSea175.nc' in roms_grd:
        sigma_params = dict(theta_s=7., theta_b=0.25, hc=90., N=32)
        bry_filename = 'bry_AlbSea175.nc'
        start_date = '20131230'
        end_date   = '20140602'
        day_zero = '20140101'
        obc_dict = dict(south=0, east=1, north=1, west=1) # 1=open, 0=closed
    
    elif 'grd_AlbSea500.nc' in roms_grd:
        sigma_params = dict(theta_s=6., theta_b=0.25, hc=100., N=32)
        bry_filename = 'bry_AlbSea500_NOPREFILTER.nc'
        start_date = '20131230'
        end_date   = '20140602'
        day_zero = '20140101'
        obc_dict = dict(south=0, east=1, north=1, west=1) # 1=open, 0=closed

    elif 'grd_meddies_1km.nc' in roms_grd:
        sigma_params = dict(theta_s=7, theta_b=6, hc=300, N=50)
        #ini_filename = 'ini_meddies_ecco_198501.nc'
        #bry_filename = 'bry_meddies_IBI_4r1M_20160120.nc'
        #bry_filename = 'bry_meddies_IBI_3r1M_20130422.nc'
        bry_filename = 'bry_meddies_IBI_2r1M_20120109.nc'
        #bry_filename = 'bry_meddies_IBI_1rM_20120109.nc'

        #ini_date = '20131130'
        #start_date, end_date, day_zero = '20130422', '20140413', '20130422'
        start_date, end_date, day_zero = '20120109', '20130421', '20120109'
        #start_date, end_date, day_zero = '20111231', '20120108', '20111231'
        obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed
    
    else:
        raise Exception('No sigma params defined for grid: %s' % roms_grd)
    
    # Child ROMS boundary file information
    bry_cycle = 0.     # days, 0 means no cycle

    
    
    if 'ORCA' in mercator_dir:
        #mercator_ssh_files = sorted(glob.glob(mercator_dir + '*_gridT*.nc'))
        mercator_ssh_files = sorted(glob.glob(mercator_dir + '*_grid2D*.nc'))
        mercator_temp_files = sorted(glob.glob(mercator_dir + '*_gridT*.nc'))
        mercator_salt_files = sorted(glob.glob(mercator_dir + '*_gridS*.nc'))
        mercator_u_files = sorted(glob.glob(mercator_dir + '*_gridU*.nc'))
        mercator_v_files = sorted(glob.glob(mercator_dir + '*_gridV*.nc'))
        balldist = 50000. # meters
    
    elif 'PSY2V4R4' in mercator_dir:
        to_be_done
        balldist = 50000. # meters
    
    elif 'IBI_daily' in mercator_dir or 'ibi' in mercator_dir:
        print 'Warning: check version numbers in filenames'
        #mercator_ssh_files = glob.glob(mercator_dir + 'pde_ibi36v?r1_ibisr_01dav_*_HC01.nc')
        #mercator_ssh_files = glob.glob(mercator_dir +
                                       #'CMEMS_4r1M_IBI_PHY_NRT_PdE_01dav_????????_????????_R*_HC01.nc')
        #mercator_ssh_files = glob.glob(mercator_dir +
                                       #'CMEMS_3r1M_IBI_PHY_NRT_PdE_01dav_????????_????????_R*_HC01.nc')
        mercator_ssh_files = glob.glob(mercator_dir +
                                       'CMEMS_2r1M_IBI_PHY_NRT_PdE_01dav_????????_????????_R*_HC01.nc')
        #mercator_ssh_files = glob.glob(mercator_dir +
                                       #'CMEMS_1r1M_IBI_PHY_NRT_PdE_01dav_????????_????????_R*_HC01.nc')
        mercator_temp_files = \
        mercator_salt_files = \
        mercator_u_files = \
        mercator_v_files = sorted(mercator_ssh_files)
        balldist = 20000. # meters
    #aaaaaa

    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    
    assert len(start_date) == 8, '*start_date* must be a length 8 string'
    assert len(end_date) == 8,  '*end_date* must be a length 8 string'
    
    fillval = 9999.

    if 'IBI_daily' in mercator_dir or 'ibi' in mercator_dir:
        k2c = -273.15
    
    day_zero = datetime(int(day_zero[:4]), int(day_zero[4:6]), int(day_zero[6:]))
    day_zero = plt.date2num(day_zero)

    start_date += '12'
    end_date += '12'
    
    dtstrdt = plt.datetime.datetime(np.int(start_date[:4]),
                                    np.int(start_date[4:6]),
                                    np.int(start_date[6:8]),
                                    np.int(start_date[8:]))
    
    dtenddt = plt.datetime.datetime(np.int(end_date[:4]),
                                    np.int(end_date[4:6]),
                                    np.int(end_date[6:8]),
                                    np.int(end_date[8:]))
    
    dtstr, dtend = plt.date2num(dtstrdt), plt.date2num(dtenddt)

    # Set up a RomsGrid object
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
    romsgrd.set_bry_dx()
    romsgrd.set_bry_maskr()
    romsgrd.set_bry_areas()
    
    
    # Get surface areas of open boundaries
    chd_bry_surface_areas = []
    for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
        
        if 'west' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_west.sum(axis=0) * romsgrd.maskr_west)
        
        elif 'east' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_east.sum(axis=0) * romsgrd.maskr_east)
        
        elif 'south' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_south.sum(axis=0) * romsgrd.maskr_south)
        
        elif 'north' in boundary and is_open:
            chd_bry_surface_areas.append(romsgrd.area_north.sum(axis=0) * romsgrd.maskr_north)
    
    # Get total surface of open boundaries
    chd_bry_total_surface_area = np.array([area.sum() for area in chd_bry_surface_areas]).sum()
    
    
    # Set up a RomsData object for creation of the boundary file
    romsbry = RomsData(roms_dir + bry_filename, 'ROMS')
    romsbry.create_bry_nc(romsgrd, obc_dict, bry_cycle, fillval, 'py_mercator2roms')
    
    mercator_vars = dict(SSH = mercator_ssh_files,
                         TEMP = mercator_temp_files,
                         SALT = mercator_salt_files,
                         U = mercator_u_files)
    
    
    for mercator_var, mercator_files in zip(mercator_vars.keys(), mercator_vars.values()):
        
        print '\nProcessing variable *%s*' % mercator_var
        proceed = False
        
        
        with netcdf.Dataset(mercator_files[0]) as nc:
            try:
                mercator_date_start = nc.variables['time_counter'][:]
                mercator_time_units = nc.variables['time_counter'].units
                mercator_time_origin = nc.variables['time_counter'].time_origin
                mercator_time_origin = plt.date2num(plt.datetime.datetime.strptime(
                                            mercator_time_origin, '%Y-%b-%d %H:%M:%S'))
            except Exception:
                mercator_date_start = nc.variables['time'][:]
                mercator_time_units = nc.variables['time'].units
                mercator_time_origin = mercator_time_units.partition(' ')[-1].partition(' ')[-1]
                mercator_time_origin = plt.date2num(plt.datetime.datetime.strptime(
                                            mercator_time_origin, '%Y-%m-%d %H:%M:%S'))
                
        with netcdf.Dataset(mercator_files[-1]) as nc:
            try:
                mercator_date_end = nc.variables['time_counter'][:]
            except Exception:
                mercator_date_end = nc.variables['time'][:]
        
        
        
        if 'seconds' in mercator_time_units:
            mercator_dates = np.arange(mercator_date_start, mercator_date_end + 86400, 86400)
            mercator_dates /= 86400.
        elif 'hours' in mercator_time_units:
            mercator_dates = np.arange(mercator_date_start, mercator_date_end + 24, 24)
            mercator_dates /= 24.
        else:
            raise Exception('deal_with_when_a_problem')
        
        mercator_dates += mercator_time_origin
        mercator_dates = mercator_dates[np.logical_and(mercator_dates >= dtstr,
                                                       mercator_dates <= dtend)]
        
        
        for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
            
            print '\n--- processing %sern boundary' % boundary
            
            if 'west' in boundary and is_open:
                romsgrd_at_bry = WestGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                proceed = True
            
            elif 'north' in boundary and is_open:
                romsgrd_at_bry = NorthGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                proceed = True
            
            elif 'east' in boundary and is_open:
                romsgrd_at_bry = EastGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                proceed = True
            
            elif 'south' in boundary and is_open:
                romsgrd_at_bry = SouthGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                proceed = True
            
            else:
                proceed = False #raise Exception
            
            if proceed:
                
                romsgrd_at_bry = prepare_romsgrd(romsgrd_at_bry)
                mercator = MercatorData(mercator_files, 'Mercator', mercator_var, romsgrd_at_bry)
                mercator, proceed = prepare_mercator(mercator, balldist)
                
                if 'U' in mercator_var:
                    mercator_v = MercatorData(mercator_v_files, 'Mercator', 'V', romsgrd_at_bry,
                                         i0=mercator.i0, i1=mercator.i1, j0=mercator.j0, j1=mercator.j1)
                    mercator_v, junk = prepare_mercator(mercator_v, balldist)
                
                tind = 0     # index for writing records to bry file
                
                for dt in mercator_dates:
                        
                        
                    dtnum = dt - day_zero
                        
                        
                    # Read in variables
                    mercator.get_variable(dt).fillmask()
                        
                    
                    # Calculate barotropic velocities
                    if mercator.vartype in 'U':
                            
                        mercator_v.get_variable(dt).fillmask()
                        
                        # Rotate to zero angle
                        for k in np.arange(mercator.depths().size):
                            u, v = mercator.datain[k], mercator_v.datain[k]
                            mercator.datain[k], mercator_v.datain[k] = mercator.rotate(u, v, sign=-1)
                        
                        mercator.interp2romsgrd()
                        mercator_v.interp2romsgrd()
                        
                        # Rotate u, v to child angle
                        for k in np.arange(romsgrd.N).astype(np.int):
                            u, v = mercator.dataout[k], mercator_v.dataout[k]
                            mercator.dataout[k], mercator_v.dataout[k] = romsgrd.rotate(u, v, sign=1,
                                                                     ob=boundary)
                        
                        mercator.set_barotropic()
                        mercator_v.set_barotropic()
                        
                    else:
                        
                        mercator.interp2romsgrd()
                        
                        
                    # Write to boundary file
                    with netcdf.Dataset(romsbry.romsfile, 'a') as nc:
                            
                        if mercator.vartype in 'U':
                            
                            if boundary in ('north', 'south'):
                                u = romsgrd.half_interp(mercator.dataout[:,:-1],
                                                        mercator.dataout[:,1:])
                                ubar = romsgrd.half_interp(mercator.barotropic[:-1],
                                                           mercator.barotropic[1:])
                                nc.variables['u_%s' % boundary][tind] = u
                                nc.variables['ubar_%s' % boundary][tind] = ubar
                                nc.variables['v_%s' %boundary][tind] = mercator_v.dataout
                                nc.variables['vbar_%s' % boundary][tind] = mercator_v.barotropic
                            
                            elif boundary in ('east', 'west'):
                                v = romsgrd.half_interp(mercator_v.dataout[:,:-1],
                                                        mercator_v.dataout[:,1:])
                                vbar = romsgrd.half_interp(mercator_v.barotropic[:-1],
                                                           mercator_v.barotropic[1:])
                                nc.variables['v_%s' % boundary][tind] = v
                                nc.variables['vbar_%s' % boundary][tind] = vbar
                                nc.variables['u_%s' % boundary][tind] = mercator.dataout
                                nc.variables['ubar_%s' % boundary][tind] = mercator.barotropic
                            
                            else:
                                raise Exception('Unknown boundary: %s' % boundary)
                        
                        elif mercator.vartype in 'SSH':
                            
                            nc.variables['zeta_%s' % boundary][tind] = mercator.dataout
                            nc.variables['bry_time'][tind] = np.float(dtnum)
            
                        elif mercator.vartype in 'TEMP':
                            
                            if (mercator.dataout > 100.).any():
                                mercator.dataout += k2c # Kelvin to Celcius
                            #mercator.dataout *= romsgrd.mask3d()
                            nc.variables['temp_%s' % boundary][tind] = mercator.dataout
                        
                        else:
                            
                            varname = mercator.vartype.lower() + '_%s' % boundary
                            nc.variables[varname][tind] = mercator.dataout
                            
                        #if 'east' in boundary:
			    #print plt.num2date(dt)
			    #plt.figure(99)
			    #plt.subplot(121)
			    #plt.pcolormesh(mercator.mapcoord_weights)
			    #plt.colorbar()
			    #plt.subplot(122)
			    #plt.pcolormesh(mercator.dataout)
			    ##plt.pcolormesh(np.log(mercator.mapcoord_weights))
			    #plt.clim(34, 39)
			    #plt.colorbar()
			    #plt.show()
                        
                    tind += 1
            #proceed = False
    
    
    # Correct volume fluxes and write to boundary file
    print '\nProcessing volume flux correction'
    with netcdf.Dataset(romsbry.romsfile, 'a') as nc:
                    
        bry_times = nc.variables['bry_time'][:]
        boundarylist = []
            
        for bry_ind in xrange(bry_times.size):
            
            uvbarlist = []
            
            for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
            
                if 'west' in boundary and is_open:
                    uvbarlist.append(nc.variables['ubar_west'][bry_ind])
                    
                elif 'east' in boundary and is_open:
                    uvbarlist.append(nc.variables['ubar_east'][bry_ind])
                
                elif 'north' in boundary and is_open:
                    uvbarlist.append(nc.variables['vbar_north'][bry_ind])
                
                elif 'south' in boundary and is_open:
                    uvbarlist.append(nc.variables['vbar_south'][bry_ind])
                    
                if bry_ind == 0 and is_open:
                    boundarylist.append(boundary)
                    
            fc = bry_flux_corr(boundarylist,
                               chd_bry_surface_areas,
                               chd_bry_total_surface_area,
                               uvbarlist)
            
            print '------ barotropic velocity correction:', fc, 'm/s'
            
            for boundary, is_open in zip(obc_dict.keys(), obc_dict.values()):
                            
                if 'west' in boundary and is_open:
                    nc.variables['u_west'][bry_ind] -= fc
                    nc.variables['ubar_west'][bry_ind] -= fc
                                
                elif 'east' in boundary and is_open:
                    nc.variables['u_east'][bry_ind] += fc
                    nc.variables['ubar_east'][bry_ind] += fc
                    
                elif 'north' in boundary and is_open:
                    nc.variables['v_north'][bry_ind] += fc
                    nc.variables['vbar_north'][bry_ind] += fc
                            
                elif 'south' in boundary and is_open:
                    nc.variables['v_south'][bry_ind] -= fc
                    nc.variables['vbar_south'][bry_ind] -= fc
    
    
    print 'all done'
            
            
      














