# -*- coding: utf-8 -*-
# %run py_ecco2roms.py

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

Create a ROMS boundary file based on ECCO2 data

===========================================================================
'''
import ftplib
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
# import progressbar as prog_bar
from py_roms2roms import vertInterp, horizInterp, horizInterpRbs, bry_flux_corr, debug0, debug1, debug2
from py_roms2roms import ROMS, RomsGrid, RomsData
from py_mercator2roms import WestGrid, EastGrid, NorthGrid, SouthGrid
from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    """
    http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/#
    """
    with open(os.devnull, "w") as devnull:
        #old_stdout = sys.stdout
        #sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            #sys.stdout = old_stdout
            sys.stderr = old_stderr



#class RomsGrid(RomsGrid):
    #'''
    #Modify the RomsGrid class
    #'''
    
    #def lon(self):
        #'''
        #We need to override 'lon' method of pyroms2roms RomsGrid to
        #account for the ECCO 0-360 degree grid.
        #'''
        ##lon = self.read_nc('lon_rho',  indices=self.indices)
        
        ## If all west of 0 degrees (e.g., for Canaries)
        ## then it's easy...
        #if np.all(self._lon < 0.):
            #self._lon += 360.
            
        ## Trickiest case (e.g. western Med)
        ## no option but to activate a flag indicating
        ## modification of the ECCO grid and variables
        #elif np.logical_and(np.any(self._lon < 0.),
                            #np.any(self._lon >= 0.)):
            #self.zero_crossing = True
        #return self._lon



class EccoData (RomsData):
    '''
    EccoData class (inherits from RomsData class)
    '''
    def __init__(self, filenames, model_type, ecco_var, romsgrd, **kwargs):
        """
        Creates a new Ecco data object.
        
        Parameters
        ----------
        
        *filenames* : list of Ecco nc files.
        *model_type* : string specifying Ecco model.
        *romsgrd* : a `RomsGrid` instance.
        
        """
        super(EccoData, self).__init__(filenames[0], model_type)
        self.filenames = filenames
        self.vartype = ecco_var
        self.romsgrd = romsgrd
        self.var_dic = {'SSH':'SSH',
                        'THETA':'TEMP', 'SALT':'SALT',
                        'UVEL':'U', 'VVEL':'V'}
        self._set_variable_type()
        not_done = True
        while not_done:
            try:
                with suppress_stdout():
                    self.FillVal = self.read_nc_att(self.varname, 'FillValue')
                    self._lon = self.read_nc('LONGITUDE_T', indices='[:]')
                    self._lat = self.read_nc('LATITUDE_T', indices='[:]')
                not_done = False
            except:
                print "------ *read_nc('LONGITUDE_T')* paused, trying again"
                time.sleep(0.5)
        self._lon, self._lat = np.meshgrid(self._lon, self._lat)
        #self.zero_crossing = romsgrd.zero_crossing
        if self.romsgrd.zero_crossing is True:
            self.zero_crossing = True
        if 'V' in self.vartype and not self.zero_crossing:
            self._lon -= 360
        if kwargs.has_key('i0'):
            self.i0, self.i1 = kwargs['i0'], kwargs['i1']
            self.j0, self.j1 = kwargs['j0'], kwargs['j1']
        else:
            self.set_subgrid(romsgrd, k=50)
            if self.i0 == 0:
                self.i0 += 1

        if 'SSH' in self.vartype:
            self._depths = np.array([0])
        else:
            not_done = True
            while not_done:
                try:
                    self._depths = self.read_nc('DEPTH_T', indices='[:]')
                    not_done = False
                except:
                    print "------ *read_nc('DEPTH_T')* paused, trying again"
                    time.sleep(0.5)

        self._set_data_shape()
        self._set_dates()
        self.datain = np.empty(self._data_shape).astype(np.float64)
        if len(self._data_shape) == 2:
            self.dataout = np.ma.empty(self.romsgrd.lon().shape)
        else:
            tmp_shape = (self._depths.size, self.romsgrd.lon().shape[0],
                         self.romsgrd.lon().shape[1])
            self.datatmp = np.squeeze(np.ma.empty(tmp_shape))
            self.dataout = np.squeeze(np.ma.empty(self.romsgrd.mask3d().shape))
        self._set_maskr()
        self._set_angle()
        
        
        
    def lon(self):
        if self.zero_crossing is True:
            lon1 = self._lon[self.j0:self.j1, :self.i0]
            lon0 = self._lon[self.j0:self.j1, self.i1:] - 360.
            return np.concatenate((lon0, lon1), axis=1)
        else:
            return self._lon[self.j0:self.j1, self.i0:self.i1]
    
    def lat(self):
        if self.zero_crossing is True:
            lat1 = self._lat[self.j0:self.j1, :self.i0]
            lat0 = self._lat[self.j0:self.j1, self.i1:]
            return np.concatenate((lat0, lat1), axis=1)
        else:
            return self._lat[self.j0:self.j1, self.i0:self.i1]
    
    def maskr(self):
        if self.zero_crossing is True:
            maskr1 = self._maskr[self.j0:self.j1, :self.i0]
            maskr0 = self._maskr[self.j0:self.j1, self.i1:]
            return np.concatenate((maskr0, maskr1), axis=1)
        else:
            return self._maskr[self.j0:self.j1, self.i0:self.i1]
    
    def maskr3d(self):
        if self.zero_crossing is True:
            maskr3d1 = self._maskr3d[:, self.j0:self.j1, :self.i0]
            maskr3d0 = self._maskr3d[:, self.j0:self.j1, self.i1:]
            return np.concatenate((maskr3d0, maskr3d1), axis=2)
        else:
            return self._maskr3d[:, self.j0:self.j1, self.i0:self.i1]
    
    def angle(self):
        if self.zero_crossing is True:
            angle1 = self._angle[self.j0:self.j1, :self.i0]
            angle0 = self._angle[self.j0:self.j1, self.i1:]
            return np.concatenate((angle0, angle1), axis=1)
        else:
            return self._angle[self.j0:self.j1, self.i0:self.i1]
    
    def depths(self):
        return self._depths
    
    def dates(self):
        return self._time_count
    
    
    def _set_data_shape(self):
        ssh_varnames = ('SSH') # clumsy; needs to be added to if they invent more names
        #if kwargs.has_key('predefined_shp'):
            #self._data_shape = kwargs['predefined_shp']
            #if len(self._data_shape ) > 2:
                #self.dimtype = '3D'
            #else:
                #self.dimtype = '2D'
        if self._depths.size == 1 or self.varname in ssh_varnames:
            _shp = self.lon().shape#(self.j1 - self.j0, self.i1 - self.i0)
            self._data_shape = _shp
            self.dimtype = '2D'
        else:
            #_shp = (self._depths.size, self.j1 - self.j0, self.i1 - self.i0)
            _shp = (self._depths.size, self.lon().shape[0], self.lon().shape[1])
            self._data_shape = _shp
            self.dimtype = '3D'
        return self
    
    
    def _set_variable_type(self):
        self.varname = None
        for varname, vartype in zip(self.var_dic.keys(), self.var_dic.values()):
            if varname in self.list_of_variables() and vartype in self.vartype:
                self.varname = varname
                return self
        if self.varname is None:
            raise Exception # no candidate variable identified
    
    
    def _get_maskr(self):#, k=None):
        """
        Called by _set_maskr()
        """
        indices = '[0]'
        
        not_done = True
        while not_done:
            try:
                _mask = self.read_nc(self.varname, indices=indices)
                not_done = False
            except:
                print "--- *_get_maskr* paused, trying again"
                time.sleep(0.5) # pause for half a second
        
        try:
            #print 'self.varname', self.varname, indices
            #_mask = self.read_nc(self.varname, indices=indices)
            _mask[:] = np.ma.masked_equal(_mask, self.FillVal).mask
        except Exception: # triggered when all points are masked
            _mask = np.ones(_mask.shape)
        _mask = np.asarray(_mask, dtype=np.int)
        _mask *= -1.
        _mask += 1.
        return _mask
    
        
    def _set_maskr(self):
        """
        Set the landsea mask (*self.maskr*) with same shape
        as input Mercator nc file. If a 3D variable, then additional
        attribute, self._maskr3d is set.
        """
        if '3D' in self.dimtype:
            self._maskr = self._get_maskr()
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
        self._maskr3d[:] = self._get_maskr()
        return self
        
        
    def _set_angle(self):
        '''
        Compute angles of local grid positive x-axis relative to east
        '''
        latu = np.deg2rad(0.5 * (self._lat[:, 1:] + self._lat[:, :-1]))
        lonu = np.deg2rad(0.5 * (self._lon[:, 1:] + self._lon[:, :-1]))
        dellat = latu[:, 1:] - latu[:, :-1]
        dellon = lonu[:, 1:] - lonu[:, :-1]
        dellon[dellon > np.pi] = dellon[dellon >  np.pi] - (2. * np.pi)
        dellon[dellon < -np.pi] = dellon[dellon < -np.pi] + (2. * np.pi)
        dellon = dellon * np.cos(0.5 * (latu[:, 1:] + latu[:, :-1]))

        self._angle = np.zeros_like(self._lat)
        ang_s = np.arctan(dellat / (dellon + np.spacing(0.4)))
        deli = np.logical_and(dellon < 0., dellat < 0.)
        ang_s[deli] = ang_s[deli] - np.pi

        deli = np.logical_and(dellon < 0., dellat >= 0.)
        ang_s[deli] = ang_s[deli] + np.pi
        ang_s[ang_s > np.pi] = ang_s[ang_s > np.pi] - np.pi
        ang_s[ang_s < -np.pi] = ang_s[ang_s <- np.pi] + np.pi

        self._angle[:, 1:-1] = ang_s
        self._angle[:, 0] = self._angle[:, 1]
        self._angle[:, -1] = self._angle[:, -2]
        return self
    
    
    
    def check_grids(self):
        """
        Test that ROMS grid points totally contained by ECCO grid points.
        """
        good_subgrid = np.array([self.lon().min() < self.romsgrd.lon().min(),
                                self.lon().max() > self.romsgrd.lon().max(),
                                self.lat().min() < self.romsgrd.lat().min(),
                                self.lat().max() > self.romsgrd.lat().max()])
        assert np.alltrue(good_subgrid), 'romsgrd points outside ecco grid'
    
    
    def get_variable(self, date):
        ind = np.nonzero(self._time_count == date)[0]
        
        #print ind, self._time_count, date
        
        """ Ecco2 SSH are daily averages, whereas the 3d variables are 3-day averages.
        """
        #print '============================================='
        not_done = True
        if 'SSH' in self.varname:
            
            while not_done:
                try:
                    with suppress_stdout():
                        with netcdf.MFDataset(self.filenames[ind-1:ind+2],
                                              aggdim='TIME') as nc:
                            if self.zero_crossing is True:
                                datain1 = nc.variables[self.varname] \
                                    [:, self.j0:self.j1, :self.i0].mean(axis=0)
                                datain0 = nc.variables[self.varname] \
                                    [:, self.j0:self.j1, self.i1:].mean(axis=0)
                                self.datain[:] = np.concatenate((datain0, datain1),
                                                                axis=1)
                            else:
                                self.datain[:] = nc.variables[self.varname] \
                                    [:, self.j0:self.j1, self.i0:self.i1].mean(axis=0)
                    not_done = False
                except:
                    print "------ *SSH* paused, trying again"
                    time.sleep(0.5) # pause for half a second

        else:
            while not_done:
                try:
                    with suppress_stdout():
                        with netcdf.Dataset(self.filenames[ind]) as nc:
                            if self.zero_crossing is True:
                                datain1 = nc.variables[self.varname] \
                                    [0, :, self.j0:self.j1, :self.i0]
                                datain0 = nc.variables[self.varname] \
                                    [0, :, self.j0:self.j1, self.i1:]
                                self.datain[:] = np.concatenate((datain0, datain1),
                                                                axis=2)
                            else:
                                self.datain[:] = nc.variables[self.varname] \
                                    [0, :, self.j0:self.j1, self.i0:self.i1]
                    not_done = False
                except:
                    print "------ *3D VAR* paused, trying again"
                    time.sleep(0.5)
        
        self.datain[:] = np.ma.masked_array(self.datain).data
        return self
    
    
    def _set_time_origin(self):
        not_done = True
        while not_done:
            try:
                self._time_counter_origin = self.read_nc_att('TIME', 'time_origin')
                not_done = False
            except:
                print "------ *_set_time_origin* paused, trying again"
                time.sleep(0.5)
        ymd, hms = self._time_counter_origin.split(' ')
        y, mo, d = ymd.split('-')
        h, mi, s = hms.split(':')
        time_origin = datetime(int(y), strptime(mo,'%m').tm_mon, int(d),
                               int(h), int(mi), int(s))
        self.time_origin = plt.date2num(time_origin)
        return self
    
    
    def _set_dates(self):
        self._set_time_origin()
        #print 'self.time_origin ',self.time_origin
        #print 'self.romsfile ',self.romsfile
        not_done = True
        while not_done:
            try:
                with netcdf.Dataset(self.filenames[0]) as nc:
                    date_start = nc.variables['TIME'][:]
                with netcdf.Dataset(self.filenames[-1]) as nc:
                    date_end = nc.variables['TIME'][:]
                not_done = False
            except:
                print "------ *_set_dates* paused, trying again"
                time.sleep(0.5)
        #self._time_count = self.read_nc_mf('TIME')
        if self.varname is not 'SSH':
            self._time_count = np.arange(date_start, date_end + 3, 3)
        else:
            #self._time_count = np.arange(date_start + 1, date_end + 3, 3)
            self._time_count = np.arange(date_start, date_end + 1, 1)
        #print 'self._time_count',self._time_count
        #self._time_count /= 86400.
        #print 'self._time_count',self._time_count
        self._time_count += self.time_origin
        #print 'self._time_count',self._time_count
        return self
    
    
    def get_date(self, ind):
        return self._time_count[ind]
        
        
    def get_fillmask_cofs(self):
        if '3D' in self.dimtype:
            self.fillmask_cof_tmp = []
            for k in np.arange(self._depths.size):
                try:
                    self.get_fillmask_cof(self.maskr3d()[k])
                    #if len(self.fillmask_cof[k][2]) > 3:
                    self.fillmask_cof_tmp.append(self.fillmask_cof)
                    #else:
                        #self.fillmask_cof_tmp.append(None)
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
        #weight[:] = np.nan_to_num(weight)
        np.place(weight, weight > 1., 0.)
        if k is not None:
            try:
	        #print 'weight1', weight
                xfill = weight * self.datain[k][igood[:,0][iquery], igood[:,1][iquery]]
            except:
	        #print 'weight2', weight
                igood = np.tile(igood, (igood.shape[0] * 2,1))
                #print 'igood',igood
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
            for k in np.arange(len(self.fillmask_cof)):
                #print k
                if self.fillmask_cof[k] is not None:
                    dist, iquery, igood, ibad = self.fillmask_cof[k]
                    self._fillmask(dist, iquery, igood, ibad, k=k)
            self._check_and_fix_deep_levels()
        else:
            dist, iquery, igood, ibad = self.fillmask_cof
            self._fillmask(dist, iquery, igood, ibad)
        return self
    
    
    def _check_and_fix_deep_levels(self):
        for k in np.arange(self._depths.size):
            if self.maskr3d()[k].sum() == 0:
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
            ecco_depths = self._3d_depths[:,j]
            roms_depths = self.romsgrd.scoord2z_r()[:,j]
        else:
            ecco_depths = self._2d_depths
            roms_depths = self.romsgrd.scoord2z_r()
        self.mapcoord_weights = self.romsgrd.get_map_coordinate_weights(
                                             roms_depths, ecco_depths)
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
            interp = horizInterpRbs(self.lon()[0],
                                    self.lat()[:, 0],
                                    self.datain[k])
            try:
                self.datatmp[k] = interp.rbs_interp(self.romsgrd.lon(),
                                                    self.romsgrd.lat())
            except Exception:
                self.datatmp[k] = interp.rbs_interp(self.romsgrd.lon(),
                                                    self.romsgrd.lat()).reshape(
                                                    self.lon().shape)
        else:
            interp = horizInterpRbs(self.lon()[0],
                                    self.lat()[:, 0],
                                    self.datain)
            self.dataout[:] = interp.rbs_interp(self.romsgrd.lon(),
                                                self.romsgrd.lat())
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
    
    
    def _get_barotropic_velocity(self, baroclinic_velocity, cell_depths):
        """
        """
        sum_baroclinic = (baroclinic_velocity * cell_depths).sum(axis=0)
        sum_baroclinic /= cell_depths.sum(axis=0)
        return sum_baroclinic
        
        
    def set_barotropic(self): #, open_boundary):
        '''
        '''
        self.barotropic = self._get_barotropic_velocity(self.dataout,
                                                        self.romsgrd.scoord2dz())
        return self
    
    
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
    if np.logical_and(romsgrd.lon().min() < 0.,
                      romsgrd.lon().max() >= 0.):
        romsgrd.zero_crossing = True
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=False)
    romsgrd.make_kdetree()
    return romsgrd


def prepare_ecco(ecco):
    ecco.proj2gnom(ignore_land_points=False, M=ecco.romsgrd.M)
    #ecco.child_contained_by_parent(ecco.romsgrd)
    ecco.make_kdetree().get_fillmask_cofs()
    #ballpoints = ecco.kdetree.query_ball_tree(ecco.romsgrd.kdetree, r=balldist)
    #ecco.ball = np.array(np.array(ballpoints).nonzero()[0])
    #ecco.tri = sp.Delaunay(ecco.points[ecco.ball])
    if '3D' in ecco.dimtype:
        ecco.set_2d_depths().set_map_coordinate_weights()
        ecco.vert_interp()
    return ecco
  



def get_list_of_ecco_files(domain, url, path, subdir):
    '''
    Use ftplib to get list of files in each directory
    '''
    ftp = ftplib.FTP(domain)
    ftp.login()
    ftp.cwd(path + subdir)
    filelist = ftp.nlst()
    ftp.close()
    url += path
    url += subdir
    url += '/{0}'
    filelist = [url.format(f) for f in filelist]
    filelist = [s for s in filelist if s.endswith('.nc')]
    return filelist
    
    
    
    
if __name__ == '__main__':
    
    '''
    py_ecco2roms
    
    
    Evan Mason 2014
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # ECCO2 information
    ecco_domain = 'ecco2.jpl.nasa.gov'
    ecco_url = 'http://ecco2.jpl.nasa.gov:80/opendap/'
    #ecco_url = 'http://ecco2.jpl.nasa.gov/opendap/hyrax/'
    ecco_path = 'data1/cube/cube92/lat_lon/quart_90S_90N/'
    
    #ecco_vars = OrderedDict([('SSH', 'SSH.nc'),
                             #('TEMP', 'THETA.nc'),
                             #('SALT', 'SALT.nc'),
                             #('U','UVEL.nc')])
    ecco_vars = OrderedDict([('TEMP', 'THETA.nc'),
                     ('SALT', 'SALT.nc'),
                     ('U','UVEL.nc')])
    ecco_vars = OrderedDict([('SALT', 'SALT.nc')])#,
    #('U','UVEL.nc')])
    ecco_vars = OrderedDict([('U','UVEL.nc')])#,
    
    
    #ecco_vars = OrderedDict([('TEMP','THETA.nc')])
    #ecco_vars = OrderedDict([('U', 'UVEL.nc')])

    # Child ROMS information
    #roms_dir     = '../'
    #roms_dir     = '/marula/emason/runs2012/MedSea5/'
    #roms_dir     = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir     = '/marula/emason/runs2009/na_2009_7pt5km/'
    #roms_dir     = '/marula/emason/runs2014/MedCan5/'
    roms_dir     = '/marula/emason/runs2014/na75/'
    #roms_dir = '/marula/emason/runs2014/nwmed5km/'
    #roms_dir     = '/marula/emason/runs2014/na6km/'
    #roms_dir = '/Users/emason/runs2009/na_2009_7pt5/'
    #roms_dir = '/marula/emason/runs2009/na_2009_7pt5/'
    #roms_dir     = '/marula/emason/runs2014/NWMED2/'
    
    #roms_grd     = 'roms_grd_NA2009_7pt5km.nc'
    #roms_grd     = 'grd_MedCan5.nc'
    #roms_grd = 'roms_grd_NA2009_7pt5km.nc'
    roms_grd = 'roms_grd_NA2014_7pt5km.nc'
    #roms_grd = 'grd_nwmed5km.nc'
    #roms_grd     = 'grd_na6km.nc'
    #roms_grd = 'grd_nwmed_2km.nc'

    if 'roms_grd_NA2014_7pt5km.nc' in roms_grd:
        sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
        obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed
    elif 'roms_grd_NA2009_7pt5km.nc' in roms_grd:
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
    else:
        print 'No sigma parameters defined for grid: %s' % roms_grd
        raise Exception
    
    # Child ROMS boundary file information
    bry_cycle    =  0.     # days, 0 means no cycle
    
    #bry_filename = 'bry_nwmed_2km.nc'
    #bry_filename = 'bry_TEST.nc'
    #bry_filename = 'bry_nwmed5km_198501_TEMP.nc'
    #bry_filename = 'bry_na5_198501_update.nc'
    #bry_filename = 'bry_na75_198501_update_zeta.nc'
    bry_filename = 'bry_NA2015_ecco_198501_UV.nc'
    #bry_filename = 'bry_na6km_198501.nc'
    
    start_date   = '19920101'
    end_date     = '20141021'

    
    '''Must correspond to:
      (1) date of an existing Mercator file 
      (2) date used for ini file
    '''
    day_zero = '19850101'
    #day_zero = '19920101'
    #day_zero = '20051230'
    
    
    #balldist = 150000. # meters

    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    
    assert len(start_date) == 8, '*start_date* must be a length 8 string'
    assert len(end_date) == 8,  '*end_date* must be a length 8 string'
    
    fillval = 9999.

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
    
    # Number of records at daily frequency
    #delta = plt.datetime.timedelta(days=1)
    #numrec = plt.drange(dtstrdt, dtenddt, delta).size + 1

    dtstr, dtend = plt.date2num(dtstrdt), plt.date2num(dtenddt)
    #time_array = np.arange(plt.date2num(dtstrdt),
                           #plt.date2num(dtenddt) + 1, 1)

    # Set up a RomsGrid object
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
    romsgrd.set_bry_dx()
    romsgrd.set_bry_maskr()
    romsgrd.set_bry_areas()
    
    # Set flag for zero longitude crossing
    romsgrd.check_zero_crossing()
    
    # Get surface areas of open boundaries
    chd_bry_surface_areas = []
    for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):
        if 'west' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_west.sum(axis=0) *
                                         romsgrd.maskr_west)
        elif 'east' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_east.sum(axis=0) *
                                         romsgrd.maskr_east)
        elif 'south' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_south.sum(axis=0) *
                                         romsgrd.maskr_south)
        elif 'north' in open_boundary and flag:
            chd_bry_surface_areas.append(romsgrd.area_north.sum(axis=0) *
                                         romsgrd.maskr_north)
    
    # Get total surface of open boundaries
    chd_bry_total_surface_area = np.array([area.sum() for area in 
                                           chd_bry_surface_areas]).sum()
    
    
    # Set up a RomsData object for creation of the boundary file
    romsbry = RomsData(roms_dir + bry_filename, 'ROMS')
    romsbry.create_bry_nc(romsgrd, obc_dict, bry_cycle, fillval, 'py_ecco2roms')
    
    
    
    
    for ecco_var, ecco_subdir in zip(ecco_vars.keys()[:4], ecco_vars.values()[:4]):
        
        if 'U' in ecco_var:
            the_ecco_var = 's *U*, *V'
        else:
            the_ecco_var = ' *%s' % ecco_var
        print '\nProcessing variable%s*' % the_ecco_var
        proceed = False
        
        ecco_files = get_list_of_ecco_files(ecco_domain, ecco_url,
                                            ecco_path, ecco_subdir)
        #if 'SSH' in ecco_var:
            #ecco_files = ecco_files#[::3]
            #print 'FIX MEEEEEEEE'
        
        if 'U' in ecco_var and ecco_var in 'U':
            ecco_v_files = get_list_of_ecco_files(ecco_domain, ecco_url,
                                                  ecco_path, 'VVEL.nc')
            #print 'FIX ME...'
            #ecco_v_files = ecco_v_files
        #print 'FIX ME...'
        #ecco_files = ecco_files
        
        not_done = True
	print ecco_files[0]
        while not_done:
            try:
                with netcdf.Dataset(ecco_files[0]) as nc:
		    print ecco_files[0]
                    ecco_date_start = nc.variables['TIME'][:]
                    ecco_time_units = nc.variables['TIME'].units
                    ecco_time_origin = nc.variables['TIME'].time_origin
                    not_done = False
            except:
                print "------ *ecco_files[0]* paused, trying again"
                time.sleep(0.5) # pause for half a second
        
        
        not_done = True
        while not_done:
            try:
                with netcdf.Dataset(ecco_files[-1]) as nc:
                    ecco_date_end = nc.variables['TIME'][:]
                    not_done = False
            except:
                print "------ *ecco_files[-1]* paused, trying again"
                time.sleep(0.5) # pause for half a second
        
        ecco_time_origin = plt.date2num(plt.datetime.datetime.strptime(
                                            ecco_time_origin, '%Y-%m-%d %H:%M:%S'))
        
        if ecco_var is not 'SSH':
            ecco_dates = np.arange(ecco_date_start, ecco_date_end + 3, 3)
        else:
            ecco_dates = np.arange(ecco_date_start + 1, ecco_date_end + 3, 3)
        
        if 'seconds' in ecco_time_units:
            ecco_dates /= 86400.
        elif 'day' in ecco_time_units:
            pass
        else:
            Exception # deal_with_when_a_problem
        
        ecco_dates += ecco_time_origin
        ecco_dates = ecco_dates[np.logical_and(ecco_dates >= dtstr,
                                               ecco_dates <= dtend)]
        assert ecco_dates.size > 0, 'Bad time range...'
        
        for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):
            
            
            if 'west' in open_boundary and flag:
                romsgrd_at_bry = WestGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                print '\n--- processing %sern boundary' % open_boundary
                proceed = True
            elif 'north' in open_boundary and flag:
                romsgrd_at_bry = NorthGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                print '\n--- processing %sern boundary' % open_boundary
                proceed = True
            elif 'east' in open_boundary and flag:
                romsgrd_at_bry = EastGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                print '\n--- processing %sern boundary' % open_boundary
                proceed = True
            elif 'south' in open_boundary and flag:
                romsgrd_at_bry = SouthGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
                print '\n--- processing %sern boundary' % open_boundary
                proceed = True
            else:
                proceed = False# raise Exception
            
            if proceed:

                romsgrd_at_bry = prepare_romsgrd(romsgrd_at_bry)
                
                ecco = EccoData(ecco_files, 'Ecco', ecco_var, romsgrd_at_bry)
                ecco = prepare_ecco(ecco)
                ecco.check_grids()
                #sssssssssssssssssss
                
                if 'U' in ecco_var:
                    ecco_v = EccoData(ecco_v_files, 'Ecco', 'V', romsgrd_at_bry,
                                     i0=ecco.i0, i1=ecco.i1, j0=ecco.j0, j1=ecco.j1)
                    ecco_v = prepare_ecco(ecco_v)
                    ecco_v.check_grids()
            
                tind = 0 # index for writing records to bry file

                for dt in ecco_dates:
      
                    #print plt.num2date(dt)
                    dtnum = dt - day_zero

                    # Read in variables
                    ecco.get_variable(dt)
                    #aaaaaaa
                    ecco.fillmask()
                    
                    # Calculate barotropic velocities
                    if ecco.vartype in 'U':
                        
                        ecco_v.get_variable(dt).fillmask()
                        
                        # Rotate to zero angle
                        for k in np.arange(ecco.depths().size):
                            u, v = ecco.datain[k], ecco_v.datain[k]
                            ecco.datain[k], ecco_v.datain[k] = ecco.rotate(u, v, sign=-1)
                        
                        ecco.interp2romsgrd()
                        ecco_v.interp2romsgrd()
                        
                        # Rotate u, v to child angle
                        for k in np.arange(romsgrd.N).astype(np.int):
                            u, v = ecco.dataout[k], ecco_v.dataout[k]
                            ecco.dataout[k], ecco_v.dataout[k] = romsgrd.rotate(u, v, sign=1,
                                                                         ob=open_boundary)
                        
                        ecco.set_barotropic()
                        ecco_v.set_barotropic()
                        
                    else:
                        
                        ecco.interp2romsgrd()

                    
                    # Write to boundary file
                    with netcdf.Dataset(romsbry.romsfile, 'a') as nc:
                    
                        if ecco.vartype in 'U':
                            
                            if open_boundary in ('north', 'south'):
                                u = romsgrd.half_interp(ecco.dataout[:,:-1],
                                                        ecco.dataout[:,1:])
                                ubar = romsgrd.half_interp(ecco.barotropic[:-1],
                                                           ecco.barotropic[1:])
                                nc.variables['u_%s' % open_boundary][tind] = u
                                nc.variables['ubar_%s' % open_boundary][tind] = ubar
                                nc.variables['v_%s' % open_boundary][tind] = ecco_v.dataout
                                nc.variables['vbar_%s' % open_boundary][tind] = ecco_v.barotropic
                            
                            elif open_boundary in ('east', 'west'):
                                v = romsgrd.half_interp(ecco_v.dataout[:,:-1],
                                                        ecco_v.dataout[:,1:])
                                vbar = romsgrd.half_interp(ecco_v.barotropic[:-1],
                                                           ecco_v.barotropic[1:])
                                nc.variables['v_%s' % open_boundary][tind] = v
                                nc.variables['vbar_%s' % open_boundary][tind] = vbar
                                nc.variables['u_%s' % open_boundary][tind] = ecco.dataout
                                nc.variables['ubar_%s' % open_boundary][tind] = ecco.barotropic
                            
                            else:
                                raise Exception
                        
                        elif ecco.vartype in 'SSH':
                            
                            nc.variables['zeta_%s' % open_boundary][tind] = ecco.dataout
                            nc.variables['bry_time'][tind] = np.float(dtnum)
                        
                        else:
                            
                            varname = ecco.vartype.lower() + '_%s' % open_boundary
                            nc.variables[varname][tind] = ecco.dataout
                    
                    tind += 1
                    #if tind == 5: ssssss
                proceed = False

    
    # Correct volume fluxes and write to boundary file
    print '\nProcessing volume flux correction'
    with netcdf.Dataset(romsbry.romsfile, 'a') as nc:
                    
        bry_times = nc.variables['bry_time'][:]
        boundarylist = []
            
        for bry_ind in np.arange(bry_times.size):
            
            uvbarlist = []
            
            for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):
            
                if 'west' in open_boundary and flag:
                    uvbarlist.append(nc.variables['ubar_west'][bry_ind])
                    
                elif 'east' in open_boundary and flag:
                    uvbarlist.append(nc.variables['ubar_east'][bry_ind])
                
                elif 'north' in open_boundary and flag:
                    uvbarlist.append(nc.variables['vbar_north'][bry_ind])
                
                elif 'south' in open_boundary and flag:
                    uvbarlist.append(nc.variables['vbar_south'][bry_ind])
                    
                if bry_ind == 0 and flag:
                    boundarylist.append(open_boundary)
                    
            fc = bry_flux_corr(boundarylist,
                               chd_bry_surface_areas,
                               chd_bry_total_surface_area,
                               uvbarlist)
            
            print '------ barotropic velocity correction:', fc, 'm/s'
            
            for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):
                            
                if 'west' in open_boundary and flag:
                    nc.variables['u_west'][bry_ind] -= fc
                    nc.variables['ubar_west'][bry_ind] -= fc
                                
                elif 'east' in open_boundary and flag:
                    nc.variables['u_east'][bry_ind] += fc
                    nc.variables['ubar_east'][bry_ind] += fc
                    
                elif 'north' in open_boundary and flag:
                    nc.variables['v_north'][bry_ind] += fc
                    nc.variables['vbar_north'][bry_ind] += fc
                            
                elif 'south' in open_boundary and flag:
                    nc.variables['v_south'][bry_ind] -= fc
                    nc.variables['vbar_south'][bry_ind] -= fc
    
    
    print 'all done'
            
            
      














