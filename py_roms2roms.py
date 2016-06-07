
# -*- coding: utf-8 -*-
# %run py_roms2roms.py

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

Create a ROMS boundary file based on ROMS data

===========================================================================
'''

import netCDF4 as netcdf
import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
import glob as glob
#import matplotlib.nxutils as nx
import time
import scipy.interpolate.interpnd as interpnd
from scipy.interpolate import RectBivariateSpline as rbs
import collections
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime


class vertInterp(object):
    '''Vertical interpolation object based on scipy.ndimage.map_coordinates()
      http://www.scipy.org/SciPyPackages/Ndimage
      http://docs.scipy.org/doc/scipy/reference/ndimage.html
    '''
    def __init__(self, vweights):
        '''
        Input parameter:
            vweights : input; computed by method get_map_coordinate_weights()
        Makes hweights : same size as vweights, columns correspond to
                       x-axis indices
        '''
        self.vweights = vweights
        self.hweights = np.tile(np.arange(vweights.shape[1], dtype=np.float64),
                                         (vweights.shape[0], 1))

    def vert_interp(self, par_var):
        '''
        par_var is 2d (depth, dist) parent variable
        '''
        return nd.map_coordinates(par_var, np.array([self.vweights,
                                                     self.hweights]),
                                                     order=1,
                                                     mode='nearest',
                                                     prefilter=False,
                                                     cval=999999)




class horizInterp(interpnd.CloughTocher2DInterpolator):
    '''

    '''
    def __init__(self, tri, values, fill_value=np.nan,
                 tol=1e-6, maxiter=400):
        interpnd.NDInterpolatorBase.__init__(self, tri.points, values, ndim=2,
                                             fill_value=fill_value)
        self.tri = tri
        self.grad = interpnd.estimate_gradients_2d_global(self.tri, self.values,
                                                          tol=tol, maxiter=maxiter)




class horizInterpRbs(rbs):
    """
    """
    def __init__(self, datalon, datalat, data, kx=1, ky=1):
        self.coeffs = rbs(datalat, datalon, data, kx=kx, ky=ky)

    def rbs_interp(self, romslon, romslat):
        return self.coeffs.ev(romslat, romslon)




class ROMS (object):
    '''
    ROMS class
    '''
    def __init__(self, datafile, model_type):
        '''
        Initialise the ROMS object
        '''
        thetype = str(type(self)).split('.')[-1].split("'")[0]
        print '--- instantiating *%s*' % thetype, datafile
        self.romsfile = datafile
        self.indices = '[self.j0:self.j1, self.i0:self.i1]'
        self.i0 = 0
        self.i1 = None
        self.j0 = 0
        self.j1 = None
        self.k = 0 # to be used as a z index
        self.t = 0 # to be used as a time index
        self.r_earth = 6371315. # Mean earth radius in metres (from scalars.h)
        
        # Flag indicating ROMS, SODA, CFSR, etc.
        assert isinstance(model_type, str), 'model_type must be a string'
        assert model_type in ('ROMS', 'CFSR', 'SODA', 'Mercator', 'Ecco'
                              ), "model_type must be one of 'ROMS', 'CFSR', 'SODA', 'Mercator'"
        self.model_type = model_type
        # To be used for handling a grid that crosses
        #   zero degree meridian
        self.zero_crossing = False
        self.fix_zero_crossing = False

        # An index along either x or y dimension
        self.ij = None

        # Used by CfsrData class
        #  Values taken from Roms_tools ("get_dqdsst.m")
        # Specific heat of atmosphere.
        self.Cp = 1004.8
        # Sensible heat transfer coefficient (stable condition)
        self.Ch = 0.66e-3
        # Latent heat transfer coefficient (stable condition)
        self.Ce = 1.15e-3
        # Emissivity coefficient
        self.eps = 0.985 # see Roms_tools function "as_consts.m"
        # Stefan constant
        self.Stefan = 5.6697e-8
        # Latent heat of vaporisation (J.kg-1)
        self.L1 = 2.5008e6
        self.L2 = 2.3e3
        # Kelvin
        self.Kelvin = 273.15


    def read_nc(self, varname, indices="[:]"):
        '''
        Read data from netcdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : string of index ranges, eg. '[0,:,0]'
        '''
        #print self.romsfile
        try:
            with netcdf.Dataset(self.romsfile) as nc:
                var = eval(''.join(("nc.variables[varname]", indices)))
        except Exception:
            try:
                with netcdf.Dataset(self.romsfile[0]) as nc:
                    var = eval(''.join(("nc.variables[varname]", indices)))
            except Exception:
                raise
        if 'float32' in var.dtype.name:
            return var.astype(np.float64)
        else:
            return var

    def read_nc_mf(self, varname, indices="[:]"):
        '''
        Read data from multi-file netcdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : string of index ranges, eg. '[0,:,0]'
        '''
        try:
            try:
                with netcdf.MFDataset(self.filenames) as nc:
                    var =  eval(''.join(("nc.variables[varname]", indices)))
            except Exception:
                with netcdf.MFDataset(self.filenames, aggdim='TIME') as nc:
                    var =  eval(''.join(("nc.variables[varname]", indices)))
        except Exception:
            raise
        if 'float32' in var.dtype.name:
            return var.astype(np.float64)
        else:
            return var


    def read_nc_at_index(self, varname, ind):
        '''
        Read data from multi-file netcdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : string of index ranges, eg. '[0,:,0]'
        '''
        try:
            with netcdf.Dataset(self.romsfile) as nc:
                return eval(''.join(("nc.variables[varname]", indices)))
        except Exception:
            try:
                with netcdf.Dataset(self.romsfile[0]) as nc:
                    return eval(''.join(("nc.variables[varname]", indices)))
            except Exception:
                raise Exception

    def read_nc_att(self, varname, att):
        '''
        Read data attribute from netcdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          att : string of attribute, eg. 'valid_range'
        '''
        try:
            with netcdf.Dataset(self.romsfile) as nc:
                return eval(''.join(("nc.variables[varname].", att)))
        except Exception:
            try:
                with netcdf.Dataset(self.romsfile[0]) as nc:
                    return eval(''.join(("nc.variables[varname].", att)))
            except Exception:
                raise Exception


    def read_dim_size(self, dim):
        '''
        Read dimension size from netcdf file
          dim : dimension ('time', 'lon_u', etc) to read
        '''
        try:
            with netcdf.Dataset(self.romsfile) as nc:
                #return len(eval("nc.dimensions[dim]"))
                return len(nc.dimensions[dim])
        except Exception:
            raise


    def list_of_variables(self):
        '''
        '''
        not_done = True
        try:
            while not_done:
                try:
                    with netcdf.Dataset(self.romsfile) as nc:
                        not_done = False
                        keys = nc.variables.keys()
                except:
                    time.sleep(0.5)
            return keys
        # NOTE: this below may break at some point; not_done clause not added
        except Exception:
            try:
                with netcdf.Dataset(self.romsfile[0]) as nc:
                    return nc.variables.keys()
            except Exception:
                raise Exception
        return self


    def half_interp(self, h_one, h_two):
        '''
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        '''
        return ne.evaluate('0.5 * (h_one + h_two)')



    def rho2u_2d(self, rho_in):
        '''
        Convert a 2D field at rho points to a field at u points
        '''
        def _r2u(rho_in, Lp):
            u_out = rho_in[:, :Lp - 1]
            u_out += rho_in[:, 1:Lp]
            u_out *= 0.5
            return u_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        Mshp, Lshp = rho_in.shape
        return _r2u(rho_in, Lshp)


    def rho2u_3d(self, rho_in):
        '''
        Convert a 3D field at rho points to a field at u points
        Calls rho2u_2d
        '''
        def levloop(rho_in):
            Nlevs, Mshp, Lshp = rho_in.shape
            rho_out = np.zeros((Nlevs, Mshp, Lshp-1))
            for k in np.arange(Nlevs):
                 rho_out[k] = self.rho2u_2d(rho_in[k])
            return rho_out
        assert rho_in.ndim == 3, 'rho_in must be 3d'
        return levloop(rho_in)


    def rho2v_2d(self, rho_in):
        '''
        Convert a 2D field at rho points to a field at v points
        '''
        def _r2v(rho_in, Mp):
            v_out = rho_in[:Mp - 1]
            v_out += rho_in[1:Mp]
            v_out *= 0.5
            return v_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        Mshp, Lshp = rho_in.shape
        return _r2v(rho_in, Mshp)


    def rho2v_3d(self, rho_in):
        '''
        Convert a 3D field at rho points to a field at v points
        Calls rho2v_2d
        '''
        def levloop(rho_in):
            Nlevs, Mshp, Lshp = rho_in.shape
            rho_out = np.zeros((Nlevs, Mshp-1, Lshp))
            for k in np.arange(Nlevs):
                 rho_out[k] = self.rho2v_2d(rho_in[k])
            return rho_out
        assert rho_in.ndim == 3, 'rho_in must be 3d'
        return levloop(rho_in)



    def u2rho_2d(self, u_in):
        '''
        Convert a 2D field at u points to a field at rho points
        '''
        def _uu2ur(uu_in, Mp, Lp):
            L, Lm = Lp - 1, Lp - 2
            u_out = np.zeros((Mp, Lp))
            u_out[:, 1:L] = 0.5 * (u_in[:, 0:Lm] + \
                                   u_in[:, 1:L])
            u_out[:, 0] = u_out[:, 1]
            u_out[:, L] = u_out[:, Lm]
            return u_out.squeeze()

        assert u_in.ndim == 2, 'u_in must be 2d'
        Mp, Lp = u_in.shape
        return _uu2ur(u_in, Mp, Lp+1)


    def u2rho_3d(self, u_in):
        '''
        Convert a 3D field at u points to a field at rho points
        Calls u2rho_2d
        '''
        def _levloop(u_in):
            Nlevs, Mshp, Lshp = u_in.shape
            u_out = np.zeros((Nlevs, Mshp, Lshp+1))
            for Nlev in np.arange(Nlevs):
                u_out[Nlev] = self.u2rho_2d(u_in[Nlev])
            return u_out
        assert u_in.ndim == 3, 'u_in must be 3d'
        return _levloop(u_in)


    def v2rho_2d(self, v_in):
        '''
        Convert a 2D field at v points to a field at rho points
        '''
        def _vv2vr(v_in, Mp, Lp):
            M, Mm = Mp - 1, Mp - 2
            v_out = np.zeros((Mp, Lp))
            v_out[1:M] = 0.5 * (v_in[:Mm] + \
                                   v_in[1:M])
            v_out[0] = v_out[1]
            v_out[M] = v_out[Mm]
            return v_out.squeeze()

        assert v_in.ndim == 2, 'v_in must be 2d'
        Mp, Lp = v_in.shape
        return _vv2vr(v_in, Mp+1, Lp)


    def v2rho_3d(self, v_in):
        '''
        Convert a 3D field at v points to a field at rho points
        Calls v2rho_2d
        '''
        def levloop(v_in):
            Nlevs, Mshp, Lshp = v_in.shape
            v_out = np.zeros((Nlevs, Mshp+1, Lshp))
            for Nlev in np.arange(Nlevs):
                v_out[Nlev] = self.v2rho_2d(v_in[Nlev])
            return v_out
        assert v_in.ndim == 3, 'v_in must be 3d'
        return levloop(v_in)


    def rotate(self, u_in, v_in, **kwargs):
        """
        Rotate velocity vectors
        'angle' from gridfile
        """
        if kwargs.has_key('ob'):
            if kwargs['ob'] in 'east':
                angle = self.angle()[:,-1]
            elif kwargs['ob'] in 'west':
                angle = self.angle()[:,0]
            elif kwargs['ob'] in 'north':
                angle = self.angle()[-1]
            elif kwargs['ob'] in 'south':
                angle = self.angle()[0]
            else:
                raise Exception
        else:
            angle = self.angle()
        cosa = np.cos(kwargs['sign'] * angle)
        sina = np.sin(kwargs['sign'] * angle)
        u_out = (u_in * cosa) + (v_in * sina)
        v_out = (v_in * cosa) - (u_in * sina)
        return u_out, v_out


    def get_fillmask_cof(self, mask):
        '''Create (i, j) point arrays for good and bad data.
            # Bad data are marked by the fill_value, good data elsewhere.
        '''
        # CHANGED Jan 14 to include *mask* argument
        igood = np.vstack(np.where(mask == 1)).T
        ibad  = np.vstack(np.where(mask == 0)).T
        tree = sp.cKDTree(igood)
        # Get the k closest points to the bad points
        # distance is squared
        try:
            dist, iquery = tree.query(ibad, k=4, p=2)
        except:
            try:
                dist, iquery = tree.query(ibad, k=3, p=2)
            except:
                try:
                    dist, iquery = tree.query(ibad, k=2, p=2)
                except:
                    dist, iquery = tree.query(ibad, k=1, p=2)
        self.fillmask_cof = np.array([dist, iquery, igood, ibad])
        return self


    def fillmask(self, x, mask, weights=False):
        '''
        Fill missing values in an array with an average of nearest
        neighbours
        From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        Input:
          x : 2-d array to be filled
          mask : 2-d mask (0s & 1s) same shape as x
        Output:
          x : filled x
          self.fillmask_cof is set
        '''
        assert x.ndim == 2, 'x must be a 2D array.'
        fill_value = 9999.99
        x[mask == 0] = fill_value

        if isinstance(weights, np.ndarray):
            dist, iquery, igood, ibad = weights
        else:
            self.get_fillmask_cof(mask)
            dist, iquery, igood, ibad = self.fillmask_cof

        # Create a normalised weight, the nearest points are weighted as 1.
        #   Points greater than one are then set to zero
        weight = dist / (dist.min(axis=1)[:,np.newaxis] * np.ones_like(dist))
        np.place(weight, weight > 1., 0.)

        # Multiply the queried good points by the weight, selecting only
        #  the nearest points.  Divide by the number of nearest points
        #  to get average
        xfill = weight * x[igood[:,0][iquery], igood[:,1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)

        # Place average of nearest good points, xfill, into bad point locations
        x[ibad[:,0], ibad[:,1]] = xfill

        if isinstance(weights, bool):
            return x, np.array([dist, iquery, igood, ibad])
        else:
            return x


    def proj2gnom(self, ignore_land_points=False, gtype='rho', index_str=None, M=None):
        '''
        Use premade Basemap instance for Gnomonic projection
          of lon, lat.
            ignore_land_points : if True returns only lon, lat from sea points.
            gtype : grid type, one of 'rho', 'u' or 'v'
            index_str : specifies a boundary.
            M : Child basemap obj must be passed in for parent projection
        '''
        def remove_masked_points(lon, lat, mask):
            lon, lat = lon[mask == True], lat[mask == True]
            return lon, lat

        if index_str is not None:
            glon = eval(''.join(('self.lon()', index_str)))
            glat = eval(''.join(('self.lat()', index_str)))

            if ignore_land_points:
                if 'rho' in gtype:
                    glon, glat = remove_masked_points(glon, glat,
                             eval(''.join(('self.maskr()', index_str))))
                elif 'u' in gtype:
                    glon, glat = remove_masked_points(glon, glat,
                             eval(''.join(('self.umask()', index_str))))
                elif 'v' in gtype:
                    glon, glat = remove_masked_points(glon, glat,
                             eval(''.join(('self.vmask()', index_str))))

        elif index_str is None and ignore_land_points is True: # exclude masked points
            glon, glat = remove_masked_points(self.lon(), self.lat(), self.maskr())
        else:
            glon, glat = self.lon(), self.lat()
        if M is None:
            glon, glat = self.M(glon, glat)
        else:
            #print 'dddd'
            glon, glat = M(glon, glat)
        self.points = np.array([glon.ravel(),glat.ravel()]).T
        return self






    def make_kdetree(self):
        ''' Make a parent kde tree that will enable selection
        minimum numbers of indices necessary to parent grid for
        successful interpolation to child grid
        Requires self.points from def proj2gnom
        '''
        self.kdetree = sp.cKDTree(self.points)
        if not hasattr(sp.ckdtree.cKDTree, "query_ball_tree"):
            print '------ cKDTree.query_ball_tree not found (update of scipy recommended)'
            self.kdetree = sp.KDTree(self.points)
        return self




    def make_gnom_transform(self):
        '''
        Create Basemap instance for Gnomonic projection
        Return the transformation, M
        '''
        self.M = Basemap(projection = 'gnom',
                         lon_0=self.lon().mean(), lat_0=self.lat().mean(),
                         llcrnrlon=self.lon().min(), llcrnrlat=self.lat().min(),
                         urcrnrlon=self.lon().max(), urcrnrlat=self.lat().max())
        return self


    def child_contained_by_parent(self, child_grid):
        '''
        Check that no child data points lie outside of the
        parent domain.
        '''
        tri = sp.Delaunay(self.points) # triangulate full parent
        tn = tri.find_simplex(child_grid.points)
        assert not np.any(tn == -1), 'Error: detected child data points outside parent domain'
        print '------ parent domain suitable for interpolation'
        return self


    def set_subgrid(self, other, k=4):
        '''
        Set indices to parent subgrid
          Parameter:
            other : another (child) RomsGrid instance
        '''
        def kdt(lon, lat, limits):
            ppoints = np.array([lon.ravel(), lat.ravel()]).T
            ptree = sp.cKDTree(ppoints)
            #print limits
            pindices = ptree.query(limits, k=k)[1]

            iind, jind = np.array([], dtype=int), np.array([], dtype=int)
            for pind in pindices.ravel():
                j, i = np.unravel_index(pind, lon.shape)
                iind = np.r_[iind, i]
                jind = np.r_[jind, j]
            return iind, jind

        if self.zero_crossing is True and 'ROMS' not in self.model_type:
            '''
            Used by pysoda2roms when there is a zero crossing,
              eg. at western Med.
            '''
            def half_limits(lon, lat):
                return np.array([np.array([lon.min(), lon.max(),
                                           lon.max(), lon.min()]),
                                 np.array([lat.min(), lat.min(),
                                           lat.max(), lat.max()])]).T

            # Get bounds for -ve part of grid
            lat = other.lat()[other.lon() < 0.]
            lon = other.lon()[other.lon() < 0.] + 360.
            limits = half_limits(lon, lat)
            iind, jind = kdt(self._lon, self._lat, limits)
            self.i1 = iind.min()
            j10, j11 = jind.min(), jind.max()

            # Get bounds for +ve part of grid
            lat = other.lat()[other.lon() >= 0.]
            lon = other.lon()[other.lon() >= 0.]
            limits = half_limits(lon, lat)
            iind, jind = kdt(self._lon, self._lat, limits)
            self.i0 = iind.max()
            j20, j21 = jind.min(), jind.max()

            self.j0 = np.min([j10, j20])
            self.j1 = np.max([j11, j21])
            #self.fix_zero_crossing = True

        else:
            ''' Used for pyroms2roms, and pysoda2roms when
                there is no zero crossing
            '''
            if np.alltrue(other.lon() < 0.) and np.alltrue(self._lon >= 0.):
                self._lon -= 360.
            iind, jind = kdt(self._lon, self._lat, other.limits())
            self.i0, self.i1 = iind.min(), iind.max()
            self.j0, self.j1 = jind.min(), jind.max()
        return self


    def _get_barotropic_velocity(self, baroclinic_velocity, cell_depths):
        '''
        '''
        sum_baroclinic = (baroclinic_velocity * cell_depths).sum(axis=0)
        total_depth = cell_depths.sum(axis=0)
        sum_baroclinic /= total_depth
        return sum_baroclinic

    def get_barotropic_velocity(self, baroclinic_velocity, cell_depths):
        '''
        Input:
          baroclinic_velocity
          cell_depths
        '''
        return self._get_barotropic_velocity(baroclinic_velocity, cell_depths)


    def set_barotropic(self): #, open_boundary):
        '''
        '''
        self.barotropic = self._get_barotropic_velocity(self.dataout,
                                                        self.romsgrd.scoord2dz())
        return self



class RomsGrid (ROMS):
    '''
    RomsGrid class (inherits from ROMS class)
    '''
    def __init__(self, filename, sigma_params, model_type):
        '''

        '''
        super(RomsGrid, self).__init__(filename, model_type)
        #self.indices = '[self.j0:self.j1, self.i0:self.i1]'
        self.grid_file = filename
        self._lon = self.read_nc('lon_rho')#, indices=self.indices)
        self._lat =  self.read_nc('lat_rho')#, indices=self.indices)
        self._pm = self.read_nc('pm')#, indices=self.indices)
        self._pn = self.read_nc('pn')#, indices=self.indices)
        self._maskr = self.read_nc('mask_rho')#, indices=self.indices)
        self._angle = self.read_nc('angle')#, indices=self.indices)
        self._h = self.read_nc('h')#, indices=self.indices)
        self._hraw = self.read_nc('hraw')#, indices=self.indices)
        self._f = self.read_nc('f')#, indices=self.indices)
        self._uvpmask()
        self.theta_s = np.double(sigma_params['theta_s'])
        self.theta_b = np.double(sigma_params['theta_b'])
        self.hc = np.double(sigma_params['hc'])
        self.N = np.double(sigma_params['N'])


    def lon(self):   return self._lon[self.j0:self.j1, self.i0:self.i1]
    def lat(self):   return self._lat[self.j0:self.j1, self.i0:self.i1]
    def pm(self):    return self._pm[self.j0:self.j1, self.i0:self.i1]
    def pn(self):    return self._pn[self.j0:self.j1, self.i0:self.i1]
    def maskr(self): return self._maskr[self.j0:self.j1, self.i0:self.i1]
    def angle(self): return self._angle[self.j0:self.j1, self.i0:self.i1]
    def h(self):     return self._h[self.j0:self.j1, self.i0:self.i1]
    def hraw(self):  return self._hraw[self.j0:self.j1, self.i0:self.i1]
    def f(self):     return self._f[self.j0:self.j1, self.i0:self.i1]

    def idata(self):
        return np.nonzero(self.maskr().ravel() == 1.)[0]

    def imask(self):
        return np.nonzero(self.maskr().ravel() == 0.)[0]


    def _uvpmask(self):
        '''
        Get mask at u, v, psi points
        '''
        try:
            self._umask = self.read_nc('mask_u')
        except:
            Mp, Lp  = self.maskr().shape
            M       = Mp - 1
            L       = Lp - 1
            self._umask   = self.maskr()[:, :L] * self.maskr()[:, 1:Lp]
            self._vmask   = self.maskr()[:M]   * self.maskr()[1:Mp]
            self._pmask = self._umask[:M] * self._umask[1:Mp]
        else:
            self._vmask = self.read_nc('mask_v')
            self._pmask = self.read_nc('mask_psi')
        return self


    def umask(self):
        return self._umask[self.j0:self.j1, self.i0:self.i1]


    def vmask(self):
        return self._vmask[self.j0:self.j1, self.i0:self.i1]


    def pmask(self):
        return self._pmask


    def mask3d(self):
        '''
        3d stack of mask same size as N
        '''
        return np.tile(self.maskr(), (np.int(self.N), 1, 1))


    def umask3d(self):
        '''
        3d stack of umask same size as N
        '''
        return np.tile(self.umask(), (np.int(self.N), 1, 1))


    def vmask3d(self):
        '''
        3d stack of vmask same size as N
        '''
        return np.tile(self.vmask(), (np.int(self.N), 1, 1))


    def boundary(self):
        '''
        Return lon,lat of perimeter around a ROMS grid
        '''
        lon = np.hstack((self.lon()[0:, 0], self.lon()[-1, 1:-1],
                         self.lon()[-1::-1, -1], self.lon()[0, -2::-1]))
        lat = np.hstack((self.lat()[0:, 0], self.lat()[-1, 1:-1],
                         self.lat()[-1::-1, -1], self.lat()[0, -2::-1]))
        return lon, lat


    def VertCoordType(self):
        nc = netcdf.Dataset(self.grdfile, 'r')
        var = nc.VertCoordType
        nc.close()
        return var

    def title(self):
        nc = netcdf.Dataset(self.grdfile, 'r')
        var = nc.title
        nc.close()
        return var


    def check_zero_crossing(self):
        if np.logical_and(np.any(self.lon() < 0.),
                          np.any(self.lon() >= 0.)):
            self.zero_crossing = True


    def _scoord2z(self, point_type, zeta, alpha, beta):
        """
        z = scoord2z(h, theta_s, theta_b, hc, N, point_type, scoord, zeta)
        scoord2z finds z at either rho or w points (positive up, zero at rest surface)
        h          = array of depths (e.g., from grd file)
        theta_s    = surface focusing parameter
        theta_b    = bottom focusing parameter
        hc         = critical depth
        N          = number of vertical rho-points
        point_type = 'r' or 'w'
        scoord     = 'new2008' :new scoord 2008, 'new2006' : new scoord 2006,
                      or 'old1994' for Song scoord
        zeta       = sea surface height
        message    = set to False if don't want message
        """
        def CSF(self, sc):
            '''
            Allows use of theta_b > 0 (July 2009)
            '''
            if self.theta_s > 0.:
                csrf = ((1. - np.cosh(self.theta_s * sc)) /
                        (np.cosh(self.theta_s) - 1.))
            else:
                csrf = -sc**2
            sc1 = csrf + 1.
            if self.theta_b > 0.:
                Cs = ((np.exp(self.theta_b * sc1) - 1.) /
                      (np.exp(self.theta_b) - 1.) - 1.)
            else:
                Cs = csrf
            return Cs
        #
        try: self.scoord
        except: self.scoord = 'new2008'
        N = self.N.copy()
        cff1 = 1. / np.sinh(self.theta_s)
        cff2 = 0.5 / np.tanh(0.5 * self.theta_s)
        sc_w = np.arange(-1., 1. / N, 1. / N, dtype=np.float64)
        sc_r = 0.5 * (sc_w[1:] + sc_w[:-1])
        if 'w' in point_type:
            sc = sc_w
            N += 1. # add a level
        else:
            sc = sc_r
        Cs = (1. - self.theta_b) * cff1 * np.sinh(self.theta_s * sc)  \
                 + self.theta_b * (cff2 * np.tanh(self.theta_s * (sc + 0.5)) - 0.5)
        z  = np.empty((N,) + self.h().shape, dtype=np.float64)
        if self.scoord in 'new2008':
            Cs = CSF(self, sc)
        if self.scoord in 'new2006' or self.scoord in 'new2008':
            ds   = 1. / N
            hinv = 1. / (self.h() + self.hc)
            cff_r   = self.hc * sc
            for k in np.arange(N) + 1.:
                cff1_r    = Cs[k-1]
                z[k-1, :] = zeta + (zeta + self.h()) * (cff_r[k-1] + cff1_r * self.h()) * hinv
        elif self.scoord in 'old1994':
            hinv = 1. / self.h()
            cff  = self.hc * (sc - Cs)
            cff1 = Cs
            cff2 = sc + 1
            for k in np.arange(N) + 1:
                z0      = cff[k-1] + cff1[k-1] * self.h()
                z[k-1, :] = z0 + zeta * (1. + z0 * hinv)
        else:
            you_really_dont_want_to_end_up_here
        return z.squeeze(), Cs


    def scoord2z_r(self, zeta=0., alpha=0., beta=1.):
        '''
        Depths at vertical rho points
        '''
        return self._scoord2z('r', zeta=zeta, alpha=alpha, beta=beta)[0]


    def Cs_r(self, zeta=0., alpha=0., beta=1.):
        '''
        S-coordinate stretching curves at rho points
        '''
        return self._scoord2z('r', zeta=zeta, alpha=alpha, beta=beta)[1]


    def scoord2z_w(self, zeta=0., alpha=0., beta=1.):
        '''
        Depths at vertical w points
        '''
        return self._scoord2z('w', zeta=zeta, alpha=alpha, beta=beta)[0]


    def Cs_w(self, zeta=0., alpha=0., beta=1.):
        '''
        S-coordinate stretching curves at w points
        '''
        return self._scoord2z('w', zeta=zeta, alpha=alpha, beta=beta)[1]

    def _set_dz_rho_points(self, zeta=0., alpha=0., beta=1):
        """
        Set depths of sigma layers at rho points, 3d matrix.
        """
        dz = self._scoord2z('w', zeta=zeta, alpha=alpha, beta=beta)[0]
        self._dz_rho_points = dz[1:] - dz[:-1]


    def scoord2dz(self, zeta=0., alpha=0., beta=1.):
        """
        dz at rho points, 3d matrix, depths of sigma layers
        """
        dz = self._scoord2z('w', zeta=zeta, alpha=alpha, beta=beta)[0]
        return dz[1:] - dz[:-1]

    def scoord2dz_u(self, zeta=0., alpha=0., beta=1.):
        '''
        dz at u points, 3d matrix, depths of sigma layers
        '''
        dz = self.scoord2dz(zeta=0., alpha=0., beta=1.)
        return self.rho2u_3d(dz)

    def scoord2dz_v(self, zeta=0., alpha=0., beta=1.):
        '''
        dz at v points, 3d matrix, depths of sigma layers
        '''
        dz = self.scoord2dz(zeta=0., alpha=0., beta=1.)
        return self.rho2v_3d(dz)



    def set_bry_dx(self):
        '''
        Set dx for all 4 boundaries
        '''
        self.set_dx_east()
        self.set_dx_west()
        self.set_dx_north()
        self.set_dx_south()
        return self


    def set_dx_east(self):
        '''
        Set dx in m along eastern boundary
        '''
        self.dx_east = np.reciprocal(0.5 * (self._pn[:,-1] + self._pn[:,-2]))
        return self

    def set_dx_west(self):
        '''
        Set dx in m along western boundary
        '''
        self.dx_west = np.reciprocal(0.5 * (self._pn[:, 0] + self._pn[:, 1]))
        return self

    def set_dx_south(self):
        '''
        Set dx in m along southern boundary
        '''
        self.dx_south = np.reciprocal(0.5 * (self._pm[0] + self._pm[1]))
        return self

    def set_dx_north(self):
        '''
        Set dx in m along northern boundary
        '''
        self.dx_north = np.reciprocal(0.5 * (self._pm[-1] + self._pm[-2]))
        return self


    def set_bry_maskr(self):
        '''
        Set mask for all 4 boundaries
        '''
        self.set_maskr_east()
        self.set_maskr_west()
        self.set_maskr_north()
        self.set_maskr_south()
        return self


    def set_maskr_east(self):
        '''
        Set dx in m along eastern boundary
        '''
        self.maskr_east = self._maskr[:, -1]
        return self

    def set_maskr_west(self):
        '''
        Set dx in m along western boundary
        '''
        self.maskr_west = self._maskr[:, 0]
        return self

    def set_maskr_south(self):
        '''
        Set dx in m along southern boundary
        '''
        self.maskr_south = self._maskr[0]
        return self

    def set_maskr_north(self):
        '''
        Set maskr along northern boundary
        '''
        self.maskr_north = self._maskr[-1]
        return self


    def set_bry_areas(self):
        '''
        Set area for all 4 boundaries
        '''
        dz = self.scoord2z_w()[1:] - self.scoord2z_w()[:-1]
        self._set_area_east(dz)
        self._set_area_west(dz)
        self._set_area_north(dz)
        self._set_area_south(dz)
        return self


    def _set_area_east(self, dz):
        '''
        Set area in m² along eastern boundary
        '''
        dz_east = dz[:,:,-1]
        dx_east = np.tile(self.dx_east, (dz_east.shape[0], 1))
        self.area_east = dx_east * dz_east
        return self

    def _set_area_west(self, dz):
        '''
        Set area in m² along western boundary
        '''
        dz_west = dz[:,:,0]
        dx_west = np.tile(self.dx_west, (dz_west.shape[0], 1))
        self.area_west = dx_west * dz_west
        return self

    def _set_area_south(self, dz):
        '''
        Set area in m² along southern boundary
        '''
        dz_south = dz[:,0]
        dx_south = np.tile(self.dx_south, (dz_south.shape[0], 1))
        self.area_south = dx_south * dz_south
        return self

    def _set_area_north(self, dz):
        '''
        Set area in m² along northern boundary
        '''
        dz_north = dz[:,-1]
        dx_north = np.tile(self.dx_north, (dz_north.shape[0], 1))
        self.area_north = dx_north * dz_north
        return self


    def limits(self):
        '''

        '''
        return np.array([np.array([self.lon().min(), self.lon().max(),
                                   self.lon().max(), self.lon().min()]),
                         np.array([self.lat().min(), self.lat().min(),
                                   self.lat().max(), self.lat().max()])]).T


    def get_map_coordinate_weights(self, czr_bry, pzr_bry):
        '''
        Calculate the weights required for the vertical interpolation
        with vertInterp (map_coordinates)
        '''
        assert pzr_bry.shape[1] == czr_bry.shape[1], \
            'pzr_bry and czr_bry must have the same lengths'

        czr_bry = np.float128(czr_bry)
        pzr_bry = np.float128(pzr_bry)
        weights = np.full_like(czr_bry, self.N - 1, dtype=np.float64)

        # Interpolate parent vertical indices at parent depths
        # to child depths
        for i in np.arange(pzr_bry.shape[1]):

            akima = si.Akima1DInterpolator(pzr_bry[:, i], np.arange(pzr_bry.shape[0]))
            akima.extrapolate = True
            akima.extend = True
            weights[:, i] = akima(czr_bry[:, i])

        return weights


    #def get_map_coordinate_weightsXXXXXXXX(self, czr_bry, pzr_bry):
        #'''
        #Calculate the weights required for the vertical interpolation
        #with vertInterp (map_coordinates)
        #'''
        #assert pzr_bry.shape[1] == czr_bry.shape[1], \
            #'pzr_bry and czr_bry must have the same lengths'

        ## For the present purposes, we are assuming that parent-child topo matching
        ## has been applied along the open boundaries
        ## THIS IS WRONG
        #'''assert np.abs((czr_bry[-1][0]-pzw_bry[-1][0])).max() <= 10., \
            #'max. depth difference between parent and child exceeds 10 m;\n \
             #has the child topo been matched?'''

        #czr_bry = np.float128(czr_bry)
        #pzr_bry = np.float128(pzr_bry)

        #weights = np.full_like(czr_bry, self.N-1, dtype=np.float64)

        ## Loop along the boundary
        #for i in np.arange(pzr_bry.shape[1]):
            #weight_tmp = np.array([], dtype=np.float64)
            #dzp = np.diff(pzr_bry[:,i])

            ## Loop from bottom to surface of parent
            ##for k, dz in enumerate(dzp):
            #for k in np.arange(pzr_bry.shape[0]):

                ##dz =

                ## If the child has deeper bottom layer than the parent
                #if k == 0 and np.any(pzr_bry[k,i] > czr_bry[:,i]): # bottom layer

                    #choices = np.nonzero(czr_bry[:,i] < pzr_bry[k,i])[0]
                    #print 'deeper child bottom depth: chd %s, par %s' %(czr_bry[k,i], pzr_bry[k,i])
                    #print 'choices bottom', k, choices, choices.shape

                ## If the child has shallower top layer than the parent
                #elif k == pzr_bry.shape[0] - 1 and np.any(pzr_bry[k,i] < czr_bry[:,i]): # top layer

                    #choices = np.nonzero(czr_bry[:,i] > pzr_bry[k,i])[0]
                    #print 'choices top', k, choices, choices.shape

                ## Everything in between
                #else:

                    #choices = np.nonzero(np.logical_and(czr_bry[:,i] < pzr_bry[k,i],
                                                        #czr_bry[:,i] >= pzr_bry[k-1,i]))[0]
                    #print 'choices', k, choices, choices.shape

                #for choice in choices:
                    ##weight     = k + np.abs(np.diff((pzr_bry[k, i], czr_bry[choice, i]))) / dz
                    #weight = k + ((pzr_bry[k,i] - czr_bry[choice,i]) / dz)
                    #if k==0:print 'kkk===0000 weight',weight
                    #weight_tmp = np.append(weight_tmp, weight)

            #print 'weight_tmp', weight_tmp
            #print 'weight_tmp.shape', weight_tmp.shape, i
            #try: weights[:-1,i] = weight_tmp
            #except: weights[:,i] = weight_tmp
            #print 'weights ---------', weights[:,i]
            #print '--------'
        #return weights





class WestGrid(RomsGrid):
    '''
    Modify the RomsGrid class to point at only the western boundary
    '''
    def __init__(self, filename, sigma_params, model_type):
        '''

        '''
        super(WestGrid, self).__init__(filename, sigma_params, model_type)

    def lon(self):
        return np.atleast_2d(self.read_nc('lon_rho',  indices='[:,0]'))

    def lat(self):
        return np.atleast_2d(self.read_nc('lat_rho',  indices='[:,0]'))

    def maskr(self):
        return np.atleast_2d(self.read_nc('mask_rho',  indices='[:,0]'))

    def angle(self):
        return self.read_nc('angle',  indices='[:,0]')

    def pm(self):
        return self.read_nc('pm',  indices='[:,0]')

    def pn(self):
        return self.read_nc('pn',  indices='[:,0]')

    def h(self):
        return self.read_nc('h',  indices='[:,0]')

    #def dz_rho_points(self):
        #"""
        #Sigma layer depths (dz) at `rho` points, 3d matrix
        #"""
        #try: return self._dz_rho_points
        #except Exception:
            #self._set_dz_rho_points()
            #return self._dz_rho_points


    def dz_v(self):
        """
        Sigma layer depths (dz) at `v` points, 2d matrix
        """
        try:
            return self._dz_v
        except Exception:
            self._dz_v = self.scoord2dz()[:,:-1] + self.scoord2dz()[:,1:]
            self._dz_v *= 0.5
            return self._dz_v

    #def scoord2z_r(self, zeta=0., alpha=0., beta=1.):
        #return self._scoord2z('r', zeta=zeta, alpha=alpha, beta=beta)[0]


class EastGrid(RomsGrid):
    '''
    Modify the RomsGrid class to point at only the eastern boundary
    '''
    def __init__(self, filename, sigma_params, model_type):
        '''

        '''
        super(EastGrid, self).__init__(filename, sigma_params, model_type)

    def lon(self):
        return np.atleast_2d(self.read_nc('lon_rho',  indices='[:,-1]'))

    def lat(self):
        return np.atleast_2d(self.read_nc('lat_rho',  indices='[:,-1]'))

    def maskr(self):
        return np.atleast_2d(self.read_nc('mask_rho',  indices='[:,-1]'))

    def angle(self):
        return self.read_nc('angle',  indices='[:,-1]')

    def pm(self):
        return self.read_nc('pm',  indices='[:,-1]')

    def pn(self):
        return self.read_nc('pn',  indices='[:,-1]')

    def h(self):
        return self.read_nc('h',  indices='[:,-1]')

    def dz_rho_points(self):
        """
        Sigma layer depths (dz) at `rho` points, 3d matrix
        """
        try: return self._dz_rho_points
        except Exception:
            self._set_dz_rho_points()
            return self._dz_rho_points

    def dz_v(self):
        """
        Sigma layer depths (dz) at `v` points, 2d matrix
        """
        try:
            return self._dz_v
        except Exception:
            self._dz_v = self.scoord2dz()[:,:-1] + self.scoord2dz()[:,1:]
            self._dz_v *= 0.5
            return self._dz_v

    #def scoord2z_r(self, zeta=0., alpha=0., beta=1.):
        #return self._scoord2z('r', zeta=zeta, alpha=alpha, beta=beta)[0]



class NorthGrid(RomsGrid):
    '''
    Modify the RomsGrid class to point at only the northern boundary
    '''
    def __init__(self, filename, sigma_params, model_type):
        '''

        '''
        super(NorthGrid, self).__init__(filename, sigma_params, model_type)

    def lon(self):
        return np.atleast_2d(self.read_nc('lon_rho',  indices='[-1]'))

    def lat(self):
        return np.atleast_2d(self.read_nc('lat_rho',  indices='[-1]'))

    def maskr(self):
        return np.atleast_2d(self.read_nc('mask_rho',  indices='[-1]'))

    def angle(self):
        return self.read_nc('angle',  indices='[-1]')

    def pm(self):
        return self.read_nc('pm',  indices='[-1]')

    def pn(self):
        return self.read_nc('pn',  indices='[-1]')

    def h(self):
        return self.read_nc('h',  indices='[-1]')

    #def dz_rho_points(self):
        #"""
        #Sigma layer depths (dz) at `rho` points, 3d matrix
        #"""
        #try: return self._dz_rho_points
        #except Exception:
            #self._set_dz_rho_points()
            #return self._dz_rho_points

    def dz_u(self):
        """
        Sigma layer depths (dz) at `u` points, 2d matrix
        """
        try:
            return self._dz_u
        except Exception:
            self._dz_u = self.scoord2dz()[:,:-1] + self.scoord2dz()[:,1:]
            self._dz_u *= 0.5
            return self._dz_u


    #def scoord2z_r(self, zeta=0., alpha=0., beta=1.):
        #return self._scoord2z('r', zeta=zeta, alpha=alpha, beta=beta)[0]


class SouthGrid(RomsGrid):
    '''
    Modify the RomsGrid class to point at only the northern boundary
    '''
    def __init__(self, filename, sigma_params, model_type):
        '''

        '''
        super(SouthGrid, self).__init__(filename, sigma_params, model_type)

    def lon(self):
        return np.atleast_2d(self.read_nc('lon_rho',  indices='[0]'))

    def lat(self):
        return np.atleast_2d(self.read_nc('lat_rho',  indices='[0]'))

    def maskr(self):
        return np.atleast_2d(self.read_nc('mask_rho',  indices='[0]'))

    def angle(self):
        return self.read_nc('angle',  indices='[0]')

    def pm(self):
        return self.read_nc('pm',  indices='[0]')

    def pn(self):
        return self.read_nc('pn',  indices='[0]')

    def h(self):
        return self.read_nc('h',  indices='[0]')

    #def dz_rho_points(self):
        #"""
        #Sigma layer depths (dz) at `rho` points, 3d matrix
        #"""
        #try: return self._dz_rho_points
        #except Exception:
            #self._set_dz_rho_points()
            #return self._dz_rho_points

    def dz_u(self):
        """
        Sigma layer depths (dz) at `u` points, 2d matrix
        """
        try:
            return self._dz_u
        except Exception:
            self._dz_u = self.scoord2dz()[:,:-1] + self.scoord2dz()[:,1:]
            self._dz_u *= 0.5
            return self._dz_u

    #def scoord2z_r(self, zeta=0., alpha=0., beta=1.):
        #return self._scoord2z('r', zeta=zeta, alpha=alpha, beta=beta)[0]













class RomsData (ROMS):
    '''
    ROMS data class
    '''

    def temp(self, indices=None):
        return self.read_nc('temp', indices)

    def salt(self, indices=None):
        return self.read_nc('salt', indices)

    def u(self, indices=None):
        return self.read_nc('u', indices)

    def v(self, indices=None):
        return self.read_nc('v', indices)

    def zeta(self, indices=None):
        return self.read_nc('zeta', indices)

    def ubar(self, indices=None):
        return self.read_nc('ubar', indices)

    def vbar(self, indices=None):
        return self.read_nc('vbar', indices)

    def ocean_time(self, indices=None):
        return self.read_nc('ocean_time', indices)


    def create_bry_nc(self, grdobj, obc_dict, cycle, fillval, madeby):
        '''
        Create a new boundary file based on dimensions from
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
        nc.created  = datetime.now().isoformat()
        nc.type     = 'ROMS boundary file produced by %s.py' %madeby
        nc.grd_file = grdobj.romsfile
        nc.hc       = grdobj.hc
        nc.theta_s  = grdobj.theta_s
        nc.theta_b  = grdobj.theta_b
        nc.Tcline   = grdobj.hc
        nc.Cs_r     = grdobj.Cs_r()
        nc.Cs_w     = grdobj.Cs_w()
        nc.VertCoordType = 'NEW'
        try: # see pysoda2roms
            nc.first_file = self.first_file
            nc.last_file = self.last_file
        except Exception:
            pass

        # Dimensions
        nc.createDimension('xi_rho',   grdobj.lon().shape[1])
        nc.createDimension('xi_u',     grdobj.lon().shape[1]-1)
        nc.createDimension('eta_rho',  grdobj.lon().shape[0])
        nc.createDimension('eta_v',    grdobj.lon().shape[0]-1)
        nc.createDimension('s_rho',    grdobj.N)
        nc.createDimension('s_w',      grdobj.N+1)
        nc.createDimension('bry_time', None)
        nc.createDimension('one',      1)

        # Create the variables and write...
        nc.createVariable('theta_s', 'f', ('one'), zlib=True)
        nc.variables['theta_s'].long_name = 'S-coordinate surface control parameter'
        nc.variables['theta_s'].units     = 'nondimensional'
        nc.variables['theta_s'][:]        = grdobj.theta_s

        nc.createVariable('theta_b', 'f', ('one'), zlib=True)
        nc.variables['theta_b'].long_name = 'S-coordinate bottom control parameter'
        nc.variables['theta_b'].units     = 'nondimensional'
        nc.variables['theta_b'][:]        = grdobj.theta_b

        nc.createVariable('Tcline', 'f', ('one'), zlib=True)
        nc.variables['Tcline'].long_name  = 'S-coordinate surface/bottom layer width'
        nc.variables['Tcline'].units      = 'meters'
        nc.variables['Tcline'][:]         = grdobj.hc

        nc.createVariable('hc', 'f', ('one'), zlib=True)
        nc.variables['hc'].long_name      = 'S-coordinate parameter, critical depth'
        nc.variables['hc'].units          = 'meters'
        nc.variables['hc'][:]             = grdobj.hc

        nc.createVariable('sc_r', 'f8', ('s_rho'), zlib=True)
        nc.variables['sc_r'].long_name    = 'S-coordinate at RHO-points'
        nc.variables['sc_r'].units        = 'nondimensional'
        nc.variables['sc_r'].valid_min    = -1.
        nc.variables['sc_r'].valid_max    = 0.

        nc.createVariable('Cs_r', 'f8', ('s_rho'), zlib=True)
        nc.variables['Cs_r'].long_name    = 'S-coordinate stretching curves at RHO-points'
        nc.variables['Cs_r'].units        = 'nondimensional'
        nc.variables['Cs_r'].valid_min    = -1.
        nc.variables['Cs_r'].valid_max    = 0.
        nc.variables['Cs_r'][:]           = grdobj.Cs_r()

        nc.createVariable('Cs_w', 'f8', ('s_w'), zlib=True)
        nc.variables['Cs_w'].long_name    = 'S-coordinate stretching curves at w-points'
        nc.variables['Cs_w'].units        = 'nondimensional'
        nc.variables['Cs_w'].valid_min    = -1.
        nc.variables['Cs_w'].valid_max    = 0.
        nc.variables['Cs_w'][:]           = grdobj.Cs_w()

        nc.createVariable('bry_time', 'f8', ('bry_time'), zlib=True)
        nc.variables['bry_time'].long_name = 'time for boundary data'
        nc.variables['bry_time'].units     = 'days'
        '''if cycle:
            nc.variables['bry_time'].cycle_length = cycle # days'''



        # dictionary for the prognostic variables
        prog_vars = OrderedDict()
        prog_vars['temp_'] = ['rho2',
                              ' boundary potential temperature',
                              'Celsius']
        prog_vars['salt_'] = ['rho2',
                              ' boundary salinity',
                              'psu']
        prog_vars['u_']    = ['u2',
                              ' boundary u-momentum component',
                              'meters second-1']
        prog_vars['v_']    = ['v2',
                              ' boundary v-momentum component',
                              'meters second-1']
        prog_vars['ubar_'] = ['u1',
                              ' boundary vertically integrated u-momentum component',
                              'meters second-1']
        prog_vars['vbar_'] = ['v1',
                              ' boundary vertically integrated v-momentum component',
                              'meters second-1']
        prog_vars['zeta_'] = ['rho1',
                              ' boundary sea surface height',
                              'meters']

        for boundary, flag in zip(obc_dict.keys(), obc_dict.values()):

            if flag:

                varlabel = '%sern'   % boundary

                for key, value in zip(prog_vars.keys(), prog_vars.values()):

                    if 'rho2' in value[0]:
                        if boundary=='east' or boundary=='west':
                            dims = ('bry_time', 's_rho', 'eta_rho')
                        elif boundary=='south' or boundary=='north':
                            dims = ('bry_time', 's_rho', 'xi_rho')

                    elif 'u2' in value[0]:
                        if boundary=='south' or boundary=='north':
                            dims = ('bry_time', 's_rho', 'xi_u')
                        else:
                            dims = ('bry_time', 's_rho', 'eta_rho')

                    elif 'v2' in value[0]:
                        if boundary=='east' or boundary=='west':
                            dims = ('bry_time', 's_rho', 'eta_v')
                        else:
                            dims = ('bry_time', 's_rho', 'xi_rho')

                    elif 'u1' in value[0]:
                        if boundary=='south' or boundary=='north':
                            dims = ('bry_time', 'xi_u')
                        else:
                            dims = ('bry_time', 'eta_rho')

                    elif 'v1' in value[0]:
                        if boundary=='east' or boundary=='west':
                            dims = ('bry_time', 'eta_v')
                        else:
                            dims = ('bry_time', 'xi_rho')

                    elif 'rho1' in value[0]:
                        if boundary=='east' or boundary=='west':
                            dims = ('bry_time', 'eta_rho')
                        elif boundary=='south' or boundary=='north':
                            dims = ('bry_time', 'xi_rho')

                    else: error

                    varname  = ''.join((key, '%s' % boundary))
                    # !! fill_value= option removed as Sasha's code doesn't like it...
                    nc.createVariable(varname, 'f8', dims, zlib=True)#, fill_value=fillval)
                    nc.variables[varname].long_name = varlabel + value[1]
                    nc.variables[varname].units     = value[2]

        nc.close()






def get_unique_indices(tree_list):
    '''
    sp.KDTree.query_ball_tree returns a list of paired indices
      1. flatten
      2. sort
      3. reshape
      4. get unique indices
    '''
    def flatten(l):
        '''
        http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
        NOTE: returns a generator
       '''
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
                for sub in flatten(el):
                    yield sub
            else:
                yield el
    indices = np.array([], dtype=np.int)
    # Empty lists will be ignored
    for flat in flatten(tree_list):
        indices = np.r_[indices, flat]
    indices = indices.reshape((0.5 * indices.size, 2))
    indices = indices.tolist()
    indices.sort()
    # http://article.gmane.org/gmane.comp.python.numeric.general/29160/match=unique+pairs
    unique_index = [i for i, x in enumerate(indices) if not i or x != indices[i-1]]
    return unique_index





def bry_flux_corr(boundarylist,
                  chd_bry_surface_areas,
                  chd_bry_total_surface_area,
                  uvbarlist):
    '''

    '''
    flux_corr = np.array(0, dtype=np.float128)

    for ind, bry in enumerate(boundarylist):

        uvbar = uvbarlist[ind]
        bry_surf_ar = chd_bry_surface_areas[ind]
        corr = (uvbar * bry_surf_ar).sum(dtype=np.float128)

        if bry in 'south':
            flux_corr += corr
        elif bry in 'north':
            flux_corr -= corr
        elif bry in 'east':
            flux_corr -= corr
        elif bry in 'west':
            flux_corr += corr

    flux_corr /= chd_bry_total_surface_area
    return flux_corr



def debug0(plon, plat, pmask, cblon, cblat):
    '''
    Check the result of the set_subgrid operation
    '''
    plt.figure()
    plt.pcolormesh(plon, plat, pmask)
    plt.plot(cblon, cblat, 'w')
    plt.axis('image')
    plt.show()


def debug1(plon, plat, pmask, par_bry_indices,
           cboundary):
    '''
    Pcolor of child boundaries on top of the parent
    mask. Parent points used for each boundary triangulation
    are shaded
    '''
    for par_bry_index in par_bry_indices:
        pmask.flat[par_bry_index] = 0.5
    plt.figure()
    plt.title("Parent triangulation to be based on green areas defined by\n    \
        'par_bry_indices' over each child boundary; increase 'cof' if necessary",
        size=11)
    plt.pcolormesh(plon, plat, pmask, zorder=2)
    cblon, cblat = cboundary
    plt.plot(cblon, cblat, 'r', lw=1, zorder=3)
    plt.axis('image')
    plt.xlim(cblon.min() - 1, cblon.max() + 1)
    plt.ylim(cblat.min() - 1, cblat.max() + 1)
    plt.show()


def debug2(ind, boundary):
    '''
    Plots of parent and child variables==var('temp', 'salt', 'u', 'v')
    at index==ind
    '''
    plt.figure()
    plt.figtext(0.5, 0.95,
        'interpolated vars at index %s on %sern boundary' %(ind, boundary),
        ha='center')
    plt.subplot(141)
    plt.title('temp')
    plt.plot(htemp[1:-1,ind], pzr_bry[bryind1][:,ind], '.-b')
    plt.plot(ctemp[:,ind],    czr_bry[bryind1][:,ind], '.-r')
    plt.grid()
    plt.subplot(142)
    plt.title('salt')
    plt.plot(hsalt[1:-1,ind], pzr_bry[bryind1][:,ind], '.-b')
    plt.plot(csalt[:,ind],    czr_bry[bryind1][:,ind], '.-r')
    plt.grid()
    plt.subplot(143)
    plt.title('u')
    plt.plot(hu[1:-1,ind], pzr_bry[bryind1][:,ind], '.-b')
    plt.plot(cu[:,ind],    czr_bry[bryind1][:,ind], '.-r')
    plt.grid()
    plt.subplot(144)
    plt.title('v')
    plt.plot(hv[1:-1,ind], pzr_bry[bryind1][:,ind], '.-b')
    plt.plot(cv[:,ind],    czr_bry[bryind1][:,ind], '.-r')
    plt.grid()
    plt.show()



def prepare_child_roms(romsgrd):
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=False)
    romsgrd.make_kdetree()
    return romsgrd



def prepare_parent_roms(roms, balldist):
    roms.proj2gnom(ignore_land_points=False, M=roms.romsgrd.M)
    roms.child_contained_by_parent(roms.romsgrd)
    roms.make_kdetree()#.get_fillmask_cofs()
    ballpoints = roms.kdetree.query_ball_tree(roms.romsgrd.kdetree, r=balldist)
    roms.ball = np.array(np.array(ballpoints).nonzero()[0])
    roms.tri = sp.Delaunay(roms.points[roms.ball])
    proceed = True
    return roms, proceed


if __name__ == '__main__':

    '''
    pyrom2roms (Python version of roms2roms written in Matlab).

    Differences from roms2roms:
      Parent/child topo matching along the child grid boundary is automatic
      in pyroms2roms (NOT YET IMPLEMENTED)

    TO DO:
      Using a KDE tree it should be possible to compute all open
       boundaries together; ie treat them as a single boundary; this
       should speed thing up significantly
      Start work on extra_variables

    Evan Mason 2012
    '''


    #_USER DEFINED VARIABLES_______________________________________
    #par_dir     = '../'
    #par_dir     = '/shared/emason/runs2009/na_2009_7pt5km/'
    #par_dir     = '/marula/emason/runs2009/na_2009_7pt5km/'
    #par_dir     = '/marula/emason/runs2009/cb_2009_3km_42/'
    par_dir     = '/marula/emason/runs2009/gc_2009_1km_60/'
    #par_grd     = 'roms_grd_NA2009_7pt5km.nc'
    #par_dir     = '/marula/emason/runs2012/na_7pt5km/'
    #par_dir     = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #par_grd     = 'roms_grd_NA2009_7pt5km.nc'
    #par_dir      = '/marula/emason/runs2012/MedSea5_intann_monthly/'
    #par_dir    = '/marula/emason/runs2013/AlbSea_1pt5/'

    #par_grd    = 'grd_MedSea5.nc'
    #par_grd    = 'roms_grd_NA2009_7pt5km.nc'
    #par_grd    = 'cb_2009_3km_grd_smooth.nc'
    par_grd     = 'gc_2009_1km_grd_smooth.nc'
    #par_grd    = 'grd_AlbSea_1pt5km.nc'

    if 'roms_grd_NA2009_7pt5km.nc' in par_grd:
        par_sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
    elif 'cb_2009_3km_grd_smooth.nc' in par_grd:
        par_sigma_params = dict(theta_s=6, theta_b=2, hc=120, N=42)
    elif 'gc_2009_1km_grd_smooth.nc' in par_grd:
        par_sigma_params = dict(theta_s=6, theta_b=2, hc=120, N=60)

    #chd_dir     = '../'
    #chd_dir     = '/nas02/emason/runs2009/cb_2009_3km_42/'
    #chd_dir     = '/home/emason/toto/yeray/'
    #chd_grd     = 'cb_2009_3km_grd_smooth.nc'
    #chd_grd     = 'grd_canbas2.5.nc'
    #chd_dir     = '/marula/emason/runs2013/canwake4km/'
    #chd_dir     = '/marula/emason/runs2013/cb_3km_2013_intann/'
    #chd_grd     = 'grd_canwake4km.nc'
    #chd_grd     = 'cb_2009_3km_grd_smooth.nc'
    #chd_dir    = '/marula/emason/runs2013/AlbSea_1pt25/'
    #chd_dir    = '/marula/emason/runs2013/cart500/'
    #chd_dir    = '/marula/emason/runs2014/canbas4/'
    #chd_dir    = '/marula/emason/runs2014/gc1km_2014/'
    chd_dir     = '/marula/emason/runs2015/GranCan250/'
    #chd_dir     = '/marula/emason/runs2015/GranCan250_nocoast/'

    #chd_grd    = 'grd_AlbSea_1pt25.nc'
    #chd_grd    = 'grd_cart500.nc'
    #chd_grd    = 'grd_canbas4.nc'
    #chd_grd    = 'grd_gc1km_2014.nc'
    chd_grd     = 'grd_gc250_coast.nc'
    #chd_grd     = 'grd_gc250.nc'

    if 'some_grid.nc' in chd_grd:
        chd_sigma_params = dict(theta_s=6, theta_b=0, hc=120, N=32)
        bry_filename = 'bry_na75_mercator.nc'
        obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed

    elif 'grd_canbas4.nc' in chd_grd:
        chd_sigma_params = dict(theta_s=6., theta_b=0, hc=120, N=32)
        bry_filename = 'bry_canbas4.nc'
        obc_dict = dict(south=1, east=0, north=1, west=1) # 1=open, 0=closed

    elif 'grd_gc1km_2014.nc' in chd_grd:
        chd_sigma_params = dict(theta_s=6., theta_b=2, hc=120, N=45)
        bry_filename = 'bry_gc1km_2014.nc'
        obc_dict = dict(south=1, east=0, north=1, west=1) # 1=open, 0=closed

    elif chd_grd in ('grd_gc250_coast.nc', 'grd_gc250.nc'):
        chd_sigma_params = dict(theta_s=6., theta_b=2, hc=120, N=60)
        bry_filename = 'bry_gc250_coast_Y2.nc'
        obc_dict = dict(south=1, east=1, north=1, west=1) # 1=open, 0=closed

    # Boundary file
    bry_cycle = 0     # number of days between records or, set to 0 for no cycle
    #bry_filename = 'bry_canwake4km.nc' # bry filename
    #bry_filename = 'bry_cb_3km_2013_intann.nc' # bry filename
    #bry_filename = 'bry_AlbSea_1pt25_20030101.nc' # bry filename
    #bry_filename = 'bry_cart500.nc' # bry filename
    bry_type     = 'roms_avg' # parent file to read data from,
                              # usually one of 'roms_avg', 'roms_his' or 'roms_rst'
    first_file   = '0510' # first/last avg/his file,
    last_file    = '0630' #   e.g., '0050' gives roms_avg.0050.nc
    first_rec    =  1     # desired record no. from first avg/his file
    last_rec     =  30     # desired record no. from last avg/his file

    roms_vars = ['zeta', 'temp', 'salt', 'u']


    # Dictionary with any additional variables and their dimensions
    # that we might want to process, these might be sediments, bio, etc...
    extra_variables = OrderedDict()


    balldist = 50000. # meters



    #_END USER DEFINED VARIABLES_______________________________________

    plt.close('all')

    assert len(first_file) == 4, 'first_file must be a length four string'
    assert len(last_file) == 4,  'last_file must be a length four string'

    fillval = 9999.99

    # Lists of boundary labels and indices
    #boundaries = ['south', 'east', 'north', 'west']
    #chd_bry_indices = ['[:,0]', '[:,:,-1]', '[:,-1]', '[:,:,0]'] # note, to 3d array

    ## List of standard ROMS variables for boundary forcing
    #variables = ['temp', 'salt', 'u', 'v', 'zeta', 'ubar', 'vbar']

    ## Empty lists to be filled later...
    ## Once filled, their lengths will be == sum(obcflag)
    #par_bry_indices = []
    #par_bry_tris = []
    #chd_bry_points  = []
    #chd_bry_masks = []
    #pzr_bry = []
    #czr_bry = []
    #vinterp_weights = []
    #chd_bry_surface_areas = []


    # Initialise RomsGrid objects for both parent and child grids
    pgrd = RomsGrid(''.join((par_dir, par_grd)), par_sigma_params, 'ROMS')
    cgrd = RomsGrid(''.join((chd_dir, chd_grd)), chd_sigma_params, 'ROMS')
    cgrd.set_bry_dx()
    cgrd.set_bry_maskr()
    cgrd.set_bry_areas()

    # Activate flag for zero crossing trickery
    cgrd.check_zero_crossing()
    if cgrd.zero_crossing is True:
        print 'The ROMS domain straddles the zero-degree meridian'
        pgrd.zero_crossing = True

    # Set pgrd indices (i0:i1, j0:j1) for minimal subgrid around chd
    pgrd.set_subgrid(cgrd)
    pgrd.make_gnom_transform()
    #pgrd.proj2gnom(ignore_land_points=False)
    #pgrd.make_kdetree()

    if 0: # check the result of set_subgrid()
        debug0(pgrd.lon(), pgrd.lat(), pgrd.maskr(),
               cgrd.boundary()[0], cgrd.boundary()[1])

    # Get surface areas of open boundaries
    chd_bry_surface_areas = []
    for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):
        if 'west' in open_boundary and flag:
            chd_bry_surface_areas.append(cgrd.area_west.sum(axis=0) * cgrd.maskr_west)
        elif 'east' in open_boundary and flag:
            chd_bry_surface_areas.append(cgrd.area_east.sum(axis=0) * cgrd.maskr_east)
        elif 'south' in open_boundary and flag:
            chd_bry_surface_areas.append(cgrd.area_south.sum(axis=0) * cgrd.maskr_south)
        elif 'north' in open_boundary and flag:
            chd_bry_surface_areas.append(cgrd.area_north.sum(axis=0) * cgrd.maskr_north)

    # Get total surface of open boundaries
    chd_bry_total_surface_area = np.array([area.sum() for area in chd_bry_surface_areas]).sum()




    # Set up a RomsData object for the boundary file
    romsbry = RomsData(chd_dir + bry_filename, 'ROMS')
    romsbry.create_bry_nc(cgrd, obc_dict, bry_cycle, fillval, 'py_roms2roms')


    # Get list of roms parent data files
    roms_files = sorted(glob.glob(par_dir + bry_type + '.????.nc'))


    romsdata = np.ma.zeros((pgrd.N, pgrd.j1-pgrd.j0, pgrd.i1-pgrd.i0))
    romsdatav = romsdata.copy()

    proceed = False


    for open_boundary, flag in zip(obc_dict.keys(), obc_dict.values()):

        if 'west' in open_boundary and flag:
            cgrd_at_bry = WestGrid(''.join((chd_dir, chd_grd)), chd_sigma_params, 'ROMS')
            proceed = True
        elif 'north' in open_boundary and flag:
            cgrd_at_bry = NorthGrid(''.join((chd_dir, chd_grd)), chd_sigma_params, 'ROMS')
            proceed = True
        elif 'east' in open_boundary and flag:
            cgrd_at_bry = EastGrid(''.join((chd_dir, chd_grd)), chd_sigma_params, 'ROMS')
            proceed = True
        elif 'south' in open_boundary and flag:
            cgrd_at_bry = SouthGrid(''.join((chd_dir, chd_grd)), chd_sigma_params, 'ROMS')
            proceed = True
        else:
            proceed = False

        #if romsgrd.zero_crossing is True:
        #cgrd_at_bry._lon[cgrd_at_bry._lon < 0] += 360.
        cgrd_at_bry = prepare_child_roms(cgrd_at_bry)
        cromsdata = np.ma.zeros((cgrd_at_bry.scoord2z_r().shape))

        pgrd.romsgrd = cgrd_at_bry
        pgrd, junk = prepare_parent_roms(pgrd, balldist)

        # Get parent zr (pzr_bry) at child points
        for k in np.arange(pgrd.N):
            pzr_bry_tmp = horizInterp(pgrd.tri, pgrd.scoord2z_r()[k].flat[pgrd.ball])(cgrd_at_bry.points)
            try:
                pzr_bry = np.vstack((pzr_bry, pzr_bry_tmp))
            except Exception:
                pzr_bry = pzr_bry_tmp.copy()

        # Prepare weights for vertical interpolations with vertInterp
        vertical_weights = cgrd_at_bry.get_map_coordinate_weights(cgrd_at_bry.scoord2z_r(), pzr_bry)

        del pzr_bry # delete (if not, affects try/except above)
        vinterp = vertInterp(vertical_weights)

        if proceed:

            print '\n--- processing %sern boundary' %open_boundary
            for roms_var in roms_vars:

                print '\nProcessing variable *%s*' %roms_var

                #if 'u' in ecco_var:
                    #romsbry_v = RomsData(roms_file, 'ROMS', 'v', cgrd_at_bry,
                                     #i0=pgrd.i0, i1=pgrd.i1, j0=pgrd.j0, j1=pgrd.j1)
                    #romsbry_v = prepare_roms(romsbry_v, balldist)

                tind = 0 # index for writing records to bry file
                active = False # flag to roms_files for loop
                fm_rwght = False # flag, if False then compute fillmask weights

                ctracer = np.zeros((cgrd_at_bry.scoord2z_r().shape))
                czeta = np.zeros((ctracer.shape[1]))
                cu = np.zeros((cgrd_at_bry.N, czeta.size))
                cv = np.zeros((cgrd_at_bry.N, czeta.size))
                cubar = np.zeros_like(czeta)
                cvbar = np.zeros_like(czeta)

                #ptracer = np.zeros((pgrd.N+2, czeta.size))
                #pu = np.zeros((pgrd.N+2, czeta.size))
                #pv = np.zeros((pgrd.N+2, czeta.size))
                ptracer = np.zeros((pgrd.N, czeta.size))
                pu = np.zeros((pgrd.N, czeta.size))
                pv = np.zeros((pgrd.N, czeta.size))

                '''
                Start main loop over roms_files list here
                '''
                for roms_file in roms_files:

                    if first_file in roms_file:
                        active = True

                    if active:

                        print 'Opening file', roms_file
                        rfile = RomsData(roms_file, 'ROMS')

                        with netcdf.Dataset(roms_file) as nc:
                            # Loop over ocean_time
                            ot_loop_start = time.time()
                            for dataind, ocean_time in enumerate(nc.variables['ocean_time'][:]):

                                #print '---processing record', dataind + 1

                                if 'zeta' in roms_var:

                                    romsdata[0] = nc.variables['zeta'][dataind,pgrd.j0:pgrd.j1,pgrd.i0:pgrd.i1]
                                    # Read in variables and fill masked areas
                                    if 'rho_weight' in locals():
                                        romsdata[0] = rfile.fillmask(romsdata[0], pgrd.maskr(), rho_weight)
                                    else:
                                        romsdata[0], rho_weight = rfile.fillmask(romsdata[0], pgrd.maskr())

                                    czeta[:] = horizInterp(pgrd.tri, romsdata[0].flat[pgrd.ball])(cgrd_at_bry.points)


                                elif 'u' in roms_var:

                                    romsdata[:] = pgrd.u2rho_3d(nc.variables['u'][dataind,:,pgrd.j0:pgrd.j1,pgrd.i0:pgrd.i1-1])
                                    romsdatav[:] = pgrd.v2rho_3d(nc.variables['v'][dataind,:,pgrd.j0:pgrd.j1-1,pgrd.i0:pgrd.i1])
                                    for k in np.arange(pgrd.N):
                                        if 'u_weight' in locals():
                                            romsdata[k] = rfile.fillmask(romsdata[k], pgrd.maskr(), u_weight)
                                            romsdatav[k] = rfile.fillmask(romsdatav[k], pgrd.maskr(), v_weight)
                                        else:
                                            romsdata[k], u_weight = rfile.fillmask(romsdata[k], pgrd.maskr())
                                            romsdatav[k], v_weight = rfile.fillmask(romsdatav[k], pgrd.maskr())
                                        romsdata[k], romsdatav[k] = pgrd.rotate(romsdata[k], romsdatav[k], sign=-1) # rotate to north
                                        #pu[k+1] = horizInterp(pgrd.tri, romsdata[k].flat[pgrd.ball])(cgrd_at_bry.points)
                                        #pv[k+1] = horizInterp(pgrd.tri, romsdatav[k].flat[pgrd.ball])(cgrd_at_bry.points)
                                        pu[k] = horizInterp(pgrd.tri, romsdata[k].flat[pgrd.ball])(cgrd_at_bry.points)
                                        pv[k] = horizInterp(pgrd.tri, romsdatav[k].flat[pgrd.ball])(cgrd_at_bry.points)
                                    # Prepare for vertical interpolations
                                    #pu[0], pu[-1] = pu[1], pu[-2]
                                    #pv[0], pv[-1] = pv[1], pv[-2]
                                    cu[:] = vinterp.vert_interp(pu)
                                    cv[:] = vinterp.vert_interp(pv)
                                    for k in np.arange(cgrd_at_bry.N):
                                        cu[k], cv[k] = cgrd_at_bry.rotate(cu[k], cv[k], sign=1)



                                else: # tracers

                                    romsdata[:] = nc.variables[roms_var][dataind,:,pgrd.j0:pgrd.j1,pgrd.i0:pgrd.i1]
                                    for k in np.arange(pgrd.N):
                                        romsdata[k] = rfile.fillmask(romsdata[k], pgrd.maskr(), rho_weight)
                                        #ptracer[k+1] = horizInterp(pgrd.tri, romsdata[k].flat[pgrd.ball])(cgrd_at_bry.points)
                                        ptracer[k] = horizInterp(pgrd.tri, romsdata[k].flat[pgrd.ball])(cgrd_at_bry.points)
                                    # Prepare for vertical interpolations
                                    #ptracer[0], ptracer[-1] = ptracer[1], ptracer[-2]
                                    ctracer[:] = vinterp.vert_interp(ptracer)


                                # Write to boundary file
                                with netcdf.Dataset(romsbry.romsfile, 'a') as ncbry:

                                    if 'zeta' in roms_var:

                                        ncbry.variables['zeta_%s' %open_boundary][tind] = czeta
                                        ocean_time /= 86400.
                                        ncbry.variables['bry_time'][tind] = ocean_time

                                    elif 'u' in roms_var:

                                        # Get barotropic velocities and save
                                        if open_boundary in ('east', 'west'):
                                            cv[:,:-1] = 0.5 * (cv[:,:-1] + cv[:,1:])
                                            cubar[:] = cgrd.get_barotropic_velocity(cu, cgrd_at_bry.scoord2dz())
                                            cvbar[:-1] = cgrd.get_barotropic_velocity(cv[:,:-1], cgrd_at_bry.dz_v())
                                            ncbry.variables['u_%s' %open_boundary][tind] = cu
                                            ncbry.variables['ubar_%s' %open_boundary][tind] = cubar
                                            ncbry.variables['v_%s' %open_boundary][tind] = cv[:,:-1]
                                            ncbry.variables['vbar_%s' %open_boundary][tind] = cvbar[:-1]

                                        elif open_boundary in ('north', 'south'):
                                            cu[:,:-1] = 0.5 * (cu[:,:-1] + cu[:,1:])
                                            cubar[:-1] = cgrd.get_barotropic_velocity(cu[:,:-1], cgrd_at_bry.dz_u())
                                            cvbar[:] = cgrd.get_barotropic_velocity(cv, cgrd_at_bry.scoord2dz())
                                            ncbry.variables['u_%s' %open_boundary][tind] = cu[:,:-1]
                                            ncbry.variables['ubar_%s' %open_boundary][tind] = cubar[:-1]
                                            ncbry.variables['v_%s' %open_boundary][tind] = cv
                                            ncbry.variables['vbar_%s' %open_boundary][tind] = cvbar

                                        else:
                                            Exception
                                    else:

                                        ncbry.variables['%s_%s' %(roms_var, open_boundary)][tind] = ctracer


                                    tind += 1



                    if last_file in roms_file:
                        active = False


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
