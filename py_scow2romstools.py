# %run py_scow2romstools.py

'''
Put SCOW wind speeds and SST into Romstools format
'''


from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne
import scipy.interpolate as si
import scipy.spatial as sp
import datetime
import calendar
from py_roms2roms import horizInterp



def fillmask_kdtree(x, mask, k=4, weights=False):
    '''
    Fill missing values in an array with an average of nearest  
    neighbours.
    From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
    '''
    assert x.ndim == 2, 'x must be a 2D array.'
    fill_value = 9999
    x[mask == 0] = fill_value
        
    if not weights:
        '''Create (i, j) point arrays for good and bad data.
           Bad data is marked by the fill_value, good data elsewhere.'''
        igood = np.vstack(np.where(x!=fill_value)).T
        ibad  = np.vstack(np.where(x==fill_value)).T

        # create a tree for the bad points, the points to be filled
        tree = sp.cKDTree(igood)

        # get the four closest points to the bad points
        # here, distance is squared
        dist, iquery = tree.query(ibad, k=k, p=2)
    else:
        dist, iquery, igood, ibad = weights
        
    '''create a normalised weight, the nearest points are weighted as 1.
       Points greater than one are then set to zero.'''        
    weight = dist/(dist.min(axis=1)[:,np.newaxis] * np.ones_like(dist))
    weight[weight>1.] = 0

    '''multiply the queried good points by the weight, selecting only  
       the nearest points.  Divide by the number of nearest points
       to get average.'''
    xfill = weight * x[igood[:,0][iquery], igood[:,1][iquery]]
    xfill = (xfill/weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)

    # place average of nearest good points, xfill, into bad point locations.
    x[ibad[:,0], ibad[:,1]] = xfill

    if not weights:
        return x, [dist, iquery, igood, ibad]
    else:
        return x



def make_file(savedir, romstools_file, lon_scow, lat_scow, mask_scow):
    '''
    '''
    print '--- Creating new file', savedir + romstools_file
    thevar = romstools_file.replace('.cdf', '')
    # Global attributes
    with Dataset(savedir + romstools_file, 'w', format='NETCDF3_CLASSIC') as nc:
        nc.created  = datetime.datetime.now().isoformat()
        
        # Dimensions
        nc.createDimension('X', lon_scow.size)
        nc.createDimension('Y', lat_scow.size)
        nc.createDimension('T', 12)
        
        # Create the variables and write...
        nc.createVariable('X', 'f8', ('X'))
        nc.variables['X'].long_name = 'longitude'
        nc.variables['X'].units = 'degrees_east'
        nc.variables['X'][:] = lon_scow
        
        nc.createVariable('Y', 'f8', ('Y'))
        nc.variables['Y'].long_name = 'latitude'
        nc.variables['Y'].units = 'degrees_north'
        nc.variables['Y'][:] = lat_scow
        
        nc.createVariable('T', 'f', ('T'))
        nc.variables['T'].long_name = 'time in months'
        nc.variables['T'].units = 'months since 01-Jan'
        nc.variables['T'][:] = np.arange(0.5, 12.5)
        
        nc.createVariable(thevar, 'f8', ('T', 'Y', 'X'), fill_value=fillval)
        nc.variables[thevar].long_name = 'time in months'
        nc.variables[thevar].units = 'months since 01-Jan'
        nc.variables[thevar].missing_value = fillval
    
    return




if __name__ == '__main__':
    
    
    
    fillval = -1e10
    
    scow_dir = '/marula/emason/data/SCOW/'

    scow_files = dict(w3 = 'wind_spd_monthly_maps.nc',
                      u3 = 'wind_zonal_monthly_maps.nc',
                      v3 = 'wind_meridional_monthly_maps.nc',
                      sst = 'avhrr_sst_monthly_maps.nc',
                      taux = 'wind_stress_zonal_monthly_maps.nc',
                      tauy = 'wind_stress_meridional_monthly_maps.nc')

    romstools_dir = '/home/emason/roms/Roms_tools/COADS05/'

    romstools_files = ['precip.cdf',
              'longrad.cdf',
              'shortrad.cdf',
              'sat.cdf',
              'rh.cdf',
              'qsea.cdf',
              'sst.cdf',
              'u3.cdf',
              'v3.cdf',
              'w3.cdf',
              'taux.cdf',
              'tauy.cdf']


    savedir = '/marula/emason/data/SCOW/SCOW_romstools/'

    # Get SCOW lon, lat, mask
    with Dataset(scow_dir + scow_files.values()[0]) as nc:
        lon_scow = nc.variables['longitude'][:]
        lat_scow = nc.variables['latitude'][:]
        var_scow = np.ma.empty((lat_scow.size, lon_scow.size))
        var_scow[:] = nc.variables['january'][:]
    
    lon_scow_180 = lon_scow  - 180.
    mask_scow = var_scow == var_scow.min()
    #mask_scow_orig = np.copy(mask_scow)
    #mask_scow[:] = np.concatenate((mask_scow[:,720:], mask_scow[:,:720]), axis=1)
    mask_scow_180 = np.concatenate((mask_scow[:,720:], mask_scow[:,:720]), axis=1)
    
    ## Get Romstools lon, lat, mask
    #with Dataset(romstools_dir + romstools_files[0]) as nc:
        
        #var_rtools = np.ma.empty((lat_rtools.size, lon_rtools.size))
        #var_rtools[:] = nc.variables['precip'][0]
    
    #mask_rtools = np.abs(var_rtools.mask.astype(int) - 1)
    
    
    
    
    for romstools_file in romstools_files:
            
        if romstools_file in ('sst.cdf', 'u3.cdf', 'v3.cdf', 'w3.cdf', 'taux.cdf', 'tauy.cdf'):
        
            make_file(savedir, romstools_file, lon_scow_180, lat_scow, mask_scow_180)
            
            # Using SCOW variables; no interpolation needed
            with Dataset(scow_dir + scow_files[romstools_file.replace('.cdf', '')]) as nc:
                
                with Dataset(savedir + romstools_file, 'a') as ncsave:
                
                    for month in np.arange(1,13):
                    
                        var_scow[:] = nc.variables[calendar.month_name[month].lower()][:]
                        var_scow[:] = np.concatenate((var_scow[:,720:], var_scow[:,:720]), axis=1)
                        np.place(var_scow, mask_scow_180, fillval)
                        #var_scow[:] = np.ma.masked_where(mask_scow == 1, var_scow)
                        ncsave.variables[romstools_file.replace('.cdf', '')][month-1] = var_scow
            
            
        else:
            
            # Using Romstools variables; need interpolation
            with Dataset(romstools_dir + romstools_file) as nc:
                
                lon_rtools = nc.variables['X'][:]
                lat_rtools = nc.variables['Y'][:]
                var_rtools = nc.variables[romstools_file.replace('.cdf', '')][0]
            
                if np.any(lon_rtools > 180.):
                    #var_rtools = np.concatenate((var_rtools[:,360:], var_rtools[:,:360]), axis=1)
                    mask_rtools = np.abs(var_rtools.mask.astype(int) - 1)
                    # Get fillmask weights
                    junk, weights = fillmask_kdtree(var_rtools, mask_rtools)
                    the_lon_scow = lon_scow
                    the_mask_scow = mask_scow
                    #mask_scow[:] = np.concatenate((mask_scow[:,720:], mask_scow[:,:720]), axis=1)
                else:
                    mask_rtools = np.abs(var_rtools.mask.astype(int) - 1)
                    junk, weights = fillmask_kdtree(var_rtools, mask_rtools)
                    the_lon_scow = lon_scow_180
                    the_mask_scow = mask_scow_180
                    #lon_rtools += 180.
                
                make_file(savedir, romstools_file, the_lon_scow, lat_scow, the_mask_scow)
                
                with Dataset(savedir + romstools_file, 'a') as ncsave:
                
                    for month in np.arange(12):
                        
                        var_rtools = nc.variables[romstools_file.replace('.cdf', '')][month]
                        #if np.any(lon_rtools > 180.):
                            #var_rtools[:] = np.concatenate((var_rtools[:,360:], var_rtools[:,:360]), axis=1)
                        var_rtools[:] = fillmask_kdtree(var_rtools, mask_rtools, weights=weights)
                        rbs = si.RectBivariateSpline(lon_rtools, lat_rtools, var_rtools.T)
                        var_scow[:] = rbs(the_lon_scow, lat_scow).T
                        np.place(var_scow, the_mask_scow, fillval)
                        #var_scow = np.ma.masked_where(mask_scow == fillval, var_scow)
                        #var_scow = np.ma.filled(var_scow, fill_value=fillval)
                        ncsave.variables[romstools_file.replace('.cdf', '')][month] = var_scow
                        var_rtools *= 0
                        var_scow *= 0
                        #aaaaaa
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    