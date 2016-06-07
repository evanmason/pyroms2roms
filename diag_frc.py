# -*- coding: utf-8 -*-
# %run diag_frc.py

'''
Compute annual means of two forcing files for comparison
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np

def u2rho_2d(uu_in):
    '''
    Convert a 2D field at u points to a field at rho points
    Checked against Jeroen's u2rho.m
    '''
    def uu2ur(uu_in, Mp, Lp):
        L             = Lp - 1
        Lm            = L  - 1
        u_out         = np.zeros((Mp, Lp))
        u_out[:, 1:L] = 0.5 * (uu_in[:, 0:Lm] + \
                               uu_in[:, 1:L])
        u_out[:, 0]   = u_out[:, 1]
        u_out[:, L]   = u_out[:, Lm]
        return (np.squeeze(u_out))
    # First check to see if has time dimension
    if uu_in.ndim < 3:
        # No time dimension
        Mshp, Lshp = uu_in.shape
        u_out      = uu2ur(uu_in, Mshp, Lshp+1)
    else:
        # Has time dimension
        time, Mshp, Lshp = uu_in.shape
        u_out            = np.zeros((time, Mshp, Lshp+1))
        for t in np.arange(time):
            u_out[t] = uu2ur(uu_in[t], Mshp, Lshp+1)
    return u_out
    
    
def v2rho_2d(vv_in):
    # Convert a 2D field at v points to a field at rho points
    def vv2vr(vv_in, Mp, Lp):
        M             = Mp - 1
        Mm            = M  - 1
        v_out         = np.zeros((Mp, Lp))
        v_out[1:M, :] = 0.5 * (vv_in[0:Mm, :] + \
                               vv_in[1:M, :])
        v_out[0, :]   = v_out[1, :]
        v_out[M, :]   = v_out[Mm, :]
        return (np.squeeze(v_out))
    # First check to see if has time dimension
    if vv_in.ndim < 3:
        # No time dimension
        Mshp, Lshp = vv_in.shape
        v_out      = vv2vr(vv_in, Mshp+1, Lshp)
    else:
        # Has time dimension
        time, Mshp, Lshp = vv_in.shape
        v_out            = np.zeros((time, Mshp+1, Lshp))
        for t in np.arange(time):
            v_out[t] = vv2vr(vv_in[t], Mshp+1, Lshp)
    return v_out

    
    
    
    
    
def get_frc(directory, frcname, varname):
    '''
    '''
    nc = netcdf.Dataset(directory + frcname)
    #lon = nc.variables['lon_rho'][:]
    #lat = nc.variables['lat_rho'][:]
    if 'sustr' in varname or 'svstr' in varname:
        time = nc.variables['sms_time'][:]
    elif 'shflux' in varname:
        time = nc.variables['shf_time'][:]
    elif 'swflux' in varname:
        time = nc.variables['swf_time'][:]
    elif 'SST' in varname or 'dQdSST' in varname:
        time = nc.variables['sst_time'][:]
    elif 'SSS' in varname:
        time = nc.variables['sss_time'][:]
    elif 'srflux' in varname or 'swrad' in varname:
        time = nc.variables['srf_time'][:]
        
    #print time
    if 'sustr' in varname:
        var = 0. * u2rho_2d(nc.variables[varname][0])
    elif 'svstr' in varname:
        var = 0. * v2rho_2d(nc.variables[varname][0])
    else:
        var = 0. * nc.variables[varname][0]
    ind = 0

    for tout in time:
        #print ind
        if 'sustr' in varname:
            var += u2rho_2d(nc.variables[varname][ind])
        elif'svstr' in varname:
            var += v2rho_2d(nc.variables[varname][ind])
        else:
            var += nc.variables[varname][ind]
        ind += 1
    
    nc.close()
    var /= ind
    print 'Averaged over', ind, 'records'
    
    return var
    
    
    

plt.close('all')

directory1 = '/home/emason/runs2012_tmp/MedSea5_R2.5/'
directory2 = '/shared/emason/marula/emason/runs2012/MedSea5/'

file1 = 'frc_intann_MedSea5.nc.TEST'
file2 = 'frc_MedSea5.nc'


grd = 'grd_MedSea5_R2.5.nc'

#var = 'sustr'
#var = 'svstr'
#var = 'shflux'
#var = 'swflux'
var = 'SST'
#var = 'SSS'
#var = 'dQdSST'
#var = 'swrad'



#----------------------------------------------------------------

nc = netcdf.Dataset(directory1 + grd)
lon = nc.variables['lon_rho'][:]
lat = nc.variables['lat_rho'][:]
mask = nc.variables['mask_rho'][:]
nc.close()

var1 = get_frc(directory1, file1, var)
var2 = get_frc(directory2, file2, var)

var1 = np.ma.masked_where(mask == 0, var1)
var2 = np.ma.masked_where(mask == 0, var2)

vmin = np.ma.minimum(var1.min(), var2.min())
vmax = np.ma.maximum(var1.max(), var2.max())


plt.figure()
plt.title(file1)
plt.pcolormesh(lon, lat, var1)
plt.axis('image')
plt.clim(vmin, vmax)
plt.colorbar()


plt.figure()
plt.title(file2)
plt.pcolormesh(lon, lat, var2)
plt.axis('image')
plt.clim(vmin, vmax)
plt.colorbar()


plt.show()







