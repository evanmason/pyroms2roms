# %run py_bry2clim.py

'''
Make weekly bry climatology from interannual bry file produced by
py_ecco2roms.py etc
'''
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import matplotlib.dates as dates
import numexpr as ne


def make_new_bry(directory, bry, nc, frequency):
    '''
    Create a new boundary file based on time dimension of 'frequency'
    '''
    madeby = 'py_bry2clim.py'
    bry = bry.replace('.nc', '_%s_clim.nc' %frequency)
    with Dataset(directory + bry, 'w') as ncnew:
        ncnew.created = dt.datetime.now().isoformat()
        ncnew.type = 'ROMS boundary file produced by %s.py' %madeby
        ncnew.grd_file = nc.grd_file
        ncnew.hc = nc.hc
        ncnew.theta_s = nc.theta_s
        ncnew.theta_b = nc.theta_b
        ncnew.Tcline = nc.hc
        ncnew.Cs_r = nc.Cs_r
        ncnew.Cs_w = nc.Cs_w
        ncnew.VertCoordType = 'NEW'
        try: # see pysoda2roms
            ncnew.first_file = nc.first_file
            ncnew.last_file = nc.last_file
        except Exception:
            pass

        # Dimensions
        ncnew.createDimension('xi_rho', len(nc.dimensions['xi_rho']))
        ncnew.createDimension('xi_u', len(nc.dimensions['xi_u']))
        ncnew.createDimension('eta_rho', len(nc.dimensions['eta_rho']))
        ncnew.createDimension('eta_v', len(nc.dimensions['eta_v']))
        ncnew.createDimension('s_rho', len(nc.dimensions['s_rho']))
        ncnew.createDimension('s_w', len(nc.dimensions['s_w']))
        ncnew.createDimension('bry_time', 52)
        ncnew.createDimension('one', 1)

        # Create the variables and write...
        ncnew.createVariable('theta_s', 'f', ('one'))
        ncnew.variables['theta_s'].long_name = 'S-coordinate surface control parameter'
        ncnew.variables['theta_s'].units     = 'nondimensional'
        ncnew.variables['theta_s'][:]        = nc.theta_s
        
        ncnew.createVariable('theta_b', 'f', ('one'))
        ncnew.variables['theta_b'].long_name = 'S-coordinate bottom control parameter'
        ncnew.variables['theta_b'].units     = 'nondimensional'
        ncnew.variables['theta_b'][:]        = nc.theta_b
        
        ncnew.createVariable('Tcline', 'f', ('one'))
        ncnew.variables['Tcline'].long_name  = 'S-coordinate surface/bottom layer width'
        ncnew.variables['Tcline'].units      = 'meters'
        ncnew.variables['Tcline'][:]         = nc.hc
        
        ncnew.createVariable('hc', 'f', ('one'))
        ncnew.variables['hc'].long_name      = 'S-coordinate parameter, critical depth'
        ncnew.variables['hc'].units          = 'meters'
        ncnew.variables['hc'][:]             = nc.hc
        
        ncnew.createVariable('sc_r', 'f8', ('s_rho'))
        ncnew.variables['sc_r'].long_name    = 'S-coordinate at RHO-points'
        ncnew.variables['sc_r'].units        = 'nondimensional'
        ncnew.variables['sc_r'].valid_min    = -1.
        ncnew.variables['sc_r'].valid_max    = 0.
        
        ncnew.createVariable('Cs_r', 'f8', ('s_rho'))
        ncnew.variables['Cs_r'].long_name    = 'S-coordinate stretching curves at RHO-points'
        ncnew.variables['Cs_r'].units        = 'nondimensional'
        ncnew.variables['Cs_r'].valid_min    = -1.
        ncnew.variables['Cs_r'].valid_max    = 0.
        ncnew.variables['Cs_r'][:]           = nc.Cs_r
        
        ncnew.createVariable('Cs_w', 'f8', ('s_w'))
        ncnew.variables['Cs_w'].long_name    = 'S-coordinate stretching curves at w-points'
        ncnew.variables['Cs_w'].units        = 'nondimensional'
        ncnew.variables['Cs_w'].valid_min    = -1.
        ncnew.variables['Cs_w'].valid_max    = 0.
        ncnew.variables['Cs_w'][:]           = nc.Cs_w
        
        ncnew.createVariable('bry_time', 'f4', ('bry_time'))
        ncnew.variables['bry_time'].long_name = 'time for boundary data'
        ncnew.variables['bry_time'].units = 'days'
        ncnew.variables['bry_time'].cycle_length = 360.
        ncnew.variables['bry_time'][:] = np.arange(3.5, 363.5, 7) * 360 / 364.
        
        return




if __name__ == '__main__':
    
    '''
    py_bry2clim
    
    
    Evan Mason 2014
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    frequency = 'weekly'
    #frequency = 'monthly' # NOT YET IMPLEMENTED
    
    # ROMS information
    directory = '/Users/emason/runs2014/na75_EOF2014/'
    
    bry = 'bry_NA75_SODA_2.1.6_5DAY.nc'
    
    with Dataset(directory + bry) as nc:
        
        #first_file = nc.first_file
        day_zero = dates.date2num(dt.datetime(1985,1,1))
        
        make_new_bry(directory, bry, nc, frequency)
        
        bry_time = nc.variables['bry_time'][:]
        bry_time += day_zero
        
        for varname in nc.variables:
            
            counts = np.zeros(52).astype(np.float64)
            
            print '--- doing variable %s' %varname
            ncobj = nc.variables[varname]
            var = ncobj[:]
            print 'read'
            varsize = var.shape
            if len(varsize) > 2:
                mp, lp = varsize[-2], varsize[-1]
                vartmp = np.zeros((52, mp, lp)).astype(np.float64)
            elif len(varsize) > 1:
                mp = varsize[-1]
                vartmp = np.zeros((52, mp)).astype(np.float64)
            else:
                mp = 0
            
            if mp:
                for i, the_time in enumerate(bry_time):
                
                    weeknum = dt.datetime.isocalendar(dates.num2date(the_time))[1]
                    #print weeknum
                    try:
                        vartmp[weeknum-1] += var[i]
                        counts[weeknum-1] += 1.
                    except Exception:
                        vartmp[weeknum-2] += var[i]
                        counts[weeknum-2] += 1.
                        
                vartmp = vartmp.T
                vartmp[:] = ne.evaluate('vartmp / counts')
                
                with Dataset(directory + bry.replace('.nc', '_%s_clim.nc' %frequency), 'a') as ncnew:
                    
                    ncnew.createVariable(varname, ncobj.datatype, ncobj.dimensions)
                    ncnew.setncattr('long_name', ncobj.getncattr('long_name'))
                    ncnew.setncattr('units', ncobj.getncattr('units'))
                    ncnew.variables[varname][:] = vartmp.T
                
                
                
                
    print 'done!'