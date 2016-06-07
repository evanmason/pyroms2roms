# %run make_pycfsr2frc_six-hourly.py

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

Create a forcing file based on six hourly CFSR data

===========================================================================

'''
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import numpy as np
import time as time

from pycfsr2frc import *






if __name__ == '__main__':
    
    '''
    make_pycfsr2frc_six-hourly

    Prepare six-hourly interannual ROMS surface forcing with, e.g. CFSv2 data (ds094.0) from
    
      http://rda.ucar.edu/pub/cfsr.html
      
      Concatenate with, e.g.:
          ncrcat rh2m.gdas.199912.grb2.nc rh2m.gdas.20????.grb2.nc REL_HUM_2000s_1hr.nc
      and compress output with:
          nc3tonc4 REL_HUM_2000s_1hr.nc REL_HUM_2000s_1hr.nc
    
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
    
    
    Evan Mason, IMEDEA, 2013
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # True for bulk forcing file, else standard dQdSTT forcing file
    bulk = True # variables ()

    
    # CFSR information_________________________________
    
    cfsr_version = 'ds094.0_6hourly'
    cfsr_version = 'ds093.0_6hourly'
    
    #domain = 'S0_N50_W-50_E44'
    domain = 'S0_N60_W-50_E44'
    
    #cfsr_dir = '/shared/emason/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)
    cfsr_dir = '/marula/emason/data/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)
    #cfsr_dir = '/shared/emason/NCEP-CFSR/%s/%s/' %(cfsr_version, domain)

    
    # Filenames and variable names of required CFSR variables
    # Note that these files have been prepared by concatenating the files 
    # dowloaded from CFSR using ncrcat
    if cfsr_version in 'ds093.0_6hourly':
        SSS_file = ('ds093.0_salt_emp.nc',  'SALTY_L160_Avg_11')
        swflux_file = ('ds093.0_salt_emp.nc',  'EMNP_L1_Avg_11')
        prate_file = ('ds093.0_precip_rate.nc',  'PRATE_L1_Avg_1')
        shflux_SW_down_file = ('ds093.0_heat_fluxes.nc',  'DSWRF_L1_Avg_1')
        shflux_SW_up_file = ('ds093.0_heat_fluxes.nc',  'USWRF_L1_Avg_1')
        shflux_LW_down_file = ('ds093.0_heat_fluxes.nc',  'DLWRF_L1_Avg_1')
        shflux_LW_up_file = ('ds093.0_heat_fluxes.nc',  'ULWRF_L1_Avg_1')
        shflux_LATENT_HEAT_file = ('ds093.0_heat_fluxes.nc',  'LHTFL_L1_Avg_1')
        shflux_SENSIBLE_HEAT_file = ('ds093.0_heat_fluxes.nc',  'SHTFL_L1_Avg_1')
        sustr_file = ('ds093.0_wind_tmp.nc',  'U_GRD_L103')
        svstr_file = ('ds093.0_wind_tmp.nc',  'V_GRD_L103')
        SST_file = ('ds093.0_wind_tmp.nc',  'TMP_L1')
        sat_file = ('ds093.0_wind_tmp.nc',  'TMP_L103')
        sap_file = ('ds093.0_pres_spec_hum.nc',  'PRES_L1')
        qair_file = ('ds093.0_pres_spec_hum.nc',  'SPF_H_L103')
        rel_hum_file = ('ds093.0_rel_hum.nc',   'R_H_L103')
    
        # Filenames of masks for the different grids
        # Note: the filenames below are symbolics links to the relevant CFSv2 land cover files
        mask_25x38 = ('LANDMASK_25x38.nc', 'LAND_L1')
        mask_32x50 = ('LANDMASK_32x50.nc', 'LAND_L1')
        mask_121x189 = ('LANDMASK_121x189.nc', 'LAND_L1')
        mask_192x301 = ('LANDMASK_192x301.nc', 'LAND_L1')
        
        cfsr_masks = OrderedDict([('mask_25x38', mask_25x38),
                                  ('mask_32x50', mask_32x50),
                                  ('mask_121x189', mask_121x189),
                                  ('mask_192x301', mask_192x301)])
    
    
    elif cfsr_version in 'ds094.0_6hourly':
        SSS_file = ('ds094.0_salt_emp.nc',  'SALTY_L160_Avg_11')
        swflux_file = ('ds094.0_salt_emp.nc',  'EMNP_L1_Avg_11')
        prate_file = ('ds094.0_precip_rate.nc',  'PRATE_L1_Avg_1')
        shflux_SW_down_file = ('ds094.0_heat_fluxes.nc',  'DSWRF_L1_Avg_1')
        shflux_SW_up_file = ('ds094.0_heat_fluxes.nc',  'USWRF_L1_Avg_1')
        shflux_LW_down_file = ('ds094.0_heat_fluxes.nc',  'DLWRF_L1_Avg_1')
        shflux_LW_up_file = ('ds094.0_heat_fluxes.nc',  'ULWRF_L1_Avg_1')
        shflux_LATENT_HEAT_file = ('ds094.0_heat_fluxes.nc',  'LHTFL_L1_Avg_1')
        shflux_SENSIBLE_HEAT_file = ('ds094.0_heat_fluxes.nc',  'SHTFL_L1_Avg_1')
        sustr_file = ('ds094.0_wind_tmp.nc',  'U_GRD_L103')
        svstr_file = ('ds094.0_wind_tmp.nc',  'V_GRD_L103')
        SST_file = ('ds094.0_wind_tmp.nc',  'TMP_L1')
        sat_file = ('ds094.0_wind_tmp.nc',  'TMP_L103')
        sap_file = ('ds094.0_pres_spec_hum.nc',  'PRES_L1')
        qair_file = ('ds094.0_pres_spec_hum.nc',  'SPF_H_L103')
        rel_hum_file = ('ds094.0_rel_hum.nc',   'R_H_L103')
    
        # Filenames of masks for the different grids
        # Note: the filenames below are symbolics links to the relevant CFSv2 land cover files
        mask_25x38 = ('LANDMASK_25x38.nc', 'LAND_L1')
        mask_32x50 = ('LANDMASK_32x50.nc', 'LAND_L1')
        mask_121x189 = ('LANDMASK_121x189.nc', 'LAND_L1')
        mask_294x460 = ('LANDMASK_294x460.nc', 'LAND_L1')
        cfsr_masks = OrderedDict([('mask_25x38', mask_25x38),
                                  ('mask_32x50', mask_32x50),
                                  ('mask_121x189', mask_121x189),
                                  ('mask_294x460', mask_294x460)])

    
    
    # ROMS configuration information_________________________________
    
    #roms_dir = '/marula/emason/runs2012/MedSea15/'
    #roms_dir = '/shared/emason/marula/emason/runs2012/MedSea5/'
    #roms_dir = '/marula/emason/runs2009/na_2009_7pt5km/'
    #roms_dir = '/home/emason/mercurial_projects/py-easygrid/easygrid-python/easygrid-python/'
    #roms_dir = '/home/emason/runs2012_tmp/MedSea5_R2.5/'
    #roms_dir = '/marula/emason/runs2012/MedSea5_intann_monthly/'
    #roms_dir = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
    #roms_dir = '/Users/emason/toto/'
    #roms_dir = '/marula/emason/runs2013/cb_3km_2013_intann/'
    #roms_dir  = '/marula/emason/runs2013/AlbSea_1pt25/'
    #roms_dir    = '/marula/emason/runs2013/cart500/'
    #roms_dir = '/marula/emason/runs2012/MedSea_Romain/'
    #roms_dir     = '/marula/emason/runs2014/MedCan5/'
    roms_dir     = '/marula/emason/runs2014/na75/'
    #roms_dir     = '/marula/emason/runs2014/NWMED2/'
    #roms_dir     = '/shared/emason/toto/'
    #roms_dir = '/marula/emason/runs2014/nwmed5km/'
    
    
    #roms_grd = 'grd_MedSea5_R2.5.nc'
    #roms_grd    = 'grd_MedSea5.nc'
    roms_grd = 'roms_grd_NA2014_7pt5km.nc'
    #roms_grd = 'grd_nwmed5km.nc'
    #roms_grd = 'grd_nwmed_2km.nc'
    #roms_grd = 'roms_grd_wmed_longterm_newmask.nc'
    #roms_grd = 'roms_grd.nc'
    #roms_grd = 'cb_2009_3km_grd_smooth.nc'
    #roms_grd    = 'grd_AlbSea_1pt25.nc'
    #roms_grd    = 'grd_cart500.nc'
    #roms_grd    = 'roms_grd_wmed_longterm.nc'
    #roms_grd     = 'grd_MedCan5.nc'
    
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
    #frc_filename = 'test_AlbSea_1pt25.nc'
    #frc_filename = 'blk_wmed_CFSR_Y1992M05.nc'
    #frc_filename = 'blk_MedCan5_1984123001.nc'
    #frc_filename = 'blk_NA2009_1988-1989.nc'
    #frc_filename = 'blk_NA2009_1990-1991.nc'
    #frc_filename = 'blk_NA2009_1992-1993.nc'
    #frc_filename = 'blk_NA2009_1994-1995.nc'
    #frc_filename = 'blk_NA2009_1996-1997.nc'
    #frc_filename = 'blk_NA2009_2008-2010.nc'
    #frc_filename = 'blk_NA2009_2011-2012.nc'
    #frc_filename = 'blk_nwmed_2km_2006-2006.nc.JUNK'
    frc_filename = 'blk_NA2014_end_2010.nc'
    #frc_filename = 'blk_nwmed5km_1992-1999.nc'

    # Variable XXX_time/blk_time will be zero at this date
    day_zero = '19850101' # string with format YYYYMMDDHH
    #day_zero = '20060101' # string with format YYYYMMDDHH
    
    # Modify filename
    if '_6hr' not in frc_filename:
        frc_filename = frc_filename.replace('.nc', '_6hr.nc')
    
    
    # True if the frc file being prepared is for a downscaled simulation
    downscaled = False
    if downscaled:
        # Point to parent directory, where make_pycfsr2frc_six-hourly expects to find 
        # start_date.mat (created by set_ROMS_interannual_start.py)
        par_dir = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
        #par_dir = '/marula/emason/runs2012/MedSea5_intann_monthly/'

    
    # Start and end dates of the ROMS simulation
    # must be strings, format 'YYYYMMDDHH'
    #start_date = '1985010100'
    #end_date   = '1987102800'
    #start_date = '1992043000'
    #end_date   = '1992060123'
    #start_date = '2005100100'
    #end_date   = '2005123018'
    #start_date = '2007022300'
    #end_date   = '2007060818'
    #start_date = '1989100100'
    #end_date   = '1991043018'
    #end_date   = '1989123118'
    #start_date = '1989120100'
    #end_date   = '1991013100'
    start_date = '2010100100'
    end_date   = '2010123118'
    #start_date = '1991110100'
    #end_date   = '2000060100'
    #end_date   = '1991120100'


    cycle_length = np.float(0)


    # Option for river runoff climatology
    #   Note, a precomputed *coast_distances.mat* must be available
    #   in roms_dir; this is computed using YET_TO_BE_DONE.py
    add_dai_runoff = True # True of False
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
    
    # This dictionary of CFSR files needs to supply some or all of the surface
    # forcing variables:
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
                                   np.int(start_date[8:10]))
    
    dtenddt = dt.datetime.datetime(np.int(end_date[:4]),
                                   np.int(end_date[4:6]),
                                   np.int(end_date[6:8]),
                                   np.int(end_date[8:10]))
    
    # Number of records at six hourly frequency
    delta = dt.datetime.timedelta(hours=6)
    numrec = dt.drange(dtstrdt, dtenddt, delta).size + 1

    dtstr, dtend = dt.date2num(dtstrdt), dt.date2num(dtenddt)
        
    day_zero = dt.datetime.datetime(int(day_zero[:4]), int(day_zero[4:6]), int(day_zero[6:]))
    day_zero = dt.date2num(day_zero)
    
    time_array = np.arange(dt.date2num(dtstrdt),
                           dt.date2num(dtenddt) + 0.25, 0.25)
    
    
    #if downscaled:
        #inidate = io.loadmat(par_dir + 'start_date.mat')
        #deltaday0 = dtstr - inidate['start_date']
    
    
    # Instantiate a RomsGrid object
    numrec = None
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)), sigma_params, 'ROMS')
    romsgrd.create_frc_nc(''.join((roms_dir, frc_filename)), start_date, end_date, numrec,
                          cycle_length, 'make_pycfsr2frc_six-hourly', bulk)
    romsgrd.make_gnom_transform().proj2gnom(ignore_land_points=True).make_kdetree()
    

    
    # Get all CFSR mask and grid sizes
    if cfsr_version in 'ds093.0_6hourly':
        mask_1 = CfsrMask(cfsr_dir, cfsr_masks['mask_25x38'], romsgrd, 800000)
        mask_2 = CfsrMask(cfsr_dir, cfsr_masks['mask_32x50'], romsgrd, 650000)
        mask_3 = CfsrMask(cfsr_dir, cfsr_masks['mask_121x189'], romsgrd, 250000)
        mask_4 = CfsrMask(cfsr_dir, cfsr_masks['mask_192x301'], romsgrd, 150000)
        
    elif cfsr_version in 'ds094.0_6hourly':
        mask_1 = CfsrMask(cfsr_dir, cfsr_masks['mask_25x38'], romsgrd, 800000)
        mask_2 = CfsrMask(cfsr_dir, cfsr_masks['mask_32x50'], romsgrd, 650000)
        mask_3 = CfsrMask(cfsr_dir, cfsr_masks['mask_121x189'], romsgrd, 250000)
        mask_4 = CfsrMask(cfsr_dir, cfsr_masks['mask_294x460'], romsgrd, 150000)

    
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
            cfsr_prate = CfsrPrate(cfsr_dir, cfsr_files['prate'], masks, romsgrd)
            
            cfsr_prate.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_prate.child_contained_by_parent(romsgrd)
            cfsr_prate.make_kdetree()
            cfsr_prate.get_delaunay_tri()
            
            
        elif cfsr_key in 'radlw': # Outgoing longwave radiation (used for bulk)
            cfsr_sst = CfsrSST(cfsr_dir, cfsr_files['wspd']['SST'], masks, romsgrd)
            cfsr_radlw = CfsrRadlw(cfsr_dir, cfsr_files['radlw'], masks, romsgrd)
            
            supp_vars = [cfsr_sst]
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
            cfsr_radsw = CfsrRadsw(cfsr_dir, cfsr_files['radsw'], masks, romsgrd)
            
            cfsr_radsw.proj2gnom(ignore_land_points=False, M=romsgrd.M)
            cfsr_radsw.child_contained_by_parent(romsgrd)
            cfsr_radsw.make_kdetree()
            cfsr_radsw.get_delaunay_tri()
        
        
        elif cfsr_key in 'wspd': # used for bulk
            cfsr_rhum = CfsrRhum(cfsr_dir, cfsr_files['wspd']['rel_hum'], masks, romsgrd)
            cfsr_qair = CfsrQair(cfsr_dir, cfsr_files['wspd']['qair'], masks, romsgrd)
            cfsr_sat = CfsrSat(cfsr_dir, cfsr_files['wspd']['sat'], masks, romsgrd)
            cfsr_sap = CfsrSap(cfsr_dir, cfsr_files['wspd']['sap'], masks, romsgrd)
            cfsr_wspd = CfsrWspd(cfsr_dir, cfsr_files['wspd'], masks, romsgrd)
            
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
            print 'Unknown key %s' % cfsr_key; raise Exception
  
  
             
        tind = 0
        
        # Loop over time
        if 1:
            
            for dt in time_array:
    
                dtnum = dt - day_zero
                
                # Precipitation rate (bulk only)
                if 'prate' in cfsr_key:
                    
                    cfsr_prate.set_date_index(dt)
                    cfsr_prate.get_cfsr_data().convert_cmday()
                    cfsr_prate.interp2romsgrd()
                    #cfsr_prate.check_interp()
                    
                    # Get indices and weights for Dai river climatology
                    if add_dai_runoff:
                        ind_min, ind_max, weights = romsgrd.get_runoff_index_weights(dt)
                        # If condition so we don't read runoff data every iteration
                        if np.logical_or(ind_min != dai_ind_min, ind_max != dai_ind_max):
                            dai_ind_min, dai_ind_max = ind_min, ind_max
                            runoff1 = romsgrd.get_runoff(dai_file, dai_ind_min+1)
                            runoff2 = romsgrd.get_runoff(dai_file, dai_ind_max+1)
                        runoff = np.average([runoff1, runoff2], weights=weights, axis=0)
                        cfsr_prate.dataout += runoff.ravel()
                    
                    cfsr_prate.dataout *= romsgrd.maskr().ravel()
                    np.place(cfsr_prate.dataout, cfsr_prate.dataout < 0., 0.)
                    
                    with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                        nc.variables['prate'][tind] = cfsr_prate.dataout
                        nc.sync()

                
                # Outgoing longwave radiation (bulk only)
                elif 'radlw' in cfsr_key:
                    
                    cfsr_sst.set_date_index(dt)
                    cfsr_sst.get_cfsr_data().fillmask()
                    cfsr_radlw.set_date_index(dt)
                    cfsr_radlw.get_cfsr_data(cfsr_sst.datain)
                    cfsr_radlw.interp2romsgrd(fillmask_radlw, fillmask_radlw_in)
                    with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                        nc.variables['radlw'][tind] = cfsr_radlw.radlw_dataout
                        nc.variables['radlw_in'][tind] = cfsr_radlw.radlw_in_dataout
                        nc.sync()


                # Net shortwave radiation (bulk only)
                elif 'radsw' in cfsr_key:
                    
                    cfsr_radsw.set_date_index(dt)
                    cfsr_radsw.get_cfsr_data().fillmask()
                    cfsr_radsw.interp2romsgrd()
                    #cfsr_radsw.check_interp()   
                    np.place(cfsr_radsw.dataout, cfsr_radsw.dataout < 0., 0)
                
                    with netcdf.Dataset(romsgrd.frcfile, 'a') as nc:
                        cfsr_radsw.dataout *= romsgrd.maskr().ravel()
                        nc.variables['radsw'][tind] = cfsr_radsw.dataout
                        nc.sync()
                
                # Wind speed (wspd, uwnd, vwnd) and stress (sustr, svstr) (bulk only)
                elif 'wspd' in cfsr_key:
                    
                    cfsr_rhum.set_date_index(dt)
                    cfsr_sat.set_date_index(dt)
                    cfsr_sap.set_date_index(dt)
                    cfsr_qair.set_date_index(dt)
                    cfsr_wspd.set_date_index(dt)
                    
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
                    cfsr_wspd.uwnd_dataout[:, :-1] = romsgrd.rho2u_2d(cfsr_wspd.uwnd_dataout)
                    cfsr_wspd.uwnd_dataout[:, :-1] *= romsgrd.umask()
                    cfsr_wspd.vwnd_dataout[:-1] = romsgrd.rho2v_2d(cfsr_wspd.vwnd_dataout)
                    cfsr_wspd.vwnd_dataout[:-1] *= romsgrd.vmask()
                    
                    cfsr_wspd.ustrs_dataout[:], \
                    cfsr_wspd.vstrs_dataout[:] = romsgrd.rotate(cfsr_wspd.ustrs_dataout,
                                                               cfsr_wspd.vstrs_dataout, sign=1)
                    cfsr_wspd.ustrs_dataout[:, :-1] = romsgrd.rho2u_2d(cfsr_wspd.ustrs_dataout)
                    cfsr_wspd.ustrs_dataout[:, :-1] *= romsgrd.umask()
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
                        nc.variables['uwnd'][tind] = cfsr_wspd.uwnd_dataout[:, :-1]
                        nc.variables['vwnd'][tind] = cfsr_wspd.vwnd_dataout[:-1]
                        nc.variables['sustr'][tind] = cfsr_wspd.ustrs_dataout[:, :-1]
                        nc.variables['svstr'][tind] = cfsr_wspd.vstrs_dataout[:-1]
                    
                        nc.variables['bulk_time'][tind] = dtnum
                        if numrec is not None:
                            nc.variables['bulk_time'].cycle_length = np.float(numrec)
                        nc.sync()
                
                tind += 1

                
                
                
                
                
                
                
                
