# -*- coding: utf-8 -*-
# %run pyccmp2frc.py

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

Create a ROMS forcing file (only wind stress) based on CCMP wind data

===========================================================================
'''

import netCDF4 as netcdf
import pylab as plt
import numpy as np
from scipy import io
import scipy.interpolate as si
import scipy.ndimage as nd
import scipy.spatial as sp
import matplotlib.nxutils as nx
import time
import scipy.interpolate.interpnd as interpnd
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
from datetime import datetime
import calendar as ca
import glob

from pyroms2roms import horizInterp
from pyroms2roms import ROMS, debug0, debug1
from pysoda2roms import SodaGrid, SodaData, RomsGrid
from pycfsr2frc import AirSea


class RomsGrid(RomsGrid):
    '''
    Modify the RomsGrid class
    '''
    
    def create_frc_nc(self, frcfile, sd, ed, nr, cl, madeby):
        '''
        Create a new forcing file based on dimensions from grdobj
            frcfile : path and name of new frc file
            sd      : start date2num
            ed      : end date
            nr      : no. of records
            cl      : cycle length
            madeby  : name of this file
        '''
        self.frcfile = frcfile
        
        # Global attributes
        ''' The best choice should be format='NETCDF4', but it will not work with
        Sasha's 2008 code (not tested with Roms-Agrif).  Therefore I use
        format='NETCDF3_64BIT; the drawback is that it is very slow'
        '''
        nc = netcdf.Dataset(frcfile, 'w', format='NETCDF3_64BIT')
        #nc = netcdf.Dataset(frcfile, 'w', format='NETCDF4')
        #nc = netcdf.Dataset(frcfile, 'w', format='NETCDF3_CLASSIC')
        nc.created  = datetime.now().isoformat()
        nc.type = 'ROMS interannual forcing file produced by %s.py' %madeby
        nc.grd_file = self.romsfile
        nc.start_date = sd
        nc.end_date = ed
        
        # Dimensions
        nc.createDimension('xi_rho', self.lon().shape[1])
        nc.createDimension('xi_u', self.lon().shape[1] - 1)
        nc.createDimension('eta_rho', self.lon().shape[0])
        nc.createDimension('eta_v', self.lon().shape[0] - 1)
        nc.createDimension('sms_time', nr)

        

        # Dictionary for the variables
        frc_vars = OrderedDict()
        
        frc_vars['sms_time'] = ['time',
                                'sms_time',
                                'surface momentum stress time',
                                'days',
                                'f8']
        
        frc_vars['sustr'] =  ['u',
                              'sms_time',
                              'surface u-momentum stress',
                              'Newton meter-2',
                              'f4']
        frc_vars['svstr'] =  ['v',
                              'sms_time',
                              'surface v-momentum stress',
                              'Newton meter-2',
                              'f4']
        
        
        
        for key, value in zip(frc_vars.keys(), frc_vars.values()):

            print key, value

            if 'time' in value[0]:
                dims = (value[1])

            elif 'u' in value[0]:
                dims = (value[1], 'eta_rho', 'xi_u')
                    
            elif 'v' in value[0]:
                dims = (value[1], 'eta_v', 'xi_rho')
                
            else:
                error
            
            #print 'key dims',key, dims
            nc.createVariable(key, value[-1], dims)
            nc.variables[key].long_name = value[2]
            nc.variables[key].units = value[3]
            
            if nr is not None and 'time' in key:
                nc.variables[key].cycle_length = cl
        
        nc.close()
    


class CcmpGrid(SodaGrid):
    '''
    
    '''
    def nobs(self, ind=':'):
        self.t = ind
        if self.fix_zero_crossing is True:
            nobs1 = self.read_nc('nobs',
                    indices='[self.t, self.j0:self.j1, :self.i0]')
            nobs0 = self.read_nc('nobs',
                    indices='[self.t, self.j0:self.j1, self.i1:]')
            nobs = np.concatenate((nobs0, nobs1), axis=1)
        else:
            nobs = self.read_nc('nobs',
                    indices='[self.t, self.j0:self.j1, self.i0:self.i1]')
        return nobs






class CcmpData(SodaData):
    '''
    CCMP data class (inherits from SodaData <- RomsData classes)
    '''
    def time(self, ind=':'):
        self.t = ind
        if isinstance(self.t, str):
            # Return full time record
            return self.read_nc('time', indices='[:]')
        else:
            # Return specified time record
            return self.read_nc('time', indices='[self.t]')


    def uwnd(self, ind=':'):
        self.t = ind
        if self.fix_zero_crossing is True:
            uwnd1 = self.read_nc('uwnd',
                    indices='[self.t, self.j0:self.j1, :self.i0]')
            uwnd0 = self.read_nc('uwnd',
                    indices='[self.t, self.j0:self.j1, self.i1:]')
            return np.concatenate((uwnd0, uwnd1), axis=1)
        else:
            return self.read_nc('uwnd',
                   indices='[self.t, self.j0:self.j1, self.i0:self.i1:]')
            
            
    def vwnd(self, ind=':'):
        self.t = ind
        if self.fix_zero_crossing is True:
            vwnd1 = self.read_nc('vwnd',
                    indices='[self.t, self.j0:self.j1, :self.i0]')
            vwnd0 = self.read_nc('vwnd',
                    indices='[self.t, self.j0:self.j1, self.i1:]')
            return np.concatenate((vwnd0, vwnd1), axis=1)
        else:
            return self.read_nc('vwnd',
                   indices='[self.t, self.j0:self.j1, self.i0:self.i1:]')
            

    def get_datetime(self, ind=':'):
        '''
        Time is hours since basedate
        '''
        basedate = plt.date2num(self.get_basedate())
        time = self.time(ind) / 24.
        time += basedate
        return time
        
        

    def get_basedate(self):
        basedate = self.read_nc_att('time', 'units')
        ymd = basedate.split(' ')[2].replace('-','')
        hms = basedate.split(' ')[3].replace(':','')
        return plt.datetime.datetime(np.int(ymd[:4]),
                                     np.int(ymd[4:6]),
                                     np.int(ymd[6:8]),
                                     np.int(hms[:2]))



if __name__ == '__main__':
    
    '''
    pyccmp2frc.py

    Add interannual CCMP winds to prepared ROMS surface forcing file
    
      http://rda.ucar.edu/pub/cfsr.html
    
    CCMP surface wind speed data are global but subgrids can be
    selected. 
    
    
    Evan Mason, IMEDEA, 2012
    '''
    

    #_USER DEFINED VARIABLES_______________________________________
    
    # CCMP information_________________________________
    #ccmp_dir = '/Users/emason/toto/'
    ccmp_dir = '/shared/emason/winds/ccmp_3.0/'
    
    ccmp_filename = 'analysis_????????_v11l30flk.nc'
    
    
    # ROMS configuration information_________________________________
    
    #roms_dir = '/marula/emason/runs2012/MedSea15/'
    #roms_dir = '/Users/emason/toto/'
    #roms_dir = '/nas02/emason/runs2012/MedSea15/MedSea15_IA/'
    roms_dir = '/marula/emason/runs2013/cb_3km_2013_intann/'

    add_to_existing = True
    
    if add_to_existing:
        existing_frc = 'frc_2013_cb3km_CFSR.nc'


    #roms_grd = 'grd_MedSea15.nc'
    roms_grd = 'cb_2009_3km_grd_smooth.nc'
    
    # Forcing file
    frc_filename = 'frc_2013_cb3km.nc' # frc filename
    
    
    frequency = 'daily' # must be one of 'monthly', 'weekly','daily', 'six-hourly'
    
    # True if the frc file being prepared is for a downscaled simulation
    downscaled = True
    if downscaled:
        # Point to parent directory, where pyccmp2frc expects to find 
        # start_date.mat (created by set_ROMS_interannual_start.py)
        par_dir = '/marula/emason/runs2013/na_7pt5km_intann_5day/'
        

    # start and end dates of the ROMS simulation
    # must be strings, format 'YYYYMMDDHH'
    start_date = '1997123000'
    end_date   = '2000013100'

    
    cycle_length = 0



    #_END USER DEFINED VARIABLES_______________________________________
    
    plt.close('all')
    
    
    
    airsea = AirSea()
    
    dtstrdt = plt.datetime.datetime(np.int(start_date[:4]),
                                    np.int(start_date[4:6]),
                                    np.int(start_date[6:8]),
                                    np.int(start_date[8:10]))
    
    dtenddt = plt.datetime.datetime(np.int(end_date[:4]),
                                    np.int(end_date[4:6]),
                                    np.int(end_date[6:8]),
                                    np.int(end_date[8:10]))
    
    
    
    dtstr, dtend = plt.date2num(dtstrdt), plt.date2num(dtenddt)
    
    if downscaled:
        inidate = io.loadmat(par_dir + 'start_date.mat')
        deltaday0 = dtstr - inidate['start_date']
    else:
        deltaday0 = 0
    
    if 'monthly' in frequency:
        numrec = (dtenddt.year - dtstrdt.year) * 12 + (dtenddt.month - dtstrdt.month)
        
    elif 'daily' in frequency:
        numrec = np.int(dtend - dtstr)
        
    elif 'six-hourly' in frequency:
        numrec = 1 + np.int(dtend - dtstr) * 4
        
    else:
        raise Exception, 'frequency not correctly defined'
    
    
    #numrec = None ##################################################
    
    
    # Initialise a RomsGrid object
    romsgrd = RomsGrid(''.join((roms_dir, roms_grd)))
    romsgrd.roms_dir = roms_dir
    
    # Append to or create a forcing file
    if add_to_existing:
        frc_filename = roms_dir + existing_frc
        nc = netcdf.Dataset(frc_filename, 'a')
        nc.renameVariable('sustr', 'Xsustr')
        nc.renameVariable('svstr', 'Xsvstr')
        nc.renameVariable('sms_time', 'Xsms_time')
        nc.renameDimension('sms_time', 'Xsms_time')
        nc.createDimension('sms_time', numrec)
        # Dictionary for the variables
        frc_vars = OrderedDict()
        frc_vars['sms_time'] = ['time',
                                'sms_time',
                                'surface momentum stress time',
                                'days']
        frc_vars['sustr'] =  ['u',
                              'sms_time',
                              'surface u-momentum stress',
                              'Newton meter-2']
        frc_vars['svstr'] =  ['v',
                              'sms_time',
                              'surface v-momentum stress',
                              'Newton meter-2']
        for key, value in zip(frc_vars.keys(), frc_vars.values()):

            #print key, value

            if 'time' in value[0]:
                dims = (value[1])

            elif 'u' in value[0]:
                dims = (value[1], 'eta_rho', 'xi_u')
                    
            elif 'v' in value[0]:
                dims = (value[1], 'eta_v', 'xi_rho')
                
            else:
                error
            
            #print 'key dims',key, dims
            nc.createVariable(key, 'f8', dims)
            nc.variables[key].long_name = value[2]
            nc.variables[key].units = value[3]
            
            if 'time' in key:
                nc.variables[key].cycle_length = cycle_length
        
        nc.close()
        
    else:
        frc_filename = roms_dir + frc_filename.replace('.', '_CCMP_%s.' %frequency)
        romsgrd.create_frc_nc(frc_filename, start_date, end_date, numrec,
                              cycle_length, 'pyccmp2frc')
                          
    # Gnomonic projections for horizontal interpolations
    roms_M = romsgrd.get_gnom_trans()
    roms_points = romsgrd.proj2gnom(roms_M, no_mask=True) # we only want data points

    # create ROMS grid KDE tree
    roms_tree = sp.KDTree(roms_points)
    

    ccmp_files = sorted(glob.glob(ccmp_dir + ccmp_filename))

    ccmpgrd = CcmpGrid(ccmp_files[0])
    
    # activate flag for zero crossing trickery
    romsgrd.check_zero_crossing()
    if romsgrd.zero_crossing is True:
        print 'The ROMS domain straddles the zero-degree meridian'
        ccmpgrd.zero_crossing = True


    # set ccmpgrd indices (i0:i1, j0:j1) for minimal subgrid around romsgrd
    ccmpgrd.set_subgrid(romsgrd, k=40)

    
    if True: # check the result of set_subgrid()
        debug0(ccmpgrd.lon(), ccmpgrd.lat(), ccmpgrd.nobs(3),
               romsgrd.boundary()[0], romsgrd.boundary()[1])
    
    
    
    # Gnomonic projections for horizontal interpolations
    roms_M = romsgrd.get_gnom_trans()
    roms_points = romsgrd.proj2gnom(roms_M, no_mask=False) # only want data points
    ccmp_points = ccmpgrd.proj2gnom(roms_M) # must use roms_M
    
    '''
    Check that no ROMS data points lie outside of the
    CCMP domain. If this occurs pyccmp2frc cannot function;
    the only solution is to make a new ROMS grid ensuring the above
    condition is met.
    '''
    ccmp_tri = sp.Delaunay(ccmp_points) # triangulate full parent 
    tn = ccmp_tri.find_simplex(roms_points)
    assert not np.any(tn==-1), 'ROMS data points outside CCMP domain detected'
   
    
    
    # 'active' flag is True when looping through a desired time range
    flags = dict(active = False)
    
    
    tind = 0
    theday = dtstrdt.day
    themonth = dtstrdt.month
    count = 0.
    
    uspd = np.zeros_like(romsgrd.lon())
    vspd = np.zeros_like(romsgrd.lon())
    
    # Only used for 6-hourly case
    ccmp_filei = 0
    
    # loop over the CCMP files
    for ccmp_file in ccmp_files:
        
        #print ccmp_file
        ccmp_date = ccmp_file.rpartition('/')[-1]
        ccmp_date = ccmp_date[ccmp_date.find('_')+1:ccmp_date.rfind('_')]
        ccmpt = plt.datetime.datetime(np.int(ccmp_date[:4]),
                                      np.int(ccmp_date[4:6]),
                                      np.int(ccmp_date[6:8]))
        ccmpdt = plt.date2num(ccmpt)
        
        if np.logical_and(ccmpdt >= dtstr, ccmpdt <= dtend):
            flags['active'] = True
        else:
            flags['active'] = False
        

        if flags['active']:
            
            ccmp = CcmpData(ccmp_file)
                
            if ccmpgrd.zero_crossing is True:
                ccmp.fix_zero_crossing = True
                    
            # tell CcmpData object about i/j limits
            ccmp.i0, ccmp.i1 = ccmpgrd.i0, ccmpgrd.i1
            ccmp.j0, ccmp.j1 = ccmpgrd.j0, ccmpgrd.j1
            
            
            # loop over the time records in the file
            for ccmpi, ccmpt in enumerate(ccmp.get_datetime()):
                
                # u wind speed
                uspd_tmp = ccmp.uwnd(ccmpi)
                uspd_tmp = horizInterp(ccmp_tri, uspd_tmp.ravel())(roms_points)
                uspd_tmp = uspd_tmp.reshape(romsgrd.lon().shape)
                
                #  vwind speed
                vspd_tmp = ccmp.vwnd(ccmpi)
                vspd_tmp = horizInterp(ccmp_tri, vspd_tmp.ravel())(roms_points)
                vspd_tmp = vspd_tmp.reshape(romsgrd.lon().shape)
                    
                ccmpdate = plt.num2date(ccmpt)

                if 'monthly' in frequency:
                
                    if themonth == ccmpdate.month:
                    
                        uspd += uspd_tmp
                        vspd += vspd_tmp
                        count += 1.
                        print 'count', count
                        
                    else:
                    
                        uspd /= count
                        vspd /= count
                        
                        # speed to stress (Smith etal 1988)
                        sustr, svstr = airsea.stresstc(uspd, vspd)
                        sustr, svstr = romsgrd.rotate(sustr, svstr, 1)
                        
                        sustr = romsgrd.rho2u_2d(sustr) * romsgrd.umask()
                        svstr = romsgrd.rho2v_2d(svstr) * romsgrd.vmask()
                        
                        nc = netcdf.Dataset(frc_filename, 'a')
                        nc.variables['sustr'][tind] = sustr
                        nc.variables['svstr'][tind] = svstr
                        nc.variables['sms_time'][tind] = deltaday0
                        if numrec is not None:
                            nc.variables['sms_time'].cycle_length = tind + 1
                        nc.close()
                        print 'STILL NEED TO PUT IN DELTADAY0'
                        uspd = uspd_tmp.copy()
                        vspd = vspd_tmp.copy()
                        count = 1.
                        
                        tind += 1
                        
                        print 'themonth', themonth
                        themonth = ccmpdate.month
                        
                elif 'daily' in frequency:
                
                    if theday == ccmpdate.day:
                    
                        uspd += uspd_tmp
                        vspd += vspd_tmp
                        count += 1.
                        #print 'count', count
                        
                    else:
                    
                        uspd /= count
                        vspd /= count
                        
                        # speed to stress (Smith etal 1988)
                        sustr, svstr = airsea.stresstc(uspd, vspd)
                        sustr, svstr = romsgrd.rotate(sustr, svstr, 1)
                        
                        sustr = romsgrd.rho2u_2d(sustr) * romsgrd.umask()
                        svstr = romsgrd.rho2v_2d(svstr) * romsgrd.vmask()
                        
                        nc = netcdf.Dataset(frc_filename, 'a')
                        nc.variables['sustr'][tind] = sustr
                        nc.variables['svstr'][tind] = svstr
                        # Plus 0.5 sets us to midday.
                        nc.variables['sms_time'][tind] = tind + 0.5 + deltaday0
                        if numrec is not None:
                            nc.variables['sms_time'].cycle_length = tind + 1
                        nc.sync()
                        nc.close()
                        
                        uspd = uspd_tmp.copy()
                        vspd = vspd_tmp.copy()
                        count = 1.
                        
                        print 'Day %s done' %plt.num2date(dtstr + tind + 0.5)
                        
                        tind += 1
                        
                        theday = ccmpdate.day
                        

                elif 'six-hourly' in frequency:
                
                    # speed to stress (Smith etal 1988)
                    sustr, svstr = airsea.stresstc(uspd_tmp, vspd_tmp)
                    sustr, svstr = romsgrd.rotate(sustr, svstr, 1)
                        
                    sustr = romsgrd.rho2u_2d(sustr) * romsgrd.umask()
                    svstr = romsgrd.rho2v_2d(svstr) * romsgrd.vmask()
                        
                    nc = netcdf.Dataset(frc_filename, 'a')
                    nc.variables['sustr'][ccmp_filei + ccmpi] = sustr
                    nc.variables['svstr'][ccmp_filei + ccmpi] = svstr
                    
                    nc.variables['sms_time'][ccmp_filei + ccmpi] = ccmpt - dtstr + deltaday0
                    if numrec is not None:
                        nc.variables['sms_time'].cycle_length = np.float(ccmp_filei + ccmpi + 1)
                    
                    nc.sync()
                    nc.close()
                    
                    #print 'ccmp_filei, ccmpi, ccmp_filei+ccmpi',ccmp_filei, ccmpi, ccmp_filei+ccmpi
                    
                    print 'Day %s done' %plt.num2date(ccmpt)
                        
            # Only used for 6-hourly case
            ccmp_filei += 4
                        


                    
                


                
          