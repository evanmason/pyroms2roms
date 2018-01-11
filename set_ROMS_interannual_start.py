# -*- coding: utf-8 -*-
# %run set_ROMS_interannual_start.py

'''
Place mat files containing datenum corresponding to beginning
of an interannual solution
Required to be in parent directory if 'downscaled' defined in, e.g.,:
  pycfsr2roms
  pyccmp2roms


'''


import scipy.io as io
import matplotlib.dates as dt
import numpy as np

# Make a list of directories
directories = ['/marula/emason/runs2012/MedSea5_intann_monthly/',
               '/marula/emason/runs2013/na_7pt5km_intann_5day/',
	           '/marula/emason/runs2014/NWMED2_unstable/']

# Make a list of corresponding start dates
starts = [np.array([1985, 1, 1, 0]),
          np.array([1985, 1, 1, 0]),
	  np.array([2005, 12, 30, 0])]

###--------------------------------------------------------------


for start, directory in zip(starts, directories):
    
    date = dt.datetime.datetime(start[0], start[1], start[2], start[3])
    date = dt.date2num(date)
    
    datedic = dict(start_date=date)
    
    io.savemat(directory + 'start_date.mat', datedic)
    
print 'All done'
