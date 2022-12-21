# James Gibson
# 11/28/22

#Preprocessing AOD data

import arcpy, sys, os
import os.path
from collections import defaultdict
from arcpy import env
from arcpy.sa import *
import time
import datetime
import pandas as pd
import numpy as np
import multiprocessing
arcpy.env.overwriteOutput = True

bounds = r'G:\HumanPlanet\GADM\GADM.gdb\GADM_NorthEast'

year = '2021'



out_gdb = r'G:\HumanPlanet\AOD\%s\aod_%s_avg.gdb' % (year,year)
if arcpy.Exists(out_gdb):
    pass
else:
    arcpy.CreateFileGDB_management(r'G:\HumanPlanet\AOD\%s' % year, 'aod_%s_avg.gdb' % year) 

#average rasters
arcpy.env.workspace = r'G:\HumanPlanet\AOD\%s\aod_%s_fill.gdb' % (year, year)

my_rasters = []
rasters = arcpy.ListRasters("*")
for raster in rasters:
    my_rasters.append(Raster(raster))

print(my_rasters)
outCellStats = CellStatistics(my_rasters, "MEAN", "DATA")
outCellStats.save(r'G:\HumanPlanet\AOD\aod_avg\aod_%s_avg.tif' % (year))
print('DONE')


