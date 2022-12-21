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



out_gdb = r'G:\HumanPlanet\AOD\%s\aod_%s_fill.gdb' % (year,year)
if arcpy.Exists(out_gdb):
    pass
else:
    arcpy.CreateFileGDB_management(r'G:\HumanPlanet\AOD\%s' % year, 'aod_%s_fill.gdb' % year) 

#fill gaps
arcpy.env.workspace = r'G:\HumanPlanet\AOD\%s\aod_%s_edited.gdb' % (year, year)

my_rasters = []
rasters = arcpy.ListRasters("*")
for raster in rasters:
    in_raster = Raster(raster)
    filled = arcpy.sa.Con(arcpy.sa.IsNull(in_raster),arcpy.sa.FocalStatistics(in_raster,
                        arcpy.sa.NbrRectangle(5, 5, "CELL"),'MEAN'), in_raster)
    out_raster = r'G:\HumanPlanet\AOD\%s\aod_%s_fill.gdb\%s_fill' % (year,year,raster)
    filled.save(out_raster)
    print('done: %s' % out_raster)
