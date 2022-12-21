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

bounds = r'G:\HumanPlanet\GADM\GADM.gdb\GADM_US_NorthEast'

year = '2021'

#Editing rasters
arcpy.env.workspace = r'G:\HumanPlanet\AOD\%s\aod_%s_clip.gdb' % (year, year)
rasters = arcpy.ListRasters("*")
for raster in rasters:
    out_gdb = r'G:\HumanPlanet\AOD\%s\aod_%s_edited.gdb' % (year,year)
    if arcpy.Exists(out_gdb):
        pass
    else:
        arcpy.CreateFileGDB_management(r'G:\HumanPlanet\AOD\%s' % year, 'aod_%s_edited.gdb' % year) 
    
    in_raster = Raster(raster)
    output_raster = SetNull(in_raster, in_raster, "VALUE > 1")
    output_raster.save(r'G:\HumanPlanet\AOD\%s\aod_%s_edited.gdb\%s_edit' % (year, year, raster))
    print('done: %s' % output_raster)
