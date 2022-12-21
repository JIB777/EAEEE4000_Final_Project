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
root_folder = r'G:\HumanPlanet\AOD\%s\float' % year

images = os.listdir(root_folder)
#Clipping rasters
for img in images:
    try:
        date = img[16:26]
        full_path_to_raster = root_folder + '/' + img
        
        #clip   
        arcpy.env.snapRaster = full_path_to_raster
        arcpy.env.cellSize = full_path_to_raster
        out_gdb = r'G:\HumanPlanet\AOD\%s\aod_%s_clip.gdb' % (year,year)
        if arcpy.Exists(out_gdb):
            pass
        else:
            arcpy.CreateFileGDB_management(r'G:\HumanPlanet\AOD\%s' % year, 'aod_%s_clip.gdb' % year) 
        out_raster = r'G:\HumanPlanet\AOD\%s\aod_%s_clip.gdb\aod_%s_ne' % (year,year,date)
        arcpy.Clip_management(full_path_to_raster,"#",out_raster,bounds,"#","#","#")
        print('done: %s' % out_raster)
    except:
        print('Error: %s' % full_path_to_raster)
    

