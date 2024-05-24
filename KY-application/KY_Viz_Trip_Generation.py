import pandas as pd
import numpy as np
import folium
import geopandas as gp

##Currently for chicago, will have to change it to kentucky, October 2019.

def ky_viz_trip_generation(est_file, maps_output, tracts_shapefile,tods):
    ##make choropleth maps
    geo = gp.read_file(tracts_shapefile)
    geo['geoid10'] = geo.geoid10.astype(float)

    for tod in tods:
        if tod == 1:
            df = est_file[est_file['TOD'] == 1]
        elif tod ==2:
            df = est_file[est_file['TOD'] == 2]
        elif tod ==3:
            df = est_file[est_file['TOD'] == 3]
        elif tod ==4:
            df = est_file[est_file['TOD'] == 4]
        elif tod ==5:
            df = est_file[est_file['TOD'] == 5]
        else:
            print('Bad TOD')