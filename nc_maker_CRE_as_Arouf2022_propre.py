#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 11:54:57 2026

@author: aarouf
"""



import numpy as np
import xarray as xr
import xesmf as xe
import glob
    
from time import time, strftime, gmtime
import glob
from os import system
from os import path
import sys
import os
from datetime import datetime, timedelta, date
import netCDF4
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from Assia_Utils import plot_functions as Apf
from Assia_Utils import opendata_functions as Aof
from Assia_Utils import constantes_files as Acf
from Assia_Utils import calcul_functions as Acalf

#%

Working_Path = os.path.dirname(os.path.abspath(__file__))+'/'

#%% data getting 

Map_Opaque = '/Users/aarouf/Postdoc/code/data/Obs/CALIPSO/Map_OPAQ330m_200606-202012_avg_CFMIP2_sat_3.1.2.nc'
ds_Map_Opaque = xr.open_mfdataset(Map_Opaque)
ds_Map_Opaque = ds_Map_Opaque.rename({'latitude':'lat','longitude':'lon' })

[lat, lon, time_G,  \
C_Op, Z_T_Op, Z_FA, \
C_Th, Z_T_Th, emis] = [ds_Map_Opaque.lat, ds_Map_Opaque.lon, ds_Map_Opaque.time, \
                       ds_Map_Opaque.cltcalipso_opaque, ds_Map_Opaque.cltcalipso_opaque_z, ds_Map_Opaque.zopaque,\
                       ds_Map_Opaque.cltcalipso_thin, ds_Map_Opaque.cltcalipso_thin_z,   ds_Map_Opaque.cltcalipso_thin_emis  ]

ds_coef_a_b = xr.open_mfdataset('/Users/aarouf/Postdoc/code/data/Kernels/Arouf_LWCRE-LIDAR-Ed1_SFC__coef_a_b.nc')

a_Global = ds_coef_a_b.a_Global_map
b_Global = ds_coef_a_b.b_Global_map

''' adding the dimension time to the coes a and b '''
month = ds_Map_Opaque['time'].dt.month                      # (time,)
a_Global_map = a_Global.sel(month=month).drop('month')       # (time, lat, lon)
b_Global_map = b_Global.sel(month=month).drop('month')       # (time, lat, lon)


b_Global_map.sel(time='2008').mean('time').plot(vmax=100)

#%% Faire le calcul du CRE map a map :)

CRE_GOCCP_SFC_Op = C_Op * ((a_Global_map * Z_T_Op) + b_Global_map)
CRE_GOCCP_SFC_Th = C_Th * ( (emis+0.06) * ((a_Global_map * Z_T_Th) + b_Global_map) )

# CRE_GOCCP_SFC_test = CRE_GOCCP_SFC_Op + CRE_GOCCP_SFC_Th
'pour eviter les 0 avec la somme des nans'
CRE_GOCCP_SFC = xr.concat(
    [CRE_GOCCP_SFC_Op, CRE_GOCCP_SFC_Th],
    dim='component').sum('component', skipna=True, min_count=1)


''' getting the one with Zopaque'''
CRE_GOCCP_SFC_Op_Z_FA = C_Op * ((a_Global_map * Z_FA) + b_Global_map)

# CRE_GOCCP_SFC_Z_FA_test = CRE_GOCCP_SFC_Op_Z_FA + CRE_GOCCP_SFC_Th
CRE_GOCCP_SFC_Z_FA = xr.concat(
    [CRE_GOCCP_SFC_Op_Z_FA, CRE_GOCCP_SFC_Th],
    dim='component').sum('component', skipna=True, min_count=1)

CRE_GOCCP_SFC.mean('time').plot()

#%%

weights = np.cos(np.deg2rad(ds_Map_Opaque.lat))
weights.name = "weights"

#%%  FIGURE 8 : Décomposition Opaque/Thin du CRE GOCCP   hspace = 0.2,wspace = -0.42
if 1 :
    fig =plt.figure(6, figsize = (12,9))
    fig.subplots_adjust(left = 0.03, bottom = 0.1, right = 0.98, top = 0.96, hspace = 0.4, wspace =-0.46)#, wspace = 0, hspace = 0.5)
    
    
    kwargs = dict(projection = Acf.projection[1],  cmap = Acf.cmap[6],  n = 26, n2=5, extend = 'max', \
                   cb = 0, cb_label=r'CRE (W $m^{-2}$)', orientation = Acf.orientation[0],\
                   shrink= 0.8, pad=0.03, colormesh_plt=0, boundinglat=70, rounds=1, fillcontin=0, fillocean=0)
        
    cax1 = plt.axes([0.33, 0.58, 0.35, 0.01])
    plt.subplot(232)
    # plt.text(0.5,0.5 , '+ ', style = 'italic',
    #         fontweight = 'bold', fontsize = 20, family = 'serif')

    Apf.plot_map(
        CRE_GOCCP_SFC.sel(time=slice('2008','2020')), CRE_GOCCP_SFC.lat, CRE_GOCCP_SFC.lon,
        title=f'CRE All {CRE_GOCCP_SFC.weighted(weights).mean(dim=["time","lat", "lon"]).values:.1f}',vmin = 0, vmax = 65,
        cax=cax1, **kwargs)
    
    
    cax2 = plt.axes([0.07, 0.07, 0.35, 0.01])
    plt.subplot(234)
    # ax = fig.add_subplot(spec[0])
    Apf.plot_map(
        CRE_GOCCP_SFC_Op.sel(time=slice('2008','2020')), CRE_GOCCP_SFC.lat, CRE_GOCCP_SFC.lon,
        title=f'CRE Opaque {CRE_GOCCP_SFC_Op.weighted(weights).mean(dim=["time","lat", "lon"]).values:.1f}',vmin = 0, vmax = 65,
        cax=cax2, **kwargs)
    
    cax3 = plt.axes([0.58, 0.07, 0.35, 0.01])
    plt.subplot(236)
    # ax = fig.add_subplot(spec[0])
    Apf.plot_map(
        CRE_GOCCP_SFC_Th.sel(time=slice('2008','2020')), CRE_GOCCP_SFC.lat, CRE_GOCCP_SFC.lon,
        title=f'CRE Thin {CRE_GOCCP_SFC_Th.weighted(weights).mean(dim=["time","lat", "lon"]).values:.1f}',vmin = 2, vmax = 14,
        cax=cax3, **kwargs)
    
    
    # plt.savefig(WORKingPATH +'fig_8_CRE_GOCCP_2008-2020',dpi = 180)
    plt.show()
    plt.close()
    
    
#%%

