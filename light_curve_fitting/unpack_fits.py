import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd 
import jax 
import jax.numpy as jnp 

def unpack_niriss_exotedrf(infile, order, trim_start, trim_end, wl_min_o1=None, wl_max_o1=None, wl_min_o2=None, wl_max_o2=None):    

    bjd = fits.getdata(infile, 9)
    wave = fits.getdata(infile, 1 + 4 * (order - 1))
    wave_err = fits.getdata(infile, 2 + 4 * (order - 1))
    fluxcube = fits.getdata(infile, 3 + 4 * (order - 1))
    fluxcube_err = fits.getdata(infile, 4 + 4 * (order -1))
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]
    
    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    if order == 2:
        ii = np.where((wave >= 0.6) & (wave <= 0.85))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    if wl_min_o1 is not None and wl_max_o1 is not None and order == 1:
        ii = np.where((wave >= wl_min_o1) & (wave <= wl_max_o1))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    
    if wl_min_o2 is not None and wl_max_o2 is not None and order == 2:
        ii = np.where((wave >= wl_min_o2) & (wave <= wl_max_o2))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    wavelength = wave
    wavelength_err = wave_err
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    fluxcube_err = np.array(fluxcube_err)

    return wavelength,wavelength_err, t, fluxcube, fluxcube_err

def unpack_nirspec_exotedrf(infile, instrument, trim_start, trim_end, wl_min=None, wl_max=None):    
    print(wl_min, wl_max)
    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)
    fluxcube_err = fits.getdata(infile, 4)
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]
    print(bjd)
    print(wave)
    
    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    if instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/G395H':
        ii = np.where((wave >= 2.9) & (wave <= 5.0))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    if instrument == 'NIRSPEC/PRISM':
        ii = np.where((wave >= 0.6) & (wave <= 5.0))[0]
        fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    if instrument == 'NIRSPEC/G140H':
        ii = np.where((wave >= 1.0) & (wave <= 1.8))[0]
        fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    if wl_min is not None and wl_max is not None:
        ii = np.where((wave >= wl_min) & (wave <= wl_max))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]

    wavelength = wave
    wavelength_err = wave_err
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    fluxcube_err = np.array(fluxcube_err)
    return wavelength, wavelength_err,  t, fluxcube, fluxcube_err

def unpack_miri_exotedrf(infile, trim_start, trim_end, wl_min=None, wl_max=None):

    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)
    fluxcube_err = fits.getdata(infile, 4)
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    fluxcube_err = fluxcube_err[:, 5:-5]


    start = 0 if (trim_start is None) else int(trim_start)
    stop  = None if (trim_end in (None, 0)) else -int(trim_end)

    fluxcube     = fluxcube[start:stop, :]
    fluxcube_err = fluxcube_err[start:stop, :]
    bjd            = bjd[start:stop]   # keep time aligned!


    ii = np.where((wave > 5) & (wave <= 12))[0]
    fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
    wave, wave_err = wave[ii], wave_err[ii]
    ii = np.argsort(wave)
    wave, wave_err = wave[ii], wave_err[ii]
    fluxcube, fluxcube_err = fluxcube[:,ii], fluxcube_err[:,ii]

    if wl_min is not None and wl_max is not None:
        ii = np.where((wave >= wl_min) & (wave <= wl_max))[0]
        fluxcube, fluxcube_err = fluxcube[:, ii], fluxcube_err[:,ii]
        wave, wave_err = wave[ii], wave_err[ii]
    
    wavelength = wave
    wavelength_err = wave_err
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    fluxcube_err = np.array(fluxcube_err)
    return wavelength, wavelength_err, t, fluxcube, fluxcube_err

