import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd 
import celerite2 
from celerite2 import terms
from scipy.optimize import minimize
import jax 
import jax.numpy as jnp 

def outlier_rejection(wavelength, fluxcube, fluxcube_err, sigma):
    meanflux = np.nanmedian(fluxcube,axis=0)
    fluxcube_orig = fluxcube * 1.0 
    fluxcube_cleaned = fluxcube * 1.0 
    fluxcube_err_cleaned = fluxcube_err * 1.0 

    output_mask = np.ones_like(fluxcube, dtype=bool)

    for time_slice in range(len(fluxcube)):
        
        y = fluxcube[time_slice,:]
        yerr = fluxcube_err[time_slice,:]
        x = wavelength

        finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0) 

        x_finite = x[finite_mask]
        y_finite = y[finite_mask]
        yerr_finite = yerr[finite_mask]
        sort_inds = np.argsort(x_finite)
        x_finite = x_finite[sort_inds]
        y_finite = y_finite[sort_inds]
        yerr_finite = yerr_finite[sort_inds]

        mean_value = np.nanmean(y_finite) 
        term1 = terms.Matern32Term(sigma=np.nanstd(y_finite), rho=0.2)
        kernel = term1

        gp = celerite2.GaussianProcess(kernel, mean=mean_value)
        gp.compute(x_finite, yerr=yerr_finite)
        #print("Initial log likelihood: {0}".format(gp.log_likelihood(y_finite)))
        def set_params(params, gp):
            gp.mean = params[0]
            theta = np.exp(params[1:])
            gp.kernel = terms.Matern32Term(
                sigma=theta[0], rho=theta[1]
            ) 
            gp.compute(x_finite, diag=theta[2], quiet=True)
            return gp
        def neg_log_like(params, gp):
            try:
                gp = set_params(params, gp)
                val = -gp.log_likelihood(y_finite)
                return val if np.isfinite(val) else np.inf
            except (ValueError, np.linalg.LinAlgError):
                return np.inf        # invalid step

        initial_params = [mean_value, np.log(np.std(y_finite)), np.log(0.2), np.log(np.std(y_finite))]
        bounds = [(0, np.max(y_finite)),  # Bounds for mean
          (np.log(np.nanstd(y_finite) / 10), np.log(np.nanstd(y_finite) * 10)), #sigma 
          (np.log(0.1), np.log(10)),
            (np.log(np.nanstd(y_finite) / 10), np.log(np.nanstd(y_finite) * 10))] # rho 
        soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,), bounds=bounds)
        opt_gp = set_params(soln.x, gp)
        mu, variance = opt_gp.predict(y_finite, t=x_finite, return_var=True)
       # sigma = np.sqrt(variance)
        residuals = y_finite - mu

        mad_of_residual = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))

        def plot_prediction_opt(gp, t, y, m, s):
            plt.scatter(t, y, alpha=1, c='k', s=6, label="data")
            plt.plot(t, m, label="prediction")
            plt.fill_between(t, m - sigma*mad_of_residual , m + sigma*mad_of_residual, color="C0", alpha=0.8)
            plt.title("optimal prediction")
        #plot_prediction_opt(opt_gp, x_finite, y_finite, mu, sigma)
        #plt.show()

  
        #print(mad_of_residual)
        '''
        if time_slice == 100:
            plt.scatter(x_finite, residuals)
            plt.axhline(np.nanmedian(residuals)-10*mad_of_residual, ls='-', c='r')
            plt.axhline(np.nanmedian(residuals)+10*mad_of_residual, ls='-', c='r')
            plt.axhline(np.nanmedian(residuals)-7*mad_of_residual, ls='-', c='b')
            plt.axhline(np.nanmedian(residuals)+7*mad_of_residual, ls='-', c='b')
            plt.axhline(np.nanmedian(residuals)-5*mad_of_residual, ls='-', c='purple')
            plt.axhline(np.nanmedian(residuals)+5*mad_of_residual, ls='-', c='purple')
            plt.axhline(np.nanmedian(residuals)-4*mad_of_residual, ls='-', c='pink')
            plt.axhline(np.nanmedian(residuals)+4*mad_of_residual, ls='-', c='pink')
            plt.savefig(f'trace_sigma_check.png')
            plt.close()
        '''
        outlier_mask_finite = np.abs(residuals - np.nanmedian(residuals)) > sigma * mad_of_residual

        outlier_indices_in_sorted_finite = np.where(outlier_mask_finite)[0]
        original_indices_in_finite = sort_inds[outlier_indices_in_sorted_finite]
        global_finite_indices = np.where(finite_mask)[0]
        outlier_indices_in_original = global_finite_indices[original_indices_in_finite]

        output_mask[time_slice, outlier_indices_in_original] = False


        fluxcube_cleaned[time_slice, outlier_indices_in_original] = np.nan
        '''
        if time_slice == 100:
            plt.scatter(wavelength, fluxcube_orig[time_slice,:], s=12)
            plt.scatter(wavelength, fluxcube_cleaned[time_slice,:], c='r', s=12, zorder=3)
            plt.savefig(f'trace_gp_check.png')
            plt.close()
            #plt.show()
        '''
    return output_mask

def outlier_rejection_median(bjd, wavelength, fluxcube, fluxcube_err, sigma, prior_t0, prior_duration):
    meanflux = np.nanmedian(fluxcube,axis=0)
    fluxcube_orig = fluxcube * 1.0 
    fluxcube_cleaned = fluxcube * 1.0 
    fluxcube_err_cleaned = fluxcube_err * 1.0 

    output_mask = np.ones_like(fluxcube, dtype=bool)

    #here, fluxcube currently has shape of time, wavelengths
    # i need to clip to out of transit baseline only,
    # sum over all times to produce a median wavelength vs flux 
    # and also produce the std deviation of this median frame
    # then apply the outlier mask from this xsigma to the raw fluxcube

    t0 = prior_t0
    duration = prior_duration
    in_transit_mask = (bjd >= (t0 - 0.75*duration)) & (bjd < (t0 + 0.75*duration))
    print(np.shape(wavelength))
    # this created a flux vs wavelength averaged over all times
    medianflux = np.nanmedian(fluxcube[~in_transit_mask,:],axis=0)
    # this created a std deviation of the flux vs wavelength averaged over all times
    stdflux =1.4826 * np.nanmedian(np.abs(fluxcube[~in_transit_mask,:] - medianflux), axis=0) #np.nanstd(fluxcube[~ootmask,:],axis=0)
    # this created a mask of outliers, where the flux is greater than median + xsigma * std
    outlierclip = np.abs(fluxcube - medianflux) > sigma * stdflux 

    '''
    for i in range(fluxcube.shape[0]):
        plt.scatter(wavelength, fluxcube[i], s=12, c='k', alpha=0.5)
        # plot outliers clipped
        plt.scatter(wavelength[outlierclip[i]], fluxcube[i][outlierclip[i]], s=12, c='r', alpha=0.5)
        plt.show()
    ''' 
    fluxcube_cleaned[outlierclip] = np.nan
    fluxcube_err_cleaned[outlierclip] = np.nan
    output_mask[outlierclip] = False
    return output_mask 

#def compile_spectra(x1dfits,seg, nrs, planet_str, output_dir, aperturekey='FLUX'):
#    return bjd,wave,fluxcube,fluxcube_err

def get_whitelightlc(fluxcube,fluxcube_err):
    whitelightflux = np.nanmean(fluxcube,axis=1) 
    return whitelightflux

def get_rms_colour(fluxcube,fluxcube_err,wavelength):
    #meanflux = np.nanmedian(fluxcube,axis=0)
    #fluxcube /= meanflux

    wavebin_rms = []
    i = 0
    binsize = 10
    print(fluxcube.shape)
    while i < len(fluxcube[0]):
        #flux_i = np.nanmedian(fluxcube[:,i:i+binsize],axis=1)
        flux_i = get_whitelightlc(fluxcube[:,i:i+binsize],fluxcube_err[:,i:i+binsize])
        flux_i/=np.nanmedian(flux_i)

        #plt.plot(flux_i, '.')
        #plt.show()
        mad = np.nanmedian(abs(np.diff(flux_i)))
        wave_i = np.nanmedian(wavelength[i:i+binsize])
        wavebin_rms.append([wave_i,mad])
        i += binsize
        print(i, mad)
    wavebin_rms = np.array(wavebin_rms)

    return wavebin_rms

def join_whitelight(whitelightlist):

    whitelightnew = list(whitelightlist[0])
    for i in np.arange(1,len(whitelightlist)):
        whitelightnew += list(whitelightlist[i])

    return np.array(whitelightnew)/np.nanmedian(whitelightnew)
                       

def unpack_nirspec(file_pattern, nrs, planet_str, output_dir, bins_wanted, prior_t0, prior_duration):
    x1dfitslist = np.sort(glob.glob(file_pattern))
    
    fluxcubeall = []
    fluxcubeall_err = []
    wavelength = None
    t = []
    aperturekey = 'FLUX'
    for i in range(len(x1dfitslist)):
        x1dfits = x1dfitslist[i]
        #bjd,wl,fluxcube,fluxcube_err = compile_spectra(x1dfits,i, nrs, planet_str, output_dir, aperturekey='FLUX')
        print(x1dfits)
        x1dfits = fits.open(x1dfits)
        try:
            bjd = x1dfits[1].data['int_mid_BJD_TDB'][1:]
        except IndexError:
            bjd = np.arange(len(x1dfits)-3)
        wave = x1dfits[2].data['WAVELENGTH']
        wave_mask = wave > 2.86
        wave = wave[wave_mask]
        
        fluxcube = []
        fluxcube_err = []
        for j in np.arange(3, len(x1dfits)):
            fluxcube.append(x1dfits[j].data[aperturekey])
            fluxcube_err.append(x1dfits[j].data["FLUX_ERROR"])
        
        fluxcube, fluxcube_err = np.array(fluxcube), np.array(fluxcube_err)
        
        # Apply wavelength cut to flux data
        fluxcube = fluxcube[:, wave_mask]
        fluxcube_err = fluxcube_err[:, wave_mask]

        fluxcubeall += list(fluxcube)
        fluxcubeall_err += list(fluxcube_err)
        wavelength = wave
        t.append(bjd)

    t = np.concatenate(t)
    fluxcube = np.array(fluxcubeall)
    fluxcube_err = np.array(fluxcubeall_err)

    col_sigma = 4
    
    if os.path.exists(f'{output_dir}/{planet_str}_mask_check_NRS{nrs}_{col_sigma}sigma.npy'):
        print('Column rejection files already exist...')
        outlier_mask = np.load(f'{output_dir}/{planet_str}_mask_check_NRS{nrs}_{col_sigma}sigma.npy')
        plt.imshow(outlier_mask, aspect='auto')
        plt.savefig(f'{output_dir}/column_rejection_mask_NRS{nrs}_{col_sigma}sigma')
        plt.close()
        print("Columns rejected have been set to NaN.")
    else: 
        print('Running column rejection...')
        #outlier_mask = outlier_rejection(wave, fluxcube, fluxcube_err, sigma=col_sigma)
        outlier_mask = outlier_rejection_median(t, wavelength, fluxcube, fluxcube_err, col_sigma, prior_t0, prior_duration)
        data = np.save(f'{output_dir}/{planet_str}_mask_check_NRS{nrs}_{col_sigma}sigma.npy', arr=outlier_mask)
        plt.imshow(outlier_mask, aspect='auto', vmin=0.5, vmax=1, cmap='seismic_r')
        plt.savefig(f'{output_dir}/column_rejection_mask_NRS{nrs}_{col_sigma}sigma')
      #  plt.show()
        plt.close()
        #plt.show()
        print("Columns rejected have been set to NaN.")
    
    fraction_outliers_per_column = np.mean(~outlier_mask, axis=0)
    reject_wavelength_mask = fraction_outliers_per_column > 0.5
    fluxcube[:, reject_wavelength_mask] = np.nan
    #fluxcube[~outlier_mask] = np.nan  # Remove specific flagged pixels
    
    base = np.concatenate([np.arange(100), np.arange(100)-100]).astype(int) 
    median_flux = np.nanmedian(fluxcube[base], axis=0)
    fluxcube = fluxcube / median_flux


    wavelengths = jnp.array(wavelength)  
    indiv_y = jnp.array(fluxcube)  

    
    
    return wavelengths, t, indiv_y, None


def unpack_niriss_exoted(infile, order, planet_str, output_dir, bins_wanted=50):    

    bjd = fits.getdata(infile, 9)
    wave = fits.getdata(infile, 1 + 4 * (order - 1))
    wave_err = fits.getdata(infile, 2 + 4 * (order - 1))
    fluxcube = fits.getdata(infile, 3 + 4 * (order - 1))
    wave = wave[5:-5]
    wave_err = wave_err[5:-5]
    fluxcube = fluxcube[:, 5:-5]
    base = np.concatenate([np.arange(150), np.arange(100)-100]).astype(int) 

    median_flux = np.nanmedian(fluxcube[base], axis=0)
  
    fluxcube = fluxcube / median_flux

    if order == 2:
        ii = np.where((wave >= 0.6) & (wave <= 0.85))[0]
        fluxcube = fluxcube[:, ii]
        wave, wave_err = wave[ii], wave_err[ii]

    wavelength = wave
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    
    return wavelength, t, fluxcube, None

def unpack_nirspec_exoted(infile):    

    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)

    base = np.concatenate([np.arange(100), np.arange(100)-100]).astype(int) 

    median_flux = np.nanmedian(fluxcube[base], axis=0)
  
    fluxcube = fluxcube / median_flux

  
    ii = np.where(wave >= 2.87)[0]
    fluxcube = fluxcube[:, ii]
    wave, wave_err = wave[ii], wave_err[ii]

    wavelength = wave
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    
    return wavelength, t, fluxcube, None

def unpack_miri_exoted(infile):

    bjd = fits.getdata(infile, 5)
    wave = fits.getdata(infile, 1)
    wave_err = fits.getdata(infile, 2)
    fluxcube = fits.getdata(infile, 3)
    ii = np.where((wave > 5) & (wave <= 12))[0]
    fluxcube = fluxcube[:, ii]
    wave, wave_err = wave[ii], wave_err[ii]
    ii = np.argsort(wave)
    wave, wave_err = wave[ii], wave_err[ii]
    fluxcube = fluxcube[:,ii]
    
    base = -1 - np.arange(200).astype(int) # np.conicatenate([np.arange(100), np.arange(100)-100]).astype(int)

    median_flux = np.nanmedian(fluxcube[base], axis=0)

    fluxcube = fluxcube / median_flux



    wavelength = wave
    t = np.array(bjd)
    fluxcube = np.array(fluxcube)
    return wavelength, t, fluxcube, None
