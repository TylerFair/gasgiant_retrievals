
def bin_at_pixel(wave, flux, error, npix):
    """Similar to bin_at_resolution, but will bin in widths of a set number of pixels instead of
    at a fixed resolution.

    Parameters
    ----------
    wave : array-like[float]
        Input wavelength axis.
    flux : array-like[float]
        Flux values.
    error : array-like[float]
        Flux error values.
    npix : int
        Number of pixels per bin.

    Returns
    -------
    wave_bin : np.ndarray[float]
        Central bin wavelength.
    wave_err : np.ndarray[float]
        Wavelength bin half widths.
    dout : np.ndarray[float]
        Binned depth.
    derrout : np.ndarray[float]
        Error on binned depth.
    """

    flux = flux.T
    error = error.T

    # Calculate number of bins given wavelength grid and npix value.
    nint, nwave = np.shape(flux)
    # If the number of pixels does not bin evenly, trim from beginning and end.
    if nwave % npix != 0:
        cut = nwave % npix
        cut_s = int(np.floor(cut/2))
        cut_e = -1*(cut - cut_s)
        flux = flux[:, cut_s:cut_e]
        error = error[:, cut_s:cut_e]
        wave = wave[cut_s:cut_e]
        nint, nwave = np.shape(flux)
    nbin = int(nwave / npix)

    # Sum flux in bins and calculate resulting errors.
    flux_bin = np.nansum(np.reshape(flux, (nint, nbin, npix)), axis=2)
    err_bin = np.sqrt(np.nansum(np.reshape(error, (nint, nbin, npix))**2, axis=2))
    # Calculate mean wavelength per bin.
    wave_bin = np.nanmean(np.reshape(wave, (nbin, npix)), axis=1)
    wave_err = make_bins(wave_bin)[1] / 2
    flux_bin = flux_bin.T
    err_bin = err_bin.T
    return wave_bin, wave_err, flux_bin, err_bin
