#!/usr/bin/env python
"""
Example script to reload and replot detrended light curves from saved CSV files.

Usage:
    python replot_lightcurves.py output/planet_instrument_R40_lightcurves.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse


def plot_detrended_multiwavelength(lc_csv, params_csv=None, max_wavelengths=10):
    """
    Plot detrended light curves for multiple wavelengths.

    Parameters
    ----------
    lc_csv : str
        Path to the lightcurves CSV file
    params_csv : str, optional
        Path to the bestfit parameters CSV file
    max_wavelengths : int
        Maximum number of wavelengths to plot
    """

    # Load the data
    print(f"Loading data from {lc_csv}")
    df = pd.read_csv(lc_csv)

    # Get unique wavelengths
    wavelengths = df['wavelength'].unique()
    n_wavelengths = len(wavelengths)
    print(f"Found {n_wavelengths} wavelengths")

    # Select subset if too many
    if n_wavelengths > max_wavelengths:
        indices = np.linspace(0, n_wavelengths - 1, max_wavelengths, dtype=int)
        wavelengths = wavelengths[indices]
        print(f"Plotting {max_wavelengths} wavelengths")

    # Create figure with multiple panels
    n_cols = 3
    n_rows = int(np.ceil(len(wavelengths) / n_cols))

    fig = plt.figure(figsize=(15, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    # Color map
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    colors = plt.cm.viridis(norm(wavelengths))

    for idx, wl in enumerate(wavelengths):
        ax = plt.subplot(gs[idx // n_cols, idx % n_cols])

        # Get data for this wavelength
        wl_data = df[df['wavelength'] == wl]

        # Plot detrended flux
        ax.scatter(wl_data['time'], wl_data['detrended_flux'],
                   c=[colors[idx]], s=3, alpha=0.5, label='Detrended data')

        # Plot transit model on top
        ax.plot(wl_data['time'], wl_data['transit_model'] + 1.0,
                c='red', lw=2, label='Transit model', zorder=5)

        # Calculate RMS
        residuals = wl_data['residual'].values
        rms = np.nanmedian(np.abs(np.diff(residuals))) * 1e6

        # Labels
        ax.text(0.05, 0.95, f'λ = {wl:.3f} μm\nRMS = {rms:.0f} ppm',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlabel('Time (BJD)')
        ax.set_ylabel('Detrended Flux')

        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Detrended Multi-wavelength Light Curves', fontsize=14)
    plt.tight_layout()

    return fig


def plot_all_components(lc_csv, wavelength_idx=0):
    """
    Plot all components (raw, model, trend, detrended) for a single wavelength.

    Parameters
    ----------
    lc_csv : str
        Path to the lightcurves CSV file
    wavelength_idx : int
        Index of wavelength to plot
    """

    # Load the data
    df = pd.read_csv(lc_csv)
    wavelengths = df['wavelength'].unique()
    wl = wavelengths[wavelength_idx]

    print(f"Plotting components for wavelength {wl:.3f} μm")

    # Get data for this wavelength
    wl_data = df[df['wavelength'] == wl]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1. Raw flux
    axes[0].errorbar(wl_data['time'], wl_data['flux_raw'],
                     yerr=wl_data['flux_err'], fmt='.', alpha=0.5, c='k')
    axes[0].plot(wl_data['time'], wl_data['full_model'], 'r-', lw=2, label='Full model')
    axes[0].set_ylabel('Raw Flux')
    axes[0].legend()
    axes[0].set_title(f'Wavelength: {wl:.3f} μm')

    # 2. Transit model only
    axes[1].plot(wl_data['time'], wl_data['transit_model'] + 1.0, 'b-', lw=2)
    axes[1].set_ylabel('Transit Model')
    axes[1].axhline(1.0, c='gray', ls='--', alpha=0.5)

    # 3. Trend
    axes[2].plot(wl_data['time'], wl_data['trend'], 'g-', lw=2)
    axes[2].set_ylabel('Trend')
    axes[2].axhline(1.0, c='gray', ls='--', alpha=0.5)

    # 4. Detrended flux
    axes[3].errorbar(wl_data['time'], wl_data['detrended_flux'],
                     yerr=wl_data['flux_err'], fmt='.', alpha=0.5, c='k')
    axes[3].plot(wl_data['time'], wl_data['transit_model'] + 1.0, 'r-', lw=2)
    axes[3].set_ylabel('Detrended Flux')
    axes[3].set_xlabel('Time (BJD)')
    axes[3].axhline(1.0, c='gray', ls='--', alpha=0.5)

    plt.tight_layout()

    return fig


def plot_transmission_spectrum(params_csv):
    """
    Plot transmission spectrum from bestfit parameters.

    Parameters
    ----------
    params_csv : str
        Path to the bestfit parameters CSV file
    """

    # Load the data
    df = pd.read_csv(params_csv)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot depth vs wavelength
    ax.errorbar(df['wavelength'], df['depth'] * 1e6,
                yerr=df['depth_err'] * 1e6,
                fmt='o', mfc='k', mec='k', ecolor='k', capsize=3)

    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Transit Depth (ppm)', fontsize=12)
    ax.set_title('Transmission Spectrum', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Replot saved light curve data')
    parser.add_argument('lightcurves_csv', help='Path to lightcurves CSV file')
    parser.add_argument('--params', help='Path to bestfit parameters CSV file (optional)')
    parser.add_argument('--max-wl', type=int, default=10, help='Max wavelengths to plot')
    parser.add_argument('--output', help='Output filename for plot (optional)')
    parser.add_argument('--components', type=int, help='Plot all components for wavelength index')

    args = parser.parse_args()

    if args.components is not None:
        # Plot all components for a single wavelength
        fig = plot_all_components(args.lightcurves_csv, wavelength_idx=args.components)
        if args.output:
            plt.savefig(args.output, dpi=200, bbox_inches='tight')
            print(f"Saved to {args.output}")
        else:
            plt.show()

    else:
        # Plot detrended multi-wavelength light curves
        fig = plot_detrended_multiwavelength(args.lightcurves_csv,
                                             params_csv=args.params,
                                             max_wavelengths=args.max_wl)
        if args.output:
            plt.savefig(args.output, dpi=200, bbox_inches='tight')
            print(f"Saved to {args.output}")
        else:
            plt.show()

    # If params CSV provided, also plot transmission spectrum
    if args.params:
        fig2 = plot_transmission_spectrum(args.params)
        if args.output:
            output2 = args.output.replace('.png', '_spectrum.png')
            plt.savefig(output2, dpi=200, bbox_inches='tight')
            print(f"Saved spectrum to {output2}")
        else:
            plt.show()


if __name__ == '__main__':
    main()
