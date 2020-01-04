import fooof
import xarray as xr
import numpy as np


def fit_spectrum(data: xr.DataArray):
    pass


def _fit_single_spectrum(frequencies: np.array, electrode_spectrum: np.array):
    """Fit one spectrum.

    Returns the background and foreground models separately.

    Parameters
    ----------
    frequencies : np.array
    electrode_spectrum : np.array

    Returns
    -------
    np.array
        The background spectrum
    np.array
        The peak spectrum
    """
    model = fooof.FOOOF(0.0, 12.0)
    model.fit(frequencies.squeeze(), electrode_spectrum.squeeze())
    return model._bg_fit, model._peak_fit
