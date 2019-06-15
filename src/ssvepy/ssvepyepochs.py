import collections
import inspect
import pickle
from copy import deepcopy
from typing import Sequence, Optional, Union

from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import sklearn.linear_model
import xarray as xr
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import lfilter

# import fooof

from . import frequencymaths


@dataclass
class EvokedFrequency:

    psd: xr.DataArray
    snr: xr.DataArray
    tfr: Optional[xr.DataArray] = None


class Ssvep(mne.Epochs):
    """
    Turns epoched EEG/MEG data into epoched evoked frequency data.

    This is the main class in ``ssvepy``.

    Args
    ----
        epochs : :class:`mne:mne.Epochs`
                An instance of Epoch data.
        stimulation_frequency : int, float, or list
                The frequencies at which the stimulation oscillated.
        noisebandwidth : float
                The width of the noise band used to calculate the
                signal-to-noise ratio.
        compute_harmonics : list | :class:`np:numpy.ndarray`
                Integers of which order of harmonics to compute. Can be None.
        compute_subharmonics : list | :class:`np:numpy.ndarray`
                Integers of which order of harmonics to compute. Can be None.
        compute_intermodulation : list | :class:`np:numpy.ndarray`
                Integers of which order of harmonics to compute. Can be None.
        psd : :class:`np:numpy.ndarray`, :class:`xr:xarray.DataArray`, str
            If you have already computed the power spectrum with some
            other method, pass it as a parameter. Otherwise, you can pass
            either 'multitaper' or 'welch' to indicate what method should be
            used to calculate the power spectrum.
        freqs : :class:`np:numpy.ndarray`
            If you provide a power spectrum, this has to be the
            frequencies over which it was evaluated.
        fmin, fmax : float
            Bounds of frequency band to be evaluated
        tmin, tmax : float
            The time points between which power will be evaluated
        compute_tfr : bool
            If you want to evaluate the time-frequency decomposition
            (this applies to the stimulation and non-linear combination
            frequencies only.)
        tfr_method : str
            Currently, only one method is implemented ('rls')
        tfr_time_window : float
            The window width for the TFR method.

    Attributes
    ----------
        stimulation : obj
            a data structure with the following attributes:
            ``stimulation.frequencies``, ``stimulation.power``,
            ``stimulation.snr``
        harmonics, subharmonics, intermodulations : obj
            non-linear combination of
            your input stimulus frequencies, all with the attributes:
            ``_.frequencies``, ``_.power``, ``_.snr``, ``_.order``
        psd : :class:`np:numpy.ndarray`
            the power spectrum
        freqs : :class:`np:numpy.ndarray`
            the frequencies at which the psd was evaluated
        snr : :class:`np:numpy.ndarray`
            the signal-to-noise ratio for each frequency in freqs

    """

    def __init__(
        self,
        epochs: mne.Epochs,
        stimulation_frequencies: Sequence,
        noisebandwidth: float = 1.0,
        compute_harmonics: Optional[Sequence[int]] = range(2, 5),
        compute_subharmonics: Optional[Sequence[int]] = range(2, 5),
        compute_intermodulation: Optional[Sequence[int]] = range(2, 5),
        psd: Union[np.ndarray, xr.DataArray, str] = "multitaper",
        freqs: np.ndarray = None,
        fmin: float = 0.1,
        fmax: float = 50,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        compute_tfr: bool = False,
        tfr_method: str = "rls",
        tfr_time_window: float = 0.9,
        compute_fooof_peaks: bool = True,
    ):

        self.info = deepcopy(epochs.info)
        self.events = deepcopy(epochs.events)

        self.noisebandwidth = noisebandwidth
        self.psd = psd
        self.frequencies = freqs
        self.fmin = fmin
        self.fmax = fmax

        if not isinstance(self.psd, str) and self.frequencies is None:
            raise ValueError(
                "If you provide psd data, you also need to provide"
                " the frequencies at which it was evaluated"
            )

        # If no power-spectrum was provided, we need to work it out
        if self.psd == "multitaper":
            # Use MNE here.
            self.psd, self.frequencies = mne.time_frequency.psd_multitaper(
                epochs, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
            )
        elif self.psd == "welch":
            # Use MNE here.
            self.psd, self.frequencies = mne.time_frequency.psd_welch(
                epochs, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
            )
        else:
            raise NotImplementedError(
                "psd has to be either 'welch' or 'multitaper' if it is not an "
                "array. You passed {}".format(self.psd)
            )

        # turn into a labelled array
        self.psd = xr.DataArray(
            self.psd,
            dims=["epoch", "channel", "frequency"],
            coords={
                "epoch": self.events[:, 2],
                "channel": [
                    epochs.ch_names[a]
                    for a in mne.pick_types(epochs.info, meg=True, eeg=True)
                ],
                "frequency": self.frequencies,
            },
        )

        self.frequency_resolution = self.frequencies[1] - self.frequencies[0]

        self.snr = self._get_snr(self.frequencies)

        if not isinstance(stimulation_frequencies, Sequence):
            stimulation_frequencies = [stimulation_frequencies]

        self.stimulation_frequencies = np.array(
            stimulation_frequencies, dtype=float
        )

        self.stimulation = EvokedFrequency(
            psd=self.psd.interp(
                frequency=self.stimulation_frequencies, method="cubic"
            ),
            snr=self.snr.interp(
                frequency=self.stimulation_frequencies, method="cubic"
            ),
        )

    # Helper functions to get specific frequencies:
    def _get_snr(self, frequencies: np.ndarray):
        """
        Helper function to work out the SNR of a given frequency
        """
        snr = []
        f = self.psd.coords["frequency"].data
        stimulation_bands = np.zeros_like(f, dtype=bool)
        # for frequency in frequencies:
        #     stimulation_bands = stimulation_bands | (
        #         (f <= (frequency + self.frequency_resolution))
        #         & (f >= (frequency - self.frequency_resolution))
        #     )
        #     print(stimulation_bands)
        for frequency in frequencies:
            noise = (
                # within the noise band:
                (
                    (f <= (frequency + self.noisebandwidth))
                    & (f >= (frequency - self.noisebandwidth))
                )
                # not within the stimulation band:
                & ~stimulation_bands
            )

            snr.append(
                self.psd.mean("epoch").interp(
                    frequency=frequency, method="cubic"
                )
                / self.psd[:, :, noise].mean("epoch").mean("frequency")
            )

        return (
            xr.concat(snr, dim=xr.DataArray(frequencies, name="frequency"))
            .rename({"dim_0": "frequency"})
            .transpose("channel", "frequency")
        )

    def _get_amp(self, freqs: np.ndarray):
        """
        Helper function to get the freq-smoothed amplitude of a frequency
        """
        return self.psd.interp(frequency=freqs, method="cubic")

    # def _fit_fooof(self):
    #     self.fooof_peaks = xr.Dataset()
    #     self._fooof = fooof.FOOOF(background_mode="fixed")
    #     for channel in self.psd.coords["channel"]:
    #         self.fooof.fit(self.psd.mean("epochs"))

    def _compute_tfr(
        self,
        epoch: mne.Epochs,
        freq: Union[np.ndarray, xr.DataArray, pd.Index],
        tfr_method: str = "rls",
        window_width: float = 1.2,
        filter_lambda: float = 1.0,
    ) -> xr.DataArray:
        """
        Work out the time-frequency composition of different frequencies.
        """

        data: np.ndarray = epoch.get_data()

        samplefreq = epoch.info["sfreq"]
        n_window = int(samplefreq * window_width)
        if filter_lambda == 1:
            lambdafilter = np.ones(n_window) / (n_window / 2)
        else:
            lambdafilter = np.power(filter_lambda, np.arange(n_window))

        t = np.arange(data.shape[-1]) / samplefreq

        channels = mne.pick_types(self.info, meg=True, eeg=True)
        data = data[:, channels, :]
        # create a data structure that matches MNE standard TFR shape
        tfr_data = xr.DataArray(
            np.zeros((data.shape[0], channels.size, freq.size, data.shape[2])),
            coords=[
                ("epoch", self.psd.coords["epoch"]),
                ("channel", self.psd.coords["channel"]),
                ("frequency", freq),
                ("time", np.arange(data.shape[2]) / samplefreq),
            ],
        )

        if tfr_method == "rls":
            # this implementation follows Sucharit Katyal's code
            for fi, f in enumerate(freq.flatten()):
                sin_component = -np.sin(2 * np.pi * f * t)
                cosin_component = np.cos(2 * np.pi * f * t)
                h_cosin = lfilter(lambdafilter, 1, data * cosin_component)
                h_sin = lfilter(lambdafilter, 1, data * cosin_component)
                if filter_lambda < 1:
                    h_cosin = h_cosin / lfilter(
                        lambdafilter, 1, cosin_component ** 2
                    )
                    h_sin = h_sin / lfilter(
                        lambdafilter, 1, sin_component ** 2
                    )
                # combine the data to get envelope
                a = np.abs(h_cosin + 1j * h_sin)
                # shift left, pad nan
                a = np.roll(a, -n_window // 2, axis=-1)
                a[:, :, (-n_window // 2) :] = np.nan
                a = gaussian_filter1d(a[:, :, :], n_window // 10)
                tfr_data[:, :, fi, :] = a

        else:
            raise NotImplementedError("Only RLS is available so far.")
        return tfr_data

    # Helper functions to compute non-linear frequency combos:
    def _compute_harmonics(
        self,
        epochs: mne.Epochs,
        orders: Sequence,
        compute_tfr: bool,
        tfr_time_window,
        type_: str = "harmonics",
    ) -> EvokedFrequency:
        """
        Calculate the harmonic frequency and return an EvokedFrequency named
        tuple.
        """
        if type_ == "harmonics":
            calculation_function = frequencymaths.compute_harmonics
        elif type_ == "subharmonics":
            calculation_function = frequencymaths.compute_subharmonics
        elif type_ == "intermodulation":
            calculation_function = frequencymaths.compute_harmonics

        frequencies, orders = calculation_function(
            self.stimulation.frequencies,
            fmin=self.fmin,
            fmax=self.fmax,
            orders=orders,
        )
        frequencies = frequencies.flatten()

        return EvokedFrequency(
            frequencies=frequencies,
            orders=orders,
            power=self._get_amp(frequencies),
            snr=self._get_snr(frequencies),
            tfr=(
                self._compute_tfr(
                    epochs, frequencies, window_width=tfr_time_window
                )
                if compute_tfr
                else None
            ),
        )

    def _compute_subharmonics(self, orders):
        """
        Helper function to compute the subharms from a list, while making sure
        they're in the frequency range
        """
        return frequencymaths.compute_subharmonics(
            self.stimulation.frequencies,
            fmin=self.fmin,
            fmax=self.fmax,
            orders=orders,
        )

    def _compute_intermodulations(self, orders):
        """
        Helper function to compute the intermods from a list, while making sure
        they're in the frequency range
        """
        return frequencymaths.compute_intermodulation(
            self.stimulation.frequencies,
            fmin=self.fmin,
            fmax=self.fmax,
            orders=orders,
        )

    # Machine learning methods:
    def predict_timepoints(
        self,
        labels,
        trainingtrials=None,
        datatransform=None,
        method=sklearn.linear_model.LogisticRegressionCV(),
    ):
        """
        This method is for predicting the labels of trials based on the SSVEP
        power in a trial.

        Args
        ----
            labels : :class:`np:numpy.ndarray`
                Labels for each timepoint; the dimensions should match
                (trial x timepoint)

            trainepochs : list
                A list of trial indices that will be used to train - e.g.
                range(6) - or an ndarray of booleans of the same size as label.
                The trials in this index will be used as training trials, and
                the returned accuracy is based on the trials *not* used for
                training.
            datatransform : str
                What transform to do to the TFR data - at the moment, 'z-score'
                works.

            method : sklearn.class
                A training class that conforms to the standard sklearn model
                (ie has the methods fit(), predict() etc.)

        """
        pass

    def predict_epochs(
        self,
        labels,
        trainepochs=None,
        method=sklearn.linear_model.LogisticRegressionCV(),
    ):
        """
        This method is for predicting the labels of trials based on the SSVEP
        power in a trial.

        Args
        ----
            labels : :class:`np:numpy.ndarray`
                Labels for each trial; needs to be the same length as there are
                epochs.
            trainepochs : list
                A list of trial indices that will be used to train - e.g.
                range(6) - or an ndarray of booleans of the same size as label.
                The trials in this index will be used as training trials, and
                the returned accuracy is based on the trials *not* used for
                training.
            method : sklearn.class
                A training class that conforms to the standard sklearn model
                (ie has the methods fit(), predict() etc.)
        """

        # check the method as a fit method
        assert inspect.ismethod(
            method.fit
        ), "The provided method has no fit function"
        trainclass = method

        # check that the labels match up with the trial numbers
        labels = np.array(labels)  # make sure it's an array
        assert labels.size == len(
            self
        ), "Number of labels needs to match number of epochs."

        # reshape the training data and concatenate them all
        trainx = self.stimulation.snr.reshape((len(self), -1))
        for freqtype in ("harmonic", "subharmonic", "intermodulation"):
            # concatenate:
            freqs = self.__getattribute__(freqtype).frequencies.flatten()
            trainx = np.concatenate(
                (
                    trainx,
                    self.__getattribute__(freqtype)
                    .snr[..., ~np.isnan(freqs)]
                    .reshape((len(self), -1)),
                ),
                axis=-1,
            )

        # if the training trials
        if trainepochs is not None:
            trainepochs = np.array(trainepochs)
            if trainepochs.size == len(self):
                # probably a boolean
                testepochs = ~trainepochs
            else:
                # probably integer index
                testepochs = np.array(
                    [i for i in range(len(self)) if i not in trainepochs]
                )
        else:
            # use all epochs for both training and test
            trainepochs = range(len(self))
            testepochs = range(len(self))

        # train the model
        trainclass.fit(trainx[trainepochs, :], labels[trainepochs])
        # predict the model
        predictions = trainclass.predict(trainx[testepochs, :])

        return np.array(predictions)

    # Plotting methods

    def plot_tfr(
        self,
        frequency="stimulation",
        collapse_epochs=True,
        collapse_electrodes=False,
        figsize=(7, 5),
    ):
        """
        Plot the time-course of one of the evoked frequencies.

        Args
        ----
            frequency : str
                Which evoked frequency to plot. Either 'stimulation',
                'harmonic', 'subharmonic' or 'intermodulation'
            collapse_epochs : bool
                Whether to average over the epochs or not.
            collapse_electrodes : bool
                Whether to average over electrodes or not.
            figsize : tuple
                Matplotlib figure size.
        """

        if frequency is None or frequency == "stimulation":
            y = self.stimulation.tfr
            z = self.stimulation.frequencies
        elif type(frequency) is str:
            y = self.__getattribute__(frequency).tfr
            z = self.__getattribute__(frequency).frequencies

        x = np.arange(y.shape[-1]) / self.info["sfreq"]

        collapse_axes = tuple(
            [
                ax
                for ax, b in enumerate([collapse_epochs, collapse_electrodes])
                if b
            ]
        )
        if len(collapse_axes) > 0:
            y = y.mean(axis=collapse_axes)
        # Make time the first dimension
        y = np.rollaxis(y, -1)
        # Make a figure (-1 is now freq. dimension)
        nplots = y.shape[-1]
        nrows = int(np.ceil(np.sqrt(nplots)))
        ncols = int(np.ceil(nplots / nrows))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        # y = np.squeeze(y)
        for idx in range(nplots):
            # Choose axes to plot in
            ax = axes.flatten()[idx] if nplots > 1 else axes
            # Plot the individual lines
            ax.plot(x, y[..., idx], color="blue", alpha=0.1)
            # Plot the mean of the data
            if y[..., idx].size > y.shape[0]:
                ax.plot(x, y[..., idx].mean(axis=-1))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(str(z.flatten()[idx]) + " Hz")

        plt.show()

    def plot_psd(
        self,
        log_scale=True,
        collapse_epochs=True,
        collapse_electrodes=False,
        **kwargs,
    ):
        """
        Plot the power-spectrum that has been calculated for this data.

        Parameters
        ----------
        log_scale : bool
            Yes or no
        collapse_epochs : bool
            Whether you want to plot the average of all epochs (default),
            or each power-spectrum individually.
        collapse_electrodes : bool
            Whether you want to plot each electrode individually
            (default), or only the average of all electrodes.

        """
        ydata = self.psd
        if collapse_electrodes:
            ydata = ydata.mean("channel")
        if collapse_epochs:
            ydata = ydata.mean("epoch")
        return self._plot_spectrum(ydata, **kwargs)

    def plot_snr(
        self, collapse_epochs=True, collapse_electrodes=False, **kwargs
    ):
        """
        Plot the signal-to-noise-ratio-spectrum that has been calculated for
        this data.

        Parameters:
            collapse_epochs : bool
                Whether you want to plot the average of all epochs (default),
                or each power-spectrum individually.
            collapse_electrodes : bool
                Whether you want to plot each electrode individually
                (default), or only the average of all electrodes.

        """

        # Construct the SNR spectrum
        ydata = self.snr
        if collapse_electrodes:
            ydata = ydata.mean("channel")
        if collapse_epochs:
            ydata = ydata.mean("epoch")
        return self._plot_spectrum(ydata, **kwargs)

    def _plot_spectrum(
        self,
        ydata,
        figsize=(15, 7),
        log_scale=True,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        """
        Helper function to plot different spectra
        """

        scale = "log" if log_scale else None

        if fmin is None:
            fmin = np.min(self.frequencies)
        if fmax is None:
            fmax = np.max(self.frequencies)

        if "channel" not in ydata.dims and "epoch" not in ydata.dims:
            facetgrid = ydata.plot.line(x="frequency", color="blue")
        elif "channel" in ydata.dims and "epoch" not in ydata.dims:
            facetgrid = ydata.plot.line(
                x="frequency", color="blue", alpha=0.1, add_legend=False
            )
            ydata.mean("channel").plot.line(
                x="frequency", color="red", add_legend=False
            )
        elif "channel" in ydata.dims and "epoch" in ydata.dims:
            # plot each channel:
            facetgrid = ydata.plot.line(
                x="frequency",
                color="blue",
                hue="channel",
                col="epoch",
                alpha=0.1,
                yscale=scale,
                add_legend=False,
                col_wrap=int(ydata.coords["epoch"].size ** 0.5),
            )
            # plot the channel mean, too:
            for ax, name_dict in zip(
                facetgrid.axes.flatten(), facetgrid.name_dicts.flatten()
            ):
                ydata.mean("channel").loc[name_dict].plot.line(
                    x="frequency", color="red", add_legend=False, ax=ax
                )
        elif "channel" not in ydata.dims and "epoch" in ydata.dims:
            facetgrid = ydata.plot.line(
                x="frequency",
                color="blue",
                add_legend=False,
                col="epoch",
                col_wrap=int(ydata.coords["epoch"].size ** 0.5),
            )

        return facetgrid

    def topoplot_psd(
        self, collapse_epochs=True, flims="stimulation", **kwargs
    ):
        """
        Plot the signal-to-noise-ratio-spectrum across the scalp.

        Parameters:
            collapse_epochs : bool
                Whether you want to plot the average of all epochs (default),
                or each power-spectrum individually.
            flims : list | str
                Which frequency bands you want to plot. By default, the
                stimulation frequencies will be plotted. Can be limits (eg.
                [6, 8]) or a string referring to an evoked frequency (eg.
                'stimulation', 'harmonic')

        """
        # Find out over which range(s) to collapse:
        fmins, fmaxs = self._get_flims(flims)
        if collapse_epochs:
            topodata = [
                self.psd.sel(frequency=slice(fmin, fmax))
                .mean("frequency")
                .mean("epoch")
                .values
                for fmin, fmax in zip(fmins, fmaxs)
            ]
            annot = [
                f"{fmin:.2f} - {fmax:.2f} Hz"
                for fmin, fmax in zip(fmins.flatten(), fmaxs.flatten())
            ]

        return self._plot_topo(topodata, annot, **kwargs)

    def topoplot_snr(self, collapse_epochs=True, frequencies=None, **kwargs):
        """
        Plot the signal-to-noise-ratio-spectrum across the scalp.

        Parameters:
            collapse_epochs : bool
                Whether you want to plot the average of all epochs (default),
                or each power-spectrum individually.
            frequencies : list of ints
                Which frequency bands you want to plot. By default, the
                stimulation frequencies will be plotted. Can be limits (eg.
                [6, 8]) or a string referring to an evoked frequency (eg.
                'stimulation', 'harmonic')
        """

        # Find out over which range(s) to collapse:
        # fmins, fmaxs = self._get_flims(flims)
        if frequencies is None:
            frequencies = self.stimulation_frequencies

        topodata = [
            self.snr.interp(frequency=freq, method="linear")
            .mean("epoch")
            .values
            for freq in frequencies
        ]
        annot = [f"{freq} Hz" for freq in frequencies]

        return self._plot_topo(topodata, annot, **kwargs)

    def _get_flims(self, flims):
        """
        Helper function that turns strings or lists into limits
        """
        if flims is not None:
            if type(flims) is str:
                fmins = (
                    self.__getattribute__(flims).frequencies
                    - self.frequency_resolution
                )
                fmaxs = (
                    self.__getattribute__(flims).frequencies
                    + self.frequency_resolution
                )
            elif type(flims) is int or type(flims) is float:
                fmins = np.array([flims - 0.1])
                fmaxs = np.array([flims + 0.1])
            else:
                try:
                    if len(flims[0]) > 1:
                        fmins = [flim[0] for flim in flims]
                        fmaxs = [flim[1] for flim in flims]
                except TypeError:
                    fmins = flims[0]
                    fmaxs = flims[1]
        else:
            fmins = self.stimulation.frequencies - 0.1
            fmaxs = self.stimulation.frequencies + 0.1

        return fmins, fmaxs

    def _plot_topo(
        self,
        topodata,
        annotation=None,
        figsize=(5, 5),
        cmap="Blues",
        channels=None,
        vmin=None,
        vmax=None,
        ax=None,
    ):
        """
        Helper function to plot scalp distribution
        """
        # Get the montage
        pos = mne.channels.layout._auto_topomap_coords(
            self.info, mne.pick_types(self.info, meg=True, eeg=True)
        )
        if type(topodata) is not list:
            topodata = [topodata]
        if type(annotation) is not list:
            annotation = [annotation]
        # get the common vmin and vmax values across all our data
        if vmin is None:
            vmin = np.min([np.nanmin(t) for t in topodata])
        if vmax is None:
            vmax = np.max([np.nanmax(t) for t in topodata])
        # Make a Figure
        fig, axes = plt.subplots(
            nrows=int(np.ceil(np.sqrt(len(topodata)))),
            ncols=int(
                np.ceil(len(topodata) / np.ceil(np.sqrt(len(topodata))))
            ),
            figsize=figsize,
        )
        for idx, t in enumerate(topodata):
            t = t.flatten()
            # Check for infs or nans and remove them
            bads = np.isnan(t) | np.isinf(t)
            t = t[~bads]
            pos = pos[~bads, :]
            # pick correct axes
            ax = axes.flatten()[idx] if type(axes) is np.ndarray else axes
            im, _ = mne.viz.plot_topomap(
                t, pos, cmap=cmap, vmin=vmin, vmax=vmax, show=False, axes=ax
            )
            ax.set_title(annotation[idx])
        # add a colorbar:
        if type(axes) is np.ndarray:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
        else:
            fig.colorbar(im, ax=axes, shrink=0.7)
        # fig = plt.gcf()
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
        # plt.tight_layout()
        return fig, axes

    # File I/O and printing

    def save(self, filename):
        """
        Save the data to file.

        Parameters
        ----------
            filename : str
                The name of the file, either bare, or with the file
                extension .hdf5
        """
        if filename[-5:] != ".hdf5":
            filename = filename + ".hdf5"
        f = h5py.File(filename, "w")  # open a file
        # First the necessary data:
        f.create_dataset("psd", data=self.psd)
        f.create_dataset("freqs", data=self.frequencies)
        # whack in the info structure:
        info = f.create_group("info")
        info.attrs["infostring"] = np.void(pickle.dumps(self.info))
        # Store the various evoked frequency structures:
        for name, tup in [
            ("stimulation", self.stimulation),
            ("harmonic", self.harmonic),
            ("subharmonic", self.subharmonic),
            ("intermodulation", self.intermodulation),
        ]:
            g = f.create_group(name)
            for key, value in tup._asdict().items():
                if value is not None:
                    g.create_dataset(key, data=value)
        # Store the frequency resolution
        psdinfo = f.create_group("psdinfo")
        psdinfo.attrs["frequency_resolution"] = self.frequency_resolution
        psdinfo.attrs["fmin"] = self.fmin
        psdinfo.attrs["fmax"] = self.fmax
        psdinfo.attrs["noisebandwidth"] = self.noisebandwidth

        # Finally, close the file:
        f.close()

    def __repr__(self):
        outstring = (
            "ssvepy data structure based on epoched data.\n"
            "The stimulation frequency(s) is {stimfreq}.\n"
            "There are {nepoch} epochs.\n The power spectrum was "
            "evaluated over {nfreqs} frequencies ({fmin} Hz - "
            "{fmax} Hz).\n".format(
                stimfreq=self.stimulation.frequencies,
                nepoch=self.psd.shape[0],
                nfreqs=self.frequencies.size,
                fmin=self.frequencies.min(),
                fmax=self.frequencies.max(),
            )
        )
        return outstring

    def __len__(self):
        return self.psd.shape[0]

    def __getitem__(self, key):
        return self.psd[key, ...]

    def __iter__(self):
        for idx in range(len(self)):
            yield (self.psd[idx, ...])


def load_ssvep(filename):
    """
    Load an hdf5 file containing ssvepy.Ssvep data (saved with Ssvep.save()).

    Args
    ----
        filename : str
            The name of the file. If the string does not end in .hdf5, this
            will be added for you.
    """
    # define namedtuple as a go-between
    DummyEpoch = collections.namedtuple("DummyEpoch", field_names=["info"])

    if filename[-5:] != ".hdf5":
        filename = filename + ".hdf5"

    f = h5py.File(filename, "r")  # open file for reading

    ssvep = Ssvep(
        DummyEpoch(
            info=pickle.loads(f["info"].attrs["infostring"].tostring())
        ),
        f["stimulation"]["frequencies"].value,
        noisebandwidth=f["psdinfo"].attrs["noisebandwidth"],
        compute_harmonics=False,
        compute_subharmonics=False,
        compute_intermodulation=False,
        psd=f["psd"].value,
        freqs=f["freqs"].value,
        fmin=f["psdinfo"].attrs["fmin"],
        fmax=f["psdinfo"].attrs["fmax"],
    )

    # Load the evoked frequency structures and try to use them:
    for ftype in ["stimulation", "harmonic", "subharmonic", "intermodulation"]:
        ssvep.__setattr__(
            ftype,
            EvokedFrequency(
                **{key: value.value for key, value in f[ftype].items()}
            ),
        )
    # close the file:
    f.close()
    # and return the loaded data:
    return ssvep
