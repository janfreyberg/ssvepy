import mne
import numpy as np
from copy import deepcopy
import collections
import matplotlib.pyplot as plt
from . import frequencymaths
import h5py
import pickle
from scipy.signal import lfilter
from scipy.ndimage.filters import gaussian_filter
import sklearn.linear_model
import inspect

EvokedFrequency = collections.namedtuple('EvokedFrequency',
                                         ['frequencies',
                                          'orders',
                                          'power',
                                          'snr',
                                          'tfr'])


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
        psd : :class:`np:numpy.ndarray`
            If you have already computed the power spectrum with some
            other method, pass it as a parameter.
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

    **Attributes**

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

    def __init__(self, epochs, stimulation_frequency,
                 noisebandwidth=1.0,
                 compute_harmonics=range(2, 5),
                 compute_subharmonics=range(2, 5),
                 compute_intermodulation=range(2, 5),
                 psd=None, freqs=None,
                 fmin=0.1, fmax=50, tmin=None, tmax=None,
                 compute_tfr=False, tfr_method='rls', tfr_time_window=0.9):

        self.info = deepcopy(epochs.info)

        self.noisebandwidth = noisebandwidth
        self.psd = psd
        self.freqs = freqs
        self.fmin = fmin
        self.fmax = fmax

        # Check if the right input was provided:
        # TODO: Change into a _check_input method
        if self.psd is not None and self.freqs is None:
            raise ValueError('If you provide psd data, you also need to provide'
                             ' the frequencies at which it was evaluated')

        # If no power-spectrum was provided, we need to work it out
        if self.psd is None:
            # Use MNE here. TODO: offer different methods of FFT eval
            self.psd, self.freqs = mne.time_frequency.psd_multitaper(
                epochs, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)

        self.frequency_resolution = self.freqs[1] - self.freqs[0]

        if type(stimulation_frequency) is not np.ndarray:
            try:
                stimulation_frequency = np.array(
                    [x for x in stimulation_frequency], dtype=float)
            except:
                stimulation_frequency = np.array(
                    stimulation_frequency, dtype=float)
        # Use a custom named tuple for the frequency-related data
        self.stimulation = EvokedFrequency(
            frequencies=stimulation_frequency,
            orders=np.ones(stimulation_frequency.shape, dtype=float),
            power=self._get_amp(stimulation_frequency),
            snr=self._get_snr(stimulation_frequency),
            tfr=(self._compute_tfr(epochs, stimulation_frequency,
                                   window_width=tfr_time_window)
                 if compute_tfr else None)
        )

        harmfreqs, harmorder = self._compute_harmonics(compute_harmonics)
        self.harmonic = EvokedFrequency(
            frequencies=harmfreqs,
            orders=harmorder,
            power=self._get_amp(harmfreqs),
            snr=self._get_snr(harmfreqs),
            tfr=(self._compute_tfr(epochs, harmfreqs,
                                   window_width=tfr_time_window)
                 if compute_tfr else None)

        ) if compute_harmonics else EvokedFrequency(
            frequencies=None, orders=None, power=None, snr=None, tfr=None
        )
        subfreqs, suborder = self._compute_subharmonics(compute_subharmonics)
        self.subharmonic = EvokedFrequency(
            frequencies=subfreqs,
            orders=suborder,
            power=self._get_amp(subfreqs),
            snr=self._get_snr(subfreqs),
            tfr=(self._compute_tfr(epochs, subfreqs,
                                   window_width=tfr_time_window)
                 if compute_tfr else None)
        ) if compute_subharmonics else EvokedFrequency(
            frequencies=None, orders=None, power=None, snr=None, tfr=None
        )
        if compute_intermodulation and stimulation_frequency.size > 1:
            interfreqs, interorder = self._compute_intermodulations(
                compute_intermodulation)
            self.intermodulation = EvokedFrequency(
                frequencies=interfreqs,
                orders=interorder,
                power=self._get_amp(interfreqs),
                snr=self._get_snr(interfreqs),
                tfr=(self._compute_tfr(epochs, interfreqs,
                                       window_width=tfr_time_window)
                     if compute_tfr else None)
            )
        else:
            self.intermodulation = EvokedFrequency(
                frequencies=None, orders=None, power=None, snr=None, tfr=None
            )

    # Helper functions to get specific frequencies:

    def _get_snr(self, freqs):
        """
        Helper function to work out the SNR of a given frequency
        """
        snr = []
        for freq in freqs.flat:
            stimband = ((self.freqs <= freq + self.frequency_resolution) &
                        (self.freqs >= freq - self.frequency_resolution))
            noiseband = ((self.freqs <= freq + self.noisebandwidth) &
                         (self.freqs >= freq - self.noisebandwidth) &
                         ~stimband)
            snr.append(self.psd[..., stimband].mean(axis=-1) /
                       self.psd[..., noiseband].mean(axis=-1))
        snr = np.stack(snr, axis=-1) if len(snr) > 1 else snr[0]
        return snr

    def _get_amp(self, freqs):
        """
        Helper function to get the freq-smoothed amplitude of a frequency
        """
        return np.stack(
            [self.psd[...,
                      ((self.freqs <= freq + self.frequency_resolution) &
                       (self.freqs >= freq - self.frequency_resolution))
                      ].mean(axis=-1)
             for freq in freqs.flat],
            axis=-1
        )

    def _compute_tfr(self, epoch, freq, tfr_method='rls', window_width=1.2,
                     filter_lambda=1.0):
        """
        Work out the time-frequency composition of different frequencies.
        """

        data = epoch.get_data()

        if type(freq) is not np.ndarray:
            raise TypeError('Frequencies need to provided in a numpy array.')

        samplefreq = epoch.info['sfreq']
        n_window = int(samplefreq * window_width)
        if filter_lambda == 1:
            lambdafilter = (np.ones(n_window) /
                            (n_window / 2))
        else:
            lambdafilter = np.power(filter_lambda,
                                    np.arange(n_window))

        t = np.arange(data.shape[-1]) / samplefreq

        # create a data structure that matches MNE standard TFR shape
        tfr_data = np.zeros((data.shape[0], data.shape[1],
                             freq.size, data.shape[2]))

        if tfr_method == 'rls':
            # this implementation follows Sucharit Katyal's code
            for fi, f in enumerate(freq.flatten()):
                s = -np.sin(2 * np.pi * f * t)
                c = np.cos(2 * np.pi * f * t)
                # Create the sin and cosine
                for trial in range(data.shape[0]):
                    for electrode in range(data.shape[1]):
                        y = data[trial, electrode, :]
                        # obtain cosine and since components
                        hc = lfilter(lambdafilter, 1, y * c)
                        hs = lfilter(lambdafilter, 1, y * s)
                        # lambda correction, if necessary
                        if filter_lambda < 1:
                            hc = hc / lfilter(lambdafilter, 1, c**2)
                            hs = hs / lfilter(lambdafilter, 1, s**2)
                        # combine the data to get envelope
                        a = np.abs(hc + 1j * hs)
                        # shift left, pad zero
                        a = np.roll(a, -n_window // 2)
                        a[(-n_window // 2):-1] = np.nan
                        # smooth with gaussian
                        a[0:(-n_window // 2)] = gaussian_filter(
                            a[0:(-n_window // 2)], n_window // 10
                        )
                        # set in tfr_data
                        tfr_data[trial, electrode, fi, :] = a
        else:
            raise NotImplementedError('Only RLS is available so far.')
        return tfr_data

    # Helper functions to compute non-linear frequency combos:

    def _compute_harmonics(self, orders):
        """
        Wrapper function around the compute_harmonics function in frequencymaths
        """
        return frequencymaths.compute_harmonics(
            self.stimulation.frequencies,
            fmin=self.fmin, fmax=self.fmax, orders=orders
        )

    def _compute_subharmonics(self, orders):
        """
        Helper function to compute the subharms from a list, while making sure
        they're in the frequency range
        """
        return frequencymaths.compute_subharmonics(
            self.stimulation.frequencies,
            fmin=self.fmin, fmax=self.fmax, orders=orders
        )

    def _compute_intermodulations(self, orders):
        """
        Helper function to compute the intermods from a list, while making sure
        they're in the frequency range
        """
        return frequencymaths.compute_intermodulation(
            self.stimulation.frequencies,
            fmin=self.fmin, fmax=self.fmax, orders=orders
        )

    # Machine learning routines:

    def predict_timepoints(
            self, labels, trainingtrials=None, datatransform=None,
            method=sklearn.linear_model.LogisticRegressionCV()):
        """
        This method is for predicting the labels of trials based on the SSVEP
        power in a trial.

        Args
        ----
            labels : :class:`np:numpy.ndarray`
                Labels for each timepoint; the dimensions should match
                (trial x timepoint)
            trainepochs : list
                A list of trial indices that will be used to train - e.g. range(6) - or an ndarray of booleans of the same size as label. The trials in this index will be used as training trials, and the returned accuracy is based on the trials *not* used for training.
            datatransform : str
                What transform to do to the TFR data - at the moment, 'z-score' works.
            method : sklearn.class
                A training class that conforms to the standard sklearn model (ie has the methods fit(), predict() etc.)
        """
        pass

    def predict_epochs(self, labels, trainepochs=None,
                       method=sklearn.linear_model.LogisticRegressionCV()):
        """
        This method is for predicting the labels of trials based on the SSVEP
        power in a trial.

        Args
        ----
            labels : :class:`np:numpy.ndarray`
                Labels for each trial; needs to be the same length as there are epochs.
            trainepochs : list
                A list of trial indices that will be used to train - e.g. range(6) - or an ndarray of booleans of the same size as label. The trials in this index will be used as training trials, and the returned accuracy is based on the trials *not* used for training.
            method : sklearn.class
                A training class that conforms to the standard sklearn model (ie has the methods fit(), predict() etc.)
        """

        # check the method as a fit method
        assert inspect.ismethod(method.fit), \
            ('The provided method has no fit function')
        trainclass = method

        # check that the labels match up with the trial numbers
        labels = np.array(labels)  # make sure it's an array
        assert labels.size == len(self), \
            'Number of labels needs to match number of epochs.'

        # reshape the training data and concatenate them all
        trainx = self.stimulation.snr.reshape((len(self), -1))
        for freqtype in ('harmonic', 'subharmonic', 'intermodulation'):
            # concatenate:
            freqs = self.__getattribute__(freqtype).frequencies.flatten()
            trainx = np.concatenate(
                (trainx,
                 self.__getattribute__(freqtype).snr[
                     ..., ~np.isnan(freqs)
                 ].reshape((len(self), -1))),
                axis=-1
            )
        # trainx = np.concatenate(
        #     [self.__getattribute__(freqtype).snr.reshape((len(self), -1))
        #      for freqtype in
        #      ('stimulation', 'harmonic', 'subharmonic', 'intermodulation')
        #      if self.__getattribute__(freqtype).snr is not None],
        #     axis=-1
        # )

        # if the training trials
        if trainepochs is not None:
            trainepochs = np.array(trainepochs)
            if trainepochs.size == len(self):
                # probably a boolean
                testepochs = ~trainepochs
            else:
                # probably integer index
                testepochs = np.array([i for i in range(len(self))
                                       if i not in trainepochs])
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

    def plot_tfr(self, frequency='stimulation', collapse_epochs=True,
                 collapse_electrodes=False,
                 figsize=(7, 5)):
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
            figsize : tup
                Matplotlib figure size.
        """

        if frequency is None or frequency == 'stimulation':
            y = self.stimulation.tfr
            z = self.stimulation.frequencies
        elif type(frequency) is str:
            y = self.__getattribute__(frequency).tfr
            z = self.__getattribute__(frequency).frequencies

        x = np.arange(y.shape[-1]) / self.info['sfreq']

        collapse_axes = tuple(
            [ax for ax, b in enumerate([collapse_epochs,
                                        collapse_electrodes])
             if b]
        )
        if len(collapse_axes) > 0:
            y = y.mean(axis=collapse_axes)
        # Make time the first dimension
        y = np.rollaxis(y, -1)
        # Make a figure (-1 is now freq. dimension)
        nplots = y.shape[-1]
        nrows = int(np.ceil(np.sqrt(nplots)))
        ncols = int(np.ceil(nplots / nrows))
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=figsize)
        # y = np.squeeze(y)
        for idx in range(nplots):
            # Choose axes to plot in
            ax = axes.flatten()[idx] if nplots > 1 else axes
            # Plot the individual lines
            ax.plot(x, y[..., idx], color='blue', alpha=0.1)
            # Plot the mean of the data
            if y[..., idx].size > y.shape[0]:
                ax.plot(x, y[..., idx].mean(axis=-1))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(str(z.flatten()[idx])
                         + ' Hz')

        plt.show()

    def plot_psd(self, collapse_epochs=True, collapse_electrodes=False,
                 **kwargs):
        """
        Plot the power-spectrum that has been calculated for this data.

        Parameters
        ----------
            collapse_epochs : bool
                Whether you want to plot the average of all epochs (default),
                or each power-spectrum individually.
            collapse_electrodes : bool
                Whether you want to plot each electrode individually
                (default), or only the average of all electrodes.

        """
        ydata = self.psd
        # Average over axes if necessary
        ydata = ydata.mean(axis=tuple([x for x in range(2)
                                       if [collapse_epochs,
                                           collapse_electrodes][x]]))

        self._plot_spectrum(ydata, **kwargs)

    def plot_snr(self, collapse_epochs=True, collapse_electrodes=False,
                 **kwargs):
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
        ydata = np.stack([self._get_snr(freq)
                          for idx, freq in enumerate(self.freqs)],
                         axis=-1)
        # Average over axes if necessary
        ydata = ydata.mean(axis=tuple([x for x in range(2)
                                       if [collapse_epochs,
                                           collapse_electrodes][x]]))
        self._plot_spectrum(ydata, **kwargs)

    def _plot_spectrum(self, ydata, figsize=(15, 7), show=True):
        """
        Helper function to plot different spectra
        """
        # Make sure frequency data is the first index
        ydata = np.transpose(
            ydata, axes=([ydata.shape.index(self.freqs.size)] +
                         [dim for dim, _ in enumerate(ydata.shape)
                          if dim != ydata.shape.index(self.freqs.size)])
        )
        # Start figure
        plt.figure(figsize=figsize)
        xmarks = np.concatenate([a.flatten() for a in
                                 [self.stimulation.frequencies,
                                  self.harmonic.frequencies,
                                  self.subharmonic.frequencies,
                                  self.intermodulation.frequencies]
                                 if np.any(a)]).tolist()
        # If we didn't collapse over epochs, split the data
        if ydata.ndim <= 2:
            plt.plot(self.freqs, ydata, color='blue', alpha=0.3)
            if ydata.ndim > 1:
                plt.plot(self.freqs, ydata.mean(axis=1), color='red')
            for xval in xmarks:
                plt.axvline(xval, linestyle='--', color='gray')
            plt.xticks(xmarks)
            plt.title('Average spectrum of all epochs')
        elif ydata.ndim > 2:
            ydatas = [ydata[:, idx, :] for idx in range(ydata.shape[1])]
            for idx, ydata in enumerate(ydatas):
                plt.subplot(np.ceil(np.sqrt(len(ydatas))),
                            np.ceil(len(ydatas) /
                                    np.ceil(np.sqrt(len(ydatas)))),
                            idx + 1)
                plt.plot(self.freqs, ydata, color='blue', alpha=0.3)
                if ydata.ndim > 1:
                    plt.plot(self.freqs, ydata.mean(axis=1), color='red')
                for xval in xmarks:
                    plt.axvline(xval, linestyle='--', color='gray')
                plt.xticks(xmarks)
                plt.title('Spectrum of epoch {n}'.format(n=idx + 1))

        if show:
            plt.show()

    def topoplot_psd(self, collapse_epochs=True,
                     flims='stimulation', **kwargs):
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

        # Get the actual data, collapse across frequency band
        if len(fmins.flatten()) > 1:
            topodata = [self.psd[...,
                                 (self.freqs > fmin) &
                                 (self.freqs < fmax)].mean(axis=-1)
                        for fmin, fmax in
                        zip(fmins.flatten(), fmaxs.flatten())]
            annot = [(str(fmin) + ' - ' + str(fmax) + ' Hz')
                     for fmin, fmax in zip(fmins.flatten(), fmaxs.flatten())]
        else:
            topodata = self.psd[...,
                                (self.freqs > fmins) &
                                (self.freqs < fmaxs)].mean(axis=-1)
            # annotation
            annot = (str(fmins[0]) + ' - ' + str(fmaxs[0]) + ' Hz')
        # also collapse epochs
        if collapse_epochs:
            topodata = topodata.mean(axis=0)
        # Call the actual plotting function
        self._plot_topo(topodata, annot, **kwargs)
        plt.suptitle('Power')
        plt.show()

    def topoplot_snr(self, collapse_epochs=True,
                     flims='stimulation', **kwargs):
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

        if len(fmins.flatten()) > 1:
            topodata = [np.stack([self._get_snr(freq)
                                  for freq in self.freqs
                                  if freq > fmin and freq < fmax],
                                 axis=-1).mean(-1)
                        for fmin, fmax in zip(fmins.flatten(), fmaxs.flatten())]
            annot = [(str(fmin) + ' - ' + str(fmax) + ' Hz')
                     for fmin, fmax in zip(fmins.flatten(), fmaxs.flatten())]
            if collapse_epochs:
                topodata = [t.mean(axis=0) for t in topodata]
        else:
            topodata = np.stack([self._get_snr(freq)
                                 for freq in self.freqs
                                 if freq > fmins and freq < fmaxs],
                                axis=-1).mean(-1)
            # annotation
            annot = (str(fmins[0]) + ' - ' + str(fmaxs[0]) + ' Hz')
            if collapse_epochs:
                topodata = topodata.mean(axis=0)

        self._plot_topo(topodata, annot, **kwargs)
        plt.gca()
        plt.suptitle('SNR', size='xx-large')
        # plt.figtext(0.05, 0.05, annot,
        #             size='small')
        plt.show()

    def _get_flims(self, flims):
        """
        Helper function that turns strings or lists into helpful
        """
        if flims is not None:
            if type(flims) is str:
                fmins = self.__getattribute__(flims).frequencies - 0.1
                fmaxs = self.__getattribute__(flims).frequencies + 0.1
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

    def _plot_topo(self, topodata, annotation=None,
                   figsize=(5, 5), cmap='Blues',
                   channels=None, vmin=None, vmax=None, ax=None):
        """
        Helper function to plot scalp distribution
        """
        # Get the montage
        pos = mne.channels.layout._auto_topomap_coords(
            self.info, range(len(self.ch_names))
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
        fig, axes = plt.subplots(nrows=int(np.ceil(np.sqrt(len(topodata)))),
                                 ncols=int(np.ceil(len(topodata) /
                                                   np.ceil(np.sqrt(len(topodata))))),
                                 figsize=figsize)
        for idx, t in enumerate(topodata):
            t = t.flatten()
            # Check for infs or nans and remove them
            bads = np.isnan(t) | np.isinf(t)
            t = t[~bads]
            pos = pos[~bads, :]
            # pick correct axes
            ax = axes.flatten()[idx] if type(axes) is np.ndarray else axes
            im, _ = mne.viz.plot_topomap(t, pos, cmap=cmap,
                                         vmin=vmin, vmax=vmax,
                                         show=False, axes=ax)
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
        return fig

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
        if filename[-5:] != '.hdf5':
            filename = filename + '.hdf5'
        f = h5py.File(filename, 'w')  # open a file
        # First the necessary data:
        psd = f.create_dataset('psd', data=self.psd)
        freqs = f.create_dataset('freqs', data=self.freqs)
        # whack in the info structure:
        info = f.create_group('info')
        info.attrs['infostring'] = np.void(pickle.dumps(self.info))
        # Store the various evoked frequency structures:
        for name, tup in [('stimulation', self.stimulation),
                          ('harmonic', self.harmonic),
                          ('subharmonic', self.subharmonic),
                          ('intermodulation', self.intermodulation)]:
            g = f.create_group(name)
            for key, value in tup._asdict().items():
                if value is not None:
                    g.create_dataset(key, data=value)
        # Store the frequency resolution
        psdinfo = f.create_group('psdinfo')
        psdinfo.attrs['frequency_resolution'] = self.frequency_resolution
        psdinfo.attrs['fmin'] = self.fmin
        psdinfo.attrs['fmax'] = self.fmax
        psdinfo.attrs['noisebandwidth'] = self.noisebandwidth

        # Finally, close the file:
        f.close()

    def __repr__(self):
        outstring = ('ssvepy data structure based on epoched data.\n'
                     'The stimulation frequency(s) is {stimfreq}.\n'
                     'There are {nepoch} epochs.\n The power spectrum was '
                     'evaluated over {nfreqs} frequencies ({fmin} Hz - '
                     '{fmax} Hz).\n'.format(
                         stimfreq=self.stimulation.frequencies,
                         nepoch=self.psd.shape[0],
                         nfreqs=self.freqs.size,
                         fmin=self.freqs.min(),
                         fmax=self.freqs.max())
                     )
        return outstring

    def __len__(self):
        return self.psd.shape[0]

    def __getitem__(self, key):
        return self.psd[key, ...]

    def __iter__(self):
        for idx in range(len(self)):
            yield(self.psd[idx, ...])


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
    DummyEpoch = collections.namedtuple('DummyEpoch', field_names=['info'])

    if filename[-5:] != '.hdf5':
        filename = filename + '.hdf5'

    f = h5py.File(filename, 'r')  # open file for reading

    ssvep = Ssvep(
        DummyEpoch(info=pickle.loads(f['info'].attrs['infostring'].tostring())),
        f['stimulation']['frequencies'].value,
        noisebandwidth=f['psdinfo'].attrs['noisebandwidth'],
        compute_harmonics=False,
        compute_subharmonics=False,
        compute_intermodulation=False,
        psd=f['psd'].value,
        freqs=f['freqs'].value,
        fmin=f['psdinfo'].attrs['fmin'],
        fmax=f['psdinfo'].attrs['fmax']
    )

    # Load the evoked frequency structures and try to use them:
    for ftype in ['stimulation', 'harmonic', 'subharmonic', 'intermodulation']:
        try:
            ssvep.__setattr__(
                ftype,
                EvokedFrequency(**{key: value.value
                                   for key, value in f[ftype].items()})
            )
        except:
            pass
    # close the file:
    f.close()
    # and return the loaded data:
    return ssvep
