import mne
import numpy as np
from copy import deepcopy
import collections
import matplotlib.pyplot as plt
from . import frequencymaths

EvokedFrequency = collections.namedtuple('EvokedFrequency',
                                         ['frequencies', 'orders',
                                          'power', 'snr'])


class Ssvep(mne.Epochs):

    def __init__(self, epochs, stimulation_frequency,
                 noisebandwidth=1.0,
                 compute_harmonics=True, compute_subharmonics=False,
                 compute_intermodulation=True,
                 psd=None, freqs=None,
                 fmin=0.1, fmax=50, tmin=None, tmax=None):

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
            # if the user provided list of freqs(?), use this input a bit

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
                    [stimulation_frequency], dtype=float)
        # Use a custom named tuple for the frequency-related data
        self.stimulation = EvokedFrequency(
            frequencies=stimulation_frequency,
            orders=np.ones(stimulation_frequency.shape, dtype=float),
            power=self._get_amp(stimulation_frequency),
            snr=self._get_snr(stimulation_frequency)
        )
        harmfreqs, harmorder = self._compute_harmonics(compute_harmonics)
        self.harmonic = EvokedFrequency(
            frequencies=harmfreqs,
            orders=harmorder,
            power=self._get_amp(harmfreqs),
            snr=self._get_snr(harmfreqs)
        ) if compute_harmonics else EvokedFrequency(
            frequencies=None, orders=None, power=None, snr=None
        )
        subfreqs, suborder = self._compute_subharmonics(compute_subharmonics)
        self.subharmonic = EvokedFrequency(
            frequencies=subfreqs,
            orders=suborder,
            power=self._get_amp(subfreqs),
            snr=self._get_snr(subfreqs)
        ) if compute_subharmonics else EvokedFrequency(
            frequencies=None, orders=None, power=None, snr=None
        )
        if compute_intermodulation and stimulation_frequency.size > 1:
            interfreqs, interorder = self._compute_intermodulations(
                compute_intermodulation)
            self.intermodulation = EvokedFrequency(
                frequencies=interfreqs,
                orders=interorder,
                power=self._get_amp(interfreqs),
                snr=self._get_snr(interfreqs)
            )
        else:
            self.intermodulation = EvokedFrequency(
                frequencies=None, orders=None, power=None, snr=None
            )

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

    def plot_psd(self, collapse_epochs=True, collapse_electrodes=False,
                 **kwargs):
        """
        Plot the power-spectrum that has been calculated for this data.

        Parameters:
        collapse_epochs: True (default) or False
            Whether you want to plot the average of all epochs (default), or
            each power-spectrum individually.
        collapse_electrodes: True or False (default)
            Whether you want to plot each electrode individually (default), or
            only the average of all electrodes.
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
        collapse_epochs: True (default) or False
            Whether you want to plot the average of all epochs (default), or
            each power-spectrum individually.
        collapse_electrodes: True or False (default)
            Whether you want to plot each electrode individually (default), or
            only the average of all electrodes.
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

    def _plot_spectrum(self, ydata, figsize=(20, 8)):
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

        plt.show()

    def __repr__(self):
        outstring = ('ssvepy data structure based on epoched data.\n'
                     'The stimulation frequency(s) is {stimfreq}.\n'
                     'There are {nepoch} epochs.\n The power spectrum was '
                     'evaluated over {nfreqs} frequencies ({fmin} Hz -'
                     '{fmax} Hz).\n'.format(
                         stimfreq=self.stimulation.frequencies,
                         nepoch=self.psd.shape[0],
                         nfreqs=self.freqs.size,
                         fmin=self.freqs.min(),
                         fmax=self.freqs.max())
                     )
        return outstring
