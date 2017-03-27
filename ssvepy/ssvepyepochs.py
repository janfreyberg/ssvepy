import mne
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class Ssvep(mne.Epochs):

    def __init__(self, epochs, stimulation_frequency,
                 noisebandwidth=1.0,
                 compute_harmonics=True, compute_subharmonics=False,
                 compute_intermodulation=True,
                 psd=None, freqs=None,
                 fmin=0.1, fmax=50, tmin=None, tmax=None):

        self.info = deepcopy(epochs.info)
        self.stimulation_frequency = stimulation_frequency
        self.noisebandwidth = noisebandwidth
        self.psd = psd
        self.freqs = freqs

        # Check if the right input was provided:
        if self.psd is not None and self.freqs is None:
            raise ValueError('If you provide psd data, you also need to provide'
                             ' the frequencies at which it was evaluated')

        # If no power-spectrum was provided, we need to work it out
        if self.psd is None:
            self.psd, self.freqs = mne.time_frequency.psd_multitaper(
                epochs, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)

        # Attribute: freq resolution
        self.frequency_resolution = self.freqs[1] - self.freqs[0]

        # If needed, work out the harmonic frequencies
        if compute_harmonics and type(compute_harmonics) is bool:
            self.harmonics = self._compute_harmonics(
                [harmonic for harmonic in range(1, 5)])
        elif compute_harmonics and type(compute_harmonics) is int:
            self.harmonics = self._compute_harmonics([compute_harmonics])
        elif compute_harmonics:
            self.harmonics = self._compute_harmonics(
                [harmonic for harmonic in compute_harmonics])
        else:
            self.harmonics = []
        # same for subharmonics
        if compute_subharmonics and type(compute_subharmonics) is bool:
            self.subharmonics = self._compute_harmonics(
                [1 / (harmonic) for harmonic in range(1, 5)])
        elif compute_subharmonics and type(compute_subharmonics) is int:
            self.subharmonics = self._compute_harmonics(
                [1 / compute_subharmonics])
        elif compute_subharmonics:
            self.subharmonics = self._compute_harmonics(
                [1 / (harmonic) for harmonic in compute_subharmonics])
        else:
            self.subharmonics = []

        # Work out the SNRs for the stimulation frequencies
        try:
            # try looping
            self.stimulation_snr = [self._compute_snr(freq)
                                    for freq in stimulation_frequency]
        except TypeError:
            # try just one frequency
            self.stimulation_snr = self._compute_snr(stimulation_frequency)

        # Work out the SNR for harmonics
        self.harmonic_snr = ([self._compute_snr(freq)
                              for freq in self.harmonics] if self.harmonics
                             else None)
        self.subharmonic_snr = ([self._compute_snr(freq)
                                 for freq in self.subharmonics]
                                if self.subharmonics
                                else None)

    def _compute_snr(self, freq):
        """
        Helper function to work out the SNR of a given frequency
        """
        stimband = ((self.freqs <= freq + self.frequency_resolution) &
                    (self.freqs >= freq - self.frequency_resolution))
        noiseband = ((self.freqs <= freq + self.noisebandwidth) &
                     (self.freqs >= freq - self.noisebandwidth) &
                     ~stimband)
        return (self.psd[..., stimband].mean(axis=-1) /
                self.psd[..., noiseband].mean(axis=-1))

    def _compute_harmonics(self, multipliers):
        """
        Helper function to compute the harmonics from a list, while making sure
        they're in the frequency range
        """
        return [self.stimulation_frequency * x for x in multipliers
                if (((self.stimulation_frequency / x) > self.freqs.min()) and
                    ((self.stimulation_frequency / x) < self.freqs.max()))]

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
        ydata = np.stack([self._compute_snr(freq)
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
        # If we didn't collapse over epochs, split the data
        if ydata.ndim <= 2:
            plt.plot(self.freqs, ydata, color='blue', alpha=0.3)
            if ydata.ndim > 1:
                plt.plot(self.freqs, ydata.mean(axis=1), color='red')
            for xval in [self.stimulation_frequency] + self.harmonics + \
                    self.subharmonics:
                plt.axvline(xval, linestyle='--', color='gray')
            plt.xticks([self.stimulation_frequency] +
                       self.harmonics + self.subharmonics)
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
                for xval in [self.stimulation_frequency] + self.harmonics + \
                        self.subharmonics:
                    plt.axvline(xval, linestyle='--', color='gray')
                plt.xticks([self.stimulation_frequency] +
                           self.harmonics + self.subharmonics)
                plt.title('Spectrum of epoch {n}'.format(n=idx + 1))

        plt.show()

    def __repr__(self):
        outstring = ('ssvepy data structure based on epoched data.\n'
                     'The stimulation frequency(s) is {stimfreq}.\n'
                     'There are {nepoch} epochs.\n The power spectrum was '
                     'evaluated over {nfreqs} frequencies ({fmin} Hz -'
                     '{fmax} Hz).\n'.format(stimfreq=self.stimulation_frequency,
                                            nepoch=self.psd.shape[0],
                                            nfreqs=self.freqs.size,
                                            fmin=self.freqs.min(),
                                            fmax=self.freqs.max()))
        return outstring

    # def plot_snr(self):
    #     plt.figure(figsize=(20, 8))
    #     plt.plot(self.freqs, self.psd.mean(axis=0).T, color='blue', alpha=0.3)
    #     plt.plot(self.freqs, )
    #     for xval in self.stimulation_frequency + self.harmonics +\
    #             self.subharmonics:
    #         plt.axvline(xval, linestyle='--', color='gray')
    #     plt.xticks(baselinefreqs)
    #     plt.show()
