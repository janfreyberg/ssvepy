# ssvepy

A package to analyse MNE-formatted EEG data for steady-state visually evoked potentials (SSVEPs).

### Install:

`pip install git+https://github.com/janfreyberg/ssvepy.git`

As always with pip packages, you can install a "development" version of this package by (forking and) cloning the git repository and installing it via `pip install -e /path/to/package`. Please do open a pull request if you make improvements.

### Usage:

You should load, preprocess and epoch your data using [MNE](https://github.com/mne-tools/mne-python).

Take a look at a notebook that sets up an SSVEP analysis structure with the example data in this package:
https://github.com/janfreyberg/ssvepy/blob/master/example.ipynb

Once you have a data structure of the class `Epoch`, you can use `ssvepy.Ssvep(epoch_data, stimulation_frequency)`, where `stimulation_frequency` is the frequency (or list of frequencies) at which you stimulated your participants.

Other input parameters and their defaults are:
- The following parameters, which are equivalent to the parameters in `mne.time_frequency.psd_multitaper`:
  - `fmin=0.1`, the low end of the frequency range
  - `fmax=50`, the high end of the frequency range
  - `tmin=None`, the start time of the segment you want to analyse
  - `tmax=None`, the end time of the segment you want to analyse
- `noisebandwidth=1.0`, what bandwidth around a frequency should be used to calculate its signal-to-noise-ratio
-  Whether you want to compute the following nonlinearity frequencies:
  - `compute_harmonics=True`
  - `compute_subharmonics=False`
  - `compute_intermodulation=True` (NB: only when there's more than one input frequency)
- You can also provide your own Power-spectrum data, if you have worked it out using another method.
  - `psd=None` The powerspectrum. Needs to be a numpy array with dimensions: (epochs, channels, frequency)
  - `freqs=None` The frequencys at which the powerspectrum was evaluated. Needs to be a one-dimensional numpy array.

The resulting data has the following attributes:

- `stimulation`: a data structure with the following attributes:
  - `stimulation.frequencies`, `stimulation.power`, `stimulation.snr`
- `harmonics`, `subharmonics`, `intermodulations`: non-linear combination of your input stimulus frequencies, all with the attributes:
  - `_.frequencies`, `_.power`, `_.snr`, `_.order`
- `psd`: the Power-spectrum
- `freqs`: the frequencies at which the psd was evaluated

And the following methods:

- `plot_psd()`: Plot the power spectrum
- `plot_snr()`: Plot the SNR spectrum
- `save(filename)`: Saves an `hdf5` file that can be loaded with `ssvepy.load_ssvep(filename)` <sup>1</sup>

More to come.

<sup>1</sup>: This package currently uses hierarchical data files (hdf5) because it seems to lend itself to the different data stored in ssvep classes, but I know it's less than ideal to have different data structures from MNE. I'm still thinking about improvements.
