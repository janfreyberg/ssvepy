import mne
import ssvepy
import numpy as np
import os.path

length = 5
samplefreq = 256
nelectrodes = 10
stimfreq = 15
nepochs = 20

# data = np.random.rand(nepochs, nelectrodes, length * samplefreq) * \
#     (np.sin(np.tile(
#         np.linspace(0, 2 * length * stimfreq * np.pi, samplefreq * length), (nepochs, nelectrodes, 1))) / 20)
# info = mne.create_info(nelectrodes, samplefreq, ch_types='eeg', montage=None)
# epochs = mne.EpochsArray(data, info)
#
# dummyssvep = ssvepy.Ssvep(epochs, stimfreq)

rootpath, _ = os.path.split(__file__)
epochfile = os.path.join(rootpath, 'exampledata', 'example-epo.fif')
epoch_example = mne.read_epochs(epochfile)
