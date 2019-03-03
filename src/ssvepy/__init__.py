import pkg_resources

from .ssvepyepochs import Ssvep, load_ssvep
from . import frequencymaths

__version__ = pkg_resources.get_distribution("ssvepy").version
