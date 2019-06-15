import pkg_resources

from .ssvepyepochs import Ssvep, load_ssvep  # noqa: F401
from . import frequencymaths  # noqa: F401

__version__ = pkg_resources.get_distribution("ssvepy").version
