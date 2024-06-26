from supergrad.version import __version__ as __version__

from jax import config

config.update("jax_enable_x64", True)
# enable fp64 for better numerical stability

from supergrad.utils import identity_wrap as identity_wrap
from supergrad.utils import tensor as tensor
from supergrad.utils import basis as basis
from supergrad.utils import permute as permute

from supergrad.quantum_system import KronObj as KronObj
from supergrad.quantum_system import LindbladObj as LindbladObj

from supergrad.helper import Helper as Helper
from supergrad.helper import Spectrum as Spectrum
from supergrad.helper import Evolve as Evolve

from supergrad.time_evolution import sesolve as sesolve
from supergrad.time_evolution import mesolve as mesolve
