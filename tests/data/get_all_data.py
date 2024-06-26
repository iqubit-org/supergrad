import os
import get_data_scqubits
import get_data_qutip

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


# Get data from scqubits
for i in get_data_scqubits.__all__:
    item = getattr(get_data_scqubits, i)
    if callable(item):
        item()

# Get data from qutip
for i in get_data_qutip.__all__:
    item = getattr(get_data_qutip, i)
    if callable(item):
        item()
