import os
import generate_state_evolution_data

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Get unitary by performing state vector evolution
for i in generate_state_evolution_data.__all__:
    item = getattr(generate_state_evolution_data, i)
    if callable(item):
        item()
