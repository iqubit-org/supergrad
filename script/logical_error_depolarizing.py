# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# Find the path of the New Times font file
font_path = font_manager.findfont(
    font_manager.FontProperties(family='Times New Roman'))

# Set the font properties
font_props = font_manager.FontProperties(fname=font_path)

# Use the font properties in Matplotlib
plt.rcParams['font.family'] = font_props.get_name()
plt.rcParams.update({'font.size': 16.5})
fig = plt.figure(figsize=(8, 5.5))
# %%
# number of samples: 100000
num_sample = 100000

depolarizing_p = np.linspace(0, 5e-5, 21)
er_w_7 = np.array([[0.00101], [0.00209], [0.00401], [0.00752], [0.01227],
                   [0.01866], [0.02788], [0.0389], [0.05242], [0.06743],
                   [0.08468], [0.10317], [0.12368], [0.14469], [0.1651],
                   [0.18733], [0.20984], [0.23348], [0.25258], [0.2727],
                   [0.29339]])
er_wo_7 = np.array([[0.00023], [0.00064], [0.00172], [0.00385], [0.00731],
                    [0.01279], [0.02059], [0.03058], [0.04196], [0.05575],
                    [0.07209], [0.08976], [0.10862], [0.1295], [0.15168],
                    [0.17178], [0.19475], [0.21601], [0.2386], [0.25981],
                    [0.28173]])
er_w_13 = np.array([[3e-05], [0.00014], [0.00054], [0.00148], [0.00362],
                    [0.00861], [0.01706], [0.03069], [0.05124], [0.07773],
                    [0.11131], [0.14976], [0.19475], [0.23886], [0.28148],
                    [0.32377], [0.36093], [0.39078], [0.41821], [0.43974],
                    [0.45622]])
std_er_w_13 = np.sqrt(er_w_13 * (1 - er_w_13) / num_sample)
er_wo_13 = np.array([[0.0], [1e-05], [1e-05], [0.00028], [0.00101], [0.00299],
                     [0.00731], [0.01616], [0.03054], [0.05119], [0.07808],
                     [0.11152], [0.15173], [0.19594], [0.24221], [0.2871],
                     [0.32772], [0.36488], [0.39826], [0.42278], [0.44371]])
std_er_wo_13 = np.sqrt(er_wo_13 * (1 - er_wo_13) / num_sample)
# %%
plt.fill_between(depolarizing_p, (np.maximum(er_w_13 - 3 * std_er_w_13,
                                             1e-6)).squeeze(),
                 (er_w_13 + 3 * std_er_w_13).squeeze(),
                 color="lightblue")
plt.plot(depolarizing_p, (er_w_13), 'bo', label='d=13 w high-weight')
plt.plot(depolarizing_p, (er_wo_13), 'bv', label='d=13 wo high-weight')
plt.plot(depolarizing_p, (er_w_7), 'yo', label='d=7 w high-weight')
plt.plot(depolarizing_p, (er_wo_7), 'yv', label='d=7 wo high-weight')
plt.xlabel('Depolarizing rate $r$ [GHz]')
plt.ylabel('Logical error rate $p_\mathrm{logical}$')
plt.yscale('log')
plt.ylim((1e-4, 1e-0))
plt.legend()
plt.savefig('logical_error_vs_decoherence_before_opt.pdf')
plt.show()
# %%
