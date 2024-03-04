# %%
import numpy as np

import scqubits as scq

__all__ = [
    'create_transmon', 'create_tunable_transmon', 'create_fluxonium',
    'create_composite_transmon', 'create_composite_fluxonium'
]


def create_transmon():
    """Create a Transmon Qubit in the charge basis by SCQbits."""
    ej_list = np.linspace(28, 32, 3)
    ec_list = np.linspace(1.0, 1.4, 3)
    ng_list = np.linspace(0.2, 0.4, 3)
    data = []
    for ej in ej_list:
        for ec in ec_list:
            for ng in ng_list:
                transmon = scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=31)
                data.append(transmon.eigenvals())
    data = np.asarray(data)
    np.save('data_create_transmon.npy', data)
    print('Create transmon data is finished.')


def create_tunable_transmon():
    """Create a Tunable Transmon Qubit in the charge basis by SCQbits."""
    ej_list = np.linspace(38, 42, 3)
    ec_list = np.linspace(0.1, 0.3, 3)
    d_list = np.linspace(0.05, 0.2, 3)
    flux_list = np.linspace(0.1, 0.5, 3)
    ng_list = np.linspace(0.2, 0.4, 3)
    data = []
    for ej in ej_list:
        for ec in ec_list:
            for d in d_list:
                for flux in flux_list:
                    for ng in ng_list:
                        tmon = scq.TunableTransmon(
                            EJmax=ej,
                            EC=ec,
                            d=d,
                            flux=flux,
                            ng=ng,
                            ncut=40
                        )
                        data.append(tmon.eigenvals())
    data = np.asarray(data)
    np.save('data_create_tunable_transmon.npy', data)
    print('Create tunable transmon data is finished.')


def create_fluxonium():
    """Create a Fluxonium Qubit in the harmonic-oscillator basis by SCQbits."""
    ej_list = np.linspace(7, 10, 3)
    ec_list = np.linspace(2.2, 2.7, 3)
    el_list = np.linspace(0.3, 0.7, 3)
    flux_list = np.linspace(0.1, 0.5, 3)
    data = []
    for ej in ej_list:
        for ec in ec_list:
            for el in el_list:
                for flux in flux_list:
                    fmon = scq.Fluxonium(EJ=ej,
                                         EC=ec,
                                         EL=el,
                                         flux=flux,
                                         cutoff=110)
                    data.append(fmon.eigenvals())
    data = np.asarray(data)
    np.save('data_create_fluxonium.npy', data)
    print('Create fluxonium data is finished.')


def create_composite_transmon():
    """Create a Hilbertspace with two transmons coupled to a harmonic mode."""
    tmon1 = scq.TunableTransmon(
        EJmax=40.0,
        EC=0.2,
        d=0.1,
        flux=0.23,
        ng=0.3,
        ncut=40,
        truncated_dim=3,  # after diagonalization, we will keep 3 levels
        id_str="tmon1"  # optional, used for referencing from within
        # ParameterSweep or InteractingSystems
    )

    tmon2 = scq.TunableTransmon(EJmax=15.0,
                                EC=0.15,
                                d=0.2,
                                flux=0.0,
                                ng=0.0,
                                ncut=30,
                                truncated_dim=3,
                                id_str="tmon2")

    resonator = scq.Oscillator(
        E_osc=4.5,
        truncated_dim=4  # up to 3 photons (0,1,2,3)
    )

    hilbertspace = scq.InteractingSystems([tmon1, tmon2, resonator])

    g1 = 0.1  # coupling resonator-CPB1 (without charge matrix elements)
    g2 = 0.2  # coupling resonator-CPB2 (without charge matrix elements)

    hilbertspace.add_interaction(
        g_strength=g1,
        op1=tmon1.n_operator,
        op2=resonator.creation_operator,
        add_hc=True,
        id_str="tmon1-resonator"  # optional keyword argument
    )

    hilbertspace.add_interaction(
        g_strength=g2,
        op1=tmon2.n_operator,
        op2=resonator.creation_operator,
        add_hc=True,
        id_str="tmon2-resonator"  # optional keyword argument
    )
    spec0 = hilbertspace.eigenvals(10)
    np.save('data_composite_transmon_spec.npy', spec0)
    hilbertspace.generate_lookup()
    bare_indices = [(i, j, k) for i in range(3) for j in range(3)
                    for k in range(4)]
    map0 = np.array(
        [hilbertspace.energy_by_bare_index(index) for index in bare_indices])
    np.save('data_composite_transmon_map.npy', map0)
    print('Create composite transmons data is finished.')


def create_composite_fluxonium():
    """Create a Hilbertspace with two fluxoniums coupled to a coupler."""
    fmon1 = scq.Fluxonium(
        EJ=3.5,
        EC=1.6,
        EL=0.5,
        flux=0.33,
        cutoff=110,
        truncated_dim=4,  # after diagonalization, we will keep 3 levels
        id_str="fmon1"  # optional, used for referencing from within
        # ParameterSweep or InteractingSystems
    )

    coupler1 = scq.Fluxonium(
        EJ=4.0,
        EC=2.0,
        EL=0.5,
        flux=0.0,
        cutoff=110,
        truncated_dim=6,
        id_str="coupler1"
    )

    fmon2 = scq.Fluxonium(
        EJ=3.5,
        EC=1.2,
        EL=0.5,
        flux=0.5,
        cutoff=110,
        truncated_dim=4,
        id_str="fmon2"
    )

    hilbertspace = scq.InteractingSystems([fmon1, coupler1, fmon2])

    jc = 0.04
    jl = -0.02

    hilbertspace.add_interaction(
        g_strength=jc,
        op1=fmon1.n_operator,
        op2=coupler1.n_operator,
        add_hc=False
    )

    hilbertspace.add_interaction(g_strength=jl,
                                 op1=fmon1.phi_operator,
                                 op2=coupler1.phi_operator,
                                 add_hc=False)

    hilbertspace.add_interaction(
        g_strength=jc,
        op1=fmon2.n_operator,
        op2=coupler1.n_operator,
        add_hc=False
    )

    hilbertspace.add_interaction(
        g_strength=jl,
        op1=fmon2.phi_operator,
        op2=coupler1.phi_operator,
        add_hc=False
    )
    spec0 = hilbertspace.eigenvals(10)
    np.save('data_composite_fluxonium_spec.npy', spec0)
    hilbertspace.generate_lookup()
    bare_indices = [(i, j, k) for i in range(4) for j in range(6)
                    for k in range(4)]
    map0 = np.array(
        [hilbertspace.energy_by_bare_index(index) for index in bare_indices])
    np.save('data_composite_fluxonium_map.npy', map0)
    print('Create composite fluxonium data is finished.')

# %%
