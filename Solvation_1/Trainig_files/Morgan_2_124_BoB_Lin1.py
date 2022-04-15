from Solvation_1.my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Morgan_BoB_Lin2',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'Morgan_fp_2_124',
        'solute_vectorizer': 'bag_of_bonds',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
