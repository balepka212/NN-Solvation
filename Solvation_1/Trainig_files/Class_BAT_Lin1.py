from Solvation_1.my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Class_BoB_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'classification',
        'solute_vectorizer': 'bag_of_bonds',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 20000}

Experiment(**kwargs)
