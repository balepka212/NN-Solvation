from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'BoB_TESA_Lin2',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'bag_of_bonds',
        'solute_vectorizer': 'solute_TESA',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
