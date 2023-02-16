from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Morgan_2_2to20_Morgan_2_2to20_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'Morgan_2_1048576',
        'solute_vectorizer': 'Morgan_2_1048576',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
