from my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BAT_TESA_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'BAT',
        'solute_vectorizer': 'solute_TESA',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
