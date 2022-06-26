from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Blank_TESA_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'blank',
        'solute_vectorizer': 'solute_TESA',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 20000}

Experiment(**kwargs)
