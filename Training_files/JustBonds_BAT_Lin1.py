from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'JustBonds_BAT_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'just_bonds',
        'solute_vectorizer': 'BAT',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 20000}

Experiment(**kwargs)
