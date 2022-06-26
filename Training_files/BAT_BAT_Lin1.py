from my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BAT_BAT_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'BAT',
        'solute_vectorizer': 'BAT',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'start_epoch': 0,
        'epochs': 20000}

Experiment(**kwargs)
