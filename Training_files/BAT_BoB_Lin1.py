from my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BAT_BoB_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'BAT',
        'solute_vectorizer': 'bag_of_bonds',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'start_epoch': 0,
        'epochs': 10000}

Experiment(**kwargs)
