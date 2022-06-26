from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'BAT_Class_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'BAT',
        'solute_vectorizer': 'classification',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 30000}

Experiment(**kwargs)
