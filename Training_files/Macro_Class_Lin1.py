from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Macro_Class_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'solvent_macro_props1',
        'solute_vectorizer': 'classification',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 20000}

Experiment(**kwargs)
