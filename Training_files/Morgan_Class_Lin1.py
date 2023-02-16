from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Morgan_Class_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'Morgan_fp_2_124',
        'solute_vectorizer': 'classification',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 30000}

Experiment(**kwargs)
