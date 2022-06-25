from Solvation_1.my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BAT_Morgan_2_124_Lin2',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'BAT',
        'solute_vectorizer': 'Morgan_fp_2_124',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
