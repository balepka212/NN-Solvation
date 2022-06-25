from Solvation_1.my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'SOAP_JustBonds_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'soap',
        'solute_vectorizer': 'just_bonds',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
