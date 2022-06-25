from Solvation_1.my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'BoB_SOAP_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'bag_of_bonds',
        'solute_vectorizer': 'soap',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
