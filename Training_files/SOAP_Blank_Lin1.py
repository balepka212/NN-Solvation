from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'SOAP_Blank_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'soap',
        'solute_vectorizer': 'blank',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

Experiment(**kwargs)
