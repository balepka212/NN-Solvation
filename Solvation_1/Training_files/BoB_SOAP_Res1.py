from Solvation_1.my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BoB_SOAP_Res1',
        'net': 'Res',
        'lr': 1e-5,
        'solvent_vectorizer': 'bag_of_bonds',
        'solute_vectorizer': 'soap',
        'norm_bools': (True, True, True),
        'net_dict': {'base_filters': 2, 'kernel_size': 3, 'stride': 2, 'groups': 1, 'n_block': 3,
                     'n_classes': 1, 'use_bn': True, 'use_do': True, 'verbose': False},
        'epochs': 10000}

Experiment(**kwargs)
