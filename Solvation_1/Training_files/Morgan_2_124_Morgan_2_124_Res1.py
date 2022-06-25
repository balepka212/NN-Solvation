from Solvation_1.my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Morgan_2_124_Morgan_2_124_Res1',
        'net': 'Res',
        'lr': 1e-5,
        'solvent_vectorizer': 'Morgan_fp_2_124',
        'solute_vectorizer': 'Morgan_fp_2_124',
        'norm_bools': (True, True, True),
        'net_dict': {'base_filters': 2, 'kernel_size': 3, 'stride': 2, 'groups': 1, 'n_block': 3,
                     'n_classes': 1, 'use_bn': True, 'use_do': True, 'verbose': False},
        'epochs': 10000
        }

Experiment(**kwargs)
