from my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BoB_TESA_Res2',
        'net': 'Res',
        'lr': 1e-5,
        'solvent_vectorizer': 'bag_of_bonds',
        'solute_vectorizer': 'solute_TESA',
        'norm_bools': (True, True, True),
        'net_dict': {'base_filters': 2, 'kernel_size': 3, 'stride': 2, 'groups': 1, 'n_block': 3,
                     'n_classes': 1, 'use_bn': True, 'use_do': True, 'verbose': False},
        'ckp_file': 'Runs/BoB_TESA_Res2/ep_9999.pt',
        'epochs': 10000}

Continue_experiment(**kwargs)
