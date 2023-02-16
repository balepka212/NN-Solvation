from my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BAT_Morgan_2_124_Res2',
        'net': 'Res',
        'lr': 1e-5,
        'solvent_vectorizer': 'BAT',
        'solute_vectorizer': 'Morgan_fp_2_124',
        'norm_bools': (True, True, True),
        'net_dict': {'base_filters': 2, 'kernel_size': 3, 'stride': 2, 'groups': 1, 'n_block': 3,
                     'n_classes': 1, 'use_bn': True, 'use_do': True, 'verbose': False},
        'ckp_file': 'Runs/BAT_Morgan_2_124_Res22/best/best_val_model.pt',
        'epochs': 20000}

Continue_experiment(**kwargs)
