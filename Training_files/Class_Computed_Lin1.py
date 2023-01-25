from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Class_Computed_Res1',
        'net': 'Res',
        'lr': 1e-5,
        'solvent_vectorizer': 'classification',
        'solute_vectorizer': 'computedprops',
        'norm_bools': (True, True, True),
        'net_dict': {'base_filters': 2, 'kernel_size': 3, 'stride': 2, 'groups': 1, 'n_block': 3,
                     'n_classes': 1, 'use_bn': True, 'use_do': True, 'verbose': False},
        'epochs': 10000
        }

Experiment(**kwargs)


# 'blank'
# 'classification'
# 'solute_TESA'
# solvent_macro_props1
# 'Morgan_fp_2_124'
# 'Morgan_2_1048576'
# 'just_bonds'
# 'bag_of_bonds'
# 'BAT'
# 'soap'
# computedprops