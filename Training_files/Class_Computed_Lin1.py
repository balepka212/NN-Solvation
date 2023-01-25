from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'Class_Computed_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'classification',
        'solute_vectorizer': 'computedprops',
        'norm_bools': (True, True, True),
        'net_dict': None,
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