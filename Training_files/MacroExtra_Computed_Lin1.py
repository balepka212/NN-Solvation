from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'MacroExtra_Computed_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'macroextra',
        'solute_vectorizer': 'computedprops',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000
        }

Experiment(**kwargs)


# 'blank'
# 'classification'
# 'solute_TESA'
# 'Morgan_fp_2_124'
# 'Morgan_2_1048576'
# 'just_bonds'
# 'bag_of_bonds'
# 'BAT'
# 'soap'
# computedprops