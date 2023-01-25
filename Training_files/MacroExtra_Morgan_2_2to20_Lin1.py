from my_nets.Experiment import Experiment

kwargs = {
        'runs_folder': 'MacroExtra_Morgan_2_2to20_Lin1',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'MacroExtra',
        'solute_vectorizer': 'Morgan_2_1048576',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'epochs': 10000}

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