from Solvation_1.my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'Morgan_2_124_Morgan_2_124_Lin1b',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'Morgan_fp_2_124',
        'solute_vectorizer': 'Morgan_fp_2_124',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'ckp_file':'Solvation_1/Runs/Morgan_2_124_Morgan_2_124_Lin1b/best/ep_9970_best.pt',
        'epochs': 10000}

Continue_experiment(**kwargs)
