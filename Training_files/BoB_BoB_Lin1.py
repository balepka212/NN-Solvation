from my_nets.Experiment import Experiment, Continue_experiment

kwargs = {
        'runs_folder': 'BoB_BoB_Lin2',
        'net': 'Lin',
        'lr': 1e-5,
        'solvent_vectorizer': 'bag_of_bonds',
        'solute_vectorizer': 'bag_of_bonds',
        'norm_bools': (True, True, True),
        'net_dict': None,
        'ckp_file': 'Runs/BoB_BoB_Lin2/best/best_val_model.pt',
        'start_epoch': 0,
        'epochs': 10000}

Continue_experiment(**kwargs)
