from Solvation_1.my_nets.KRR_training import KRR_experiment
import warnings
warnings.filterwarnings("ignore")


solvents = ('soap',)
solutes = ('morgan', 'jb', 'bob', 'bat', 'soap')
exclude = ('blankblank','classblank', 'classclass',)
for solvent in (solvents):
    for solute in (solutes):
        if solvent+solute not in exclude:
            KRR_experiment(solvent, solute, number=212, n_jobs=None)

# KRR_experiment('bat', 'soap', number=1)

# BAT BAT
# BAT SOAP
# solutes = ('blank', 'class', 'tesa', 'morgan', 'jb', 'bob', 'bat', 'soap')