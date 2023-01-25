from my_nets.KRR_training import KRR_experiment
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

def what_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)

# solvents = ('blank', 'class', 'macro', 'morgan', 'morgan2to20', 'jb', 'bob', 'bat', 'soap')
# # solutes = ('blank', 'class', 'tesa', 'morgan', 'morgan2to20', 'jb', 'bob', 'bat', 'soap')
# solutes = ('computedprops',)
exclude = ('blankblank', 'morgan2to20morgan2to20', 'morgan2to20computedprops', )
# for solvent in (solvents):
#     for solute in (solutes):
#         if solvent+solute not in exclude:
#             what_time()
#             KRR_experiment(solvent, solute, number=2, n_jobs=None)

# solvents = ('blank', 'class', 'macro', 'morgan', 'morgan2to20', 'jb', 'bob', 'bat', 'soap')
solutes = ('blank', 'class', 'tesa', 'computedprops', 'morgan', 'morgan2to20', 'jb', 'bob', 'bat', 'soap')
solvents = ('morgan2to20',)
for solvent in (solvents):
    for solute in (solutes):
        if solvent+solute not in exclude:
            what_time()
            KRR_experiment(solvent, solute, number=2, n_jobs=None)

# KRR_experiment('bat', 'soap', number=1)
# BAT BAT
# BAT SOAP
# solutes = ('blank', 'class', 'tesa', 'morgan', 'jb', 'bob', 'bat', 'soap')