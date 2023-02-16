from my_nets.KRR_training import KRR_experiment
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

def what_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)

exclude = ('blankblank',)

solvents = ('blank', 'class', 'macro', 'macrox', 'morgan', 'morgan2to20', 'jb', 'bob', 'bat', 'soap')
solutes = ('blank', 'class', 'tesa', 'computedprops', 'morgan', 'morgan2to20', 'jb', 'bob',  'bat', 'soap')
for solvent in (solvents):
    for solute in (solutes):
        if solvent+solute not in exclude:
            what_time()
            KRR_experiment(solvent, solute, number=1, n_jobs=None)
