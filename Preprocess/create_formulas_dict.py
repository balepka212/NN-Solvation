from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import pickle as pkl
from config import project_path
from Vectorizers.vectorizers import get_dictionary
from Vectorizers.vectorizers import parse_formula


with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

S_formulas = {}
U_formulas = {}
for solvent in Solvents:
    smiles = get_dictionary('smiles')[solvent]
    mol = Chem.MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    dict_f = parse_formula(formula)
    S_formulas[solvent] = dict_f

for solute in Solutes:
    smiles = get_dictionary('smiles')[solute]
    mol = Chem.MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    dict_f = parse_formula(formula)
    U_formulas[solute] = dict_f

with open(project_path('Tables/Formulas.pkl'), 'wb') as f:
    pkl.dump((S_formulas, U_formulas), f)
