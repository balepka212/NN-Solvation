from Solvation_1.my_nets.Create_dataset import create_SS_table
from Solvation_1.config import *

table = feather.read_feather(project_path('Solvation_1/Tables/df3_3'))
SS_table = create_SS_table(table)
SS_table.to_csv(project_path('Solvation_1/Tables/Entire_table3.csv'))