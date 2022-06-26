from my_nets.Create_dataset import create_SS_table
from config import *

table = feather.read_feather(project_path('Tables/df3_3'))
SS_table = create_SS_table(table)
SS_table.to_csv(project_path('Tables/Entire_table3.csv'))