import os
import pandas as pd
from pyarrow import feather


def project_path(path):
    """TODO description"""
    project = os.path.dirname(os.path.abspath(__file__))
    final_path = project + '/../' + path
    # print(final_path)
    return final_path


def read_format(format):
    """TODO description"""
    if format == 'feather':
        return feather.read_feather
    elif format == 'tsv':
        return pd.read_table
    elif format == 'csv':
        return pd.read_csv
    elif format == 'txt':
        return pd.read_table
    else:
        print(f'No such format option {format}')
