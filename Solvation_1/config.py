import os
import pandas as pd
from pyarrow import feather


def project_path(path):
    """
    Returns a full path from a directory path

    Parameters
    ----------
    path: str
        Path of file in the project (starts with 'Solvation_1/')
    """
    project = os.path.dirname(os.path.abspath(__file__))
    final_path = project + '/../' + path
    # print(final_path)
    return final_path


def read_format(format):
    """
    Returns a function to read table from give format

    Parameters
    ----------
    format: str
        format of the import file. Available: feather, tsv, csv or txt
    """
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
