import pandas as pd
import io
import zipfile

def load_data(filename):
    columns = ['unit_number', 'time_cycles']
    columns += [f'op_setting_{i+1}' for i in range(3)]
    columns += [f'sensor_{i+1}' for i in range(21)]
    data = pd.read_csv(filename, sep=' ', header=None, names=columns)
    data = data.dropna(axis=1, how='all')
    return data

def extract_zip(uploaded_files):
    for fn in uploaded_files.keys():
        if fn.endswith('.zip'):
            zip_ref = zipfile.ZipFile(io.BytesIO(uploaded_files[fn]), 'r')
            zip_ref.extractall('.')
            zip_ref.close()