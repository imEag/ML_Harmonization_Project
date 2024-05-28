import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def process_data(path, neuro, space, group, A, B, ratio):
    if ratio == 79:
        str_ratio = '2to1'
    elif ratio == 31:
        str_ratio = '5to1'
    elif ratio == 15:
        str_ratio = '10to1'
    else:
        str_ratio = str(ratio)

    base_dir = os.path.join(path, neuro, f'complete{str_ratio}', space, A if group else B)
    
    # Verificar y mostrar la construcción del directorio base
    print(f'Base directory: {base_dir}')
    if not os.path.exists(base_dir):
        print(f'Directory does not exist: {base_dir}')
        return
    
    data_frames = []
    for metric in ['power', 'sl', 'cohfreq', 'entropy', 'crossfreq']:
        # Corregir el formato del nombre del archivo
        file_path = os.path.join(base_dir, f'Data_complete_{space}_{neuro}_{A if group else B}_{metric}.feather')
        # Verificar y mostrar la construcción de la ruta del archivo
        print(f'Constructed file path: {file_path}')
        if os.path.exists(file_path):
            print(f'Reading {file_path}')
            data_frames.append(pd.read_feather(file_path))
        else:
            print(f'File not found: {file_path}')
    
    if data_frames:
        data = pd.concat(data_frames, axis=1)
        data = data.loc[:, ~data.columns.duplicated()]
        
        new_name = f'Data_integration_{space}_{neuro}_{A if group else B}'
        path_integration = os.path.join(path, neuro, f'integration{str_ratio}', space, A if group else '')
        os.makedirs(path_integration, exist_ok=True)
        
        output_file = os.path.join(path_integration, f'{new_name}.feather')
        print(f'Saving {output_file}')
        data.reset_index(drop=True).to_feather(output_file)
    else:
        print(f'No data files found for {neuro}, {space}, ratio {ratio}')

# Define paths and parameters
path = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\dataframes'
A = 'G1'
B = ''
s = ['ic']
h = ['neuroHarmonize', 'sovaharmony']
group = 1  # Ajustado a 1 ya que A está definido

# Process data
for space in s:
    for neuro in h:
        process_data(path, neuro, space, group, A, B, 79)
        process_data(path, neuro, space, group, A, B, 31)
        process_data(path, neuro, space, group, A, B, 15)




