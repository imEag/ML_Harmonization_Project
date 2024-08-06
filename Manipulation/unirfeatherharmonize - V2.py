"""
Importing Libraries
os: library for interacting with the operating system and manipulating directories.
pandas: library for data manipulation and analysis.
warnings: library for handling warnings.
process_data Function
This function takes seven parameters:
path: base directory path.
neuro: name of the neuroharmonization directory.
space: name of the space directory.
group: boolean indicating whether to use group A or B.
A and B: names of groups A and B.
ratio: numerical ratio used to determine the directory name.
The function performs the following tasks:
Determines the base directory name based on the ratio.
Checks if the base directory exists, and if not, prints an error message.
Reads Feather files from the base directory and combines them into a single DataFrame.
Removes duplicate columns from the combined DataFrame.
Creates a new directory for data integration and saves the combined DataFrame to a Feather file.
Parameter Definition
path: base directory path.
A and B: names of groups A and B.
s: list of space directory names.
h: list of neuroharmonization directory names.
group: boolean indicating whether to use group A or B.
Data Processing
The code iterates over the s and h lists and calls the process_data function for each parameter combination.
Three different ratio values are processed: 79, 31, and 15.
Notes
The code uses the warnings library to ignore warnings.
The process_data function uses the os library to interact with the operating system and manipulate directories.
The code uses the pandas library for data manipulation and analysis.
The code uses Feather files for data storage and retrieval.
"""

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
#path = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\dataframes'
#path = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper\Datosparaorganizardataframes/'
ica = '58x25'
A = 'G1'
B = ''
s = ['ic']
#h = ['neuroHarmonize', 'sovaharmony']
h = ['neuroHarmonize']
group = 1  # Ajustado a 1 ya que A está definido
path = fr'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper\Nuevo analisis corr 2to1 PSM (54X10)(58X25)/{ica}'

for space in s:
    for neuro in h:
        process_data(path, neuro, space, group, A, B, 79)
        #process_data(path, neuro, space, group, A, B, 31)
        #process_data(path, neuro, space, group, A, B, 15)




