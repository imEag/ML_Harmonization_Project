"""
Importing Libraries
The necessary libraries are imported, including Pandas, NumPy, Matplotlib, and MNE-Python.
Defining the topograph Function
This function takes in a dataset and an output path to save the topographic maps.
It replaces group labels in the data and defines frequency bands for analysis.
It generates standard positions in a circle for components and obtains average metric values for each component and frequency band.
It configures the figure and axes for each metric and generates topographic maps using MNE-Python's plot_topomap function.
It saves and displays each figure.
Configuring Directories and File Names
Directories and file names are defined for reading and saving data.
Iterating Over Different Cases and Generating Topographic Maps
The code iterates over different cases (neuro, name, ratio) and calls the topograph function to generate topographic maps for each metric.
It reads the corresponding data file and calls the topograph function with the corresponding output path.
Using the Code
Group labels are replaced in the data, and unnecessary columns are dropped.
Average values are calculated for each group, and topographic maps are generated for each feature using the single_topomap function from the utils library.
Figures are saved to the corresponding output path.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import os
import utils as us


def topograph(data, path_out):
    # Remplazar etiquetas de grupo
    data['group'] = data['group'].replace({'G1': 'ACr', 'Control': 'HC'})

    # Definir bandas de frecuencia
    bands = {
        'Delta': (1.5, 6),
        'Theta': (6, 8.5),
        'Alpha-1': (8.5, 10.5),
        'Alpha-2': (10.5, 12.5),
        'Beta1': (12.5, 18.5),
        'Beta2': (18.5, 21),
        'Beta3': (21, 30),
        'Gamma': (30, 45)
    }

    # Definir componentes por defecto
    default_components = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']

    # Generar posiciones estándar en un círculo
    def generate_standard_positions(num_components):
        theta = np.linspace(0, 2 * np.pi, num_components, endpoint=False)
        radius = 0.5  # Radio del círculo, puedes ajustarlo si es necesario
        positions = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
        return positions

    # Obtener posiciones estándar para componentes
    standard_positions = generate_standard_positions(len(default_components))
    standard_component_positions = dict(zip(default_components, standard_positions))

    # Función para obtener valores promedio de métricas
    def get_mean_values(df, components):
        metrics = ['crossfreq', 'power', 'sl', 'cohfreq', 'entropy']
        crossfreq_columns = [col for col in df.columns if col.startswith('crossfreq')]
        mbands = list(set([col.split('_')[2] for col in crossfreq_columns if col.split('_')[2].startswith('M')]))

        topomaps = {metric: {band: {group: {component: None for component in components} for group in ['ACr', 'HC']} for band in bands} for metric in metrics}
        
        for metric in metrics:
            for band in bands:
                for group in ['ACr', 'HC']:
                    mean_values = {component: 0 for component in components}
                    for component in components:
                        if metric == 'crossfreq':
                            for mband in mbands:
                                metric_band_column = f'{metric}_{component}_{mband}_{band}'
                                if metric_band_column in df.columns:
                                    mean_values[component] += df[df['group'] == group][metric_band_column].mean()
                        else:
                            metric_band_column = f'{metric}_{component}_{band}'
                            if metric_band_column in df.columns:
                                mean_values[component] = df[df['group'] == group][metric_band_column].mean()

                    if components[0] in mean_values and components[1] in mean_values:
                        diff = mean_values[components[0]] - mean_values[components[1]]
                        topomaps[metric][band][group] = [diff for _ in components]

        return topomaps

    # Obtener los valores medios de las métricas
    topomaps = get_mean_values(data, default_components)

    # Configurar la figura y los ejes para cada métrica
    for metric, metric_data in topomaps.items():
        fig, axes = plt.subplots(len(bands), len(['ACr', 'HC']), figsize=(15, 10))
        fig.suptitle(f'Diferencias Relativas de {metric.capitalize()} entre Grupos para Todas las Bandas de Frecuencia')

        # Iterar sobre las bandas y los grupos para generar los mapas topográficos
        for i, (band, group_data) in enumerate(metric_data.items()):
            for j, (group, values) in enumerate(group_data.items()):
                ax = axes[i, j]

                # Generar un pequeño desplazamiento aleatorio para las posiciones de los electrodos
                jitter_amount = 0.1
                pos = np.array([standard_component_positions[c] + np.random.uniform(-jitter_amount, jitter_amount, size=2) for c in default_components])
                
                # Graficar el mapa topográfico
                im, _ = plot_topomap(values, pos, axes=ax, show=False, cmap='RdBu_r')

                # Añadir título y etiquetas a los ejes
                if i == 0:
                    ax.set_title(f'{group} Power')
                if j == 0:
                    ax.set_ylabel(f'{band} ({bands[band][0]}-{bands[band][1]} Hz)')
                ax.axis('off')

        # Añadir barra de color y ajustar el diseño
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
        cbar.set_label('t-values')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Guardar y mostrar la figura
        os.makedirs(path_out, exist_ok=True)
        plt.savefig(os.path.join(path_out, f'generated_topomap_{metric}.png'))
        plt.show()
        print('listo')

# Configuración de los directorios y nombres de archivo
neuros = ['neuroHarmonize', 'sovaharmony']
names = ['G1']
space = 'ic'
path = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\dataframes"
path_save = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\Resultados"
ratios = [79, 31, 15]

# Iterar sobre los diferentes casos y generar los mapas topográficos para cada métrica
for neuro in neuros:
    for name in names:
        for ratio in ratios:
            if ratio == 79:
                str_ratio = '2to1'
            elif ratio == 31:
                str_ratio = '5to1'
            elif ratio == 15:
                str_ratio = '10to1'
            else:
                str_ratio = str(ratio)
            
            # Leer el archivo de datos y llamar a la función para generar los mapas topográficos para cada métrica
            file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
            data = pd.read_feather(file_path)
            path_out = os.path.join(path_save, f'graphics/ML/{neuro}/{name}_{str_ratio}_{space}')
            #topograph(data, path_out)

            # Uso del código
            data['group'] = data['group'].replace({'G1': 'ACr', 'Control': 'HC'})
            data = data.drop(['age','MM_total','FAS_F','FAS_S','FAS_A','education'],axis=1)
            ACr = data[data['group']=='ACr']
            ACr=ACr.select_dtypes(include=['number'])
            ACr=ACr.mean() 
            HC = data[data['group']=='HC']
            HC=HC.select_dtypes(include=['number'])
            HC=HC.mean()

            for group,glabel in zip([ACr,HC],['ACr','HC']):
                A, W, ch_names = us.get_spatial_filter('54x10')
                A_old=A.copy()
                max_values = np.max(A_old, axis=0)
                A=A_old/max_values
                
                ch_names = [x.replace(' ', '') for x in ch_names]
                label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
                comp = 1
                features = []

                for f in ACr.index.copy().tolist():
                    properties=f.split('_')
                    properties.pop(1)
                    features.append('_'.join(properties))
                
                features=list(set(features))
                features.sort()
                #features=list(set(['_'.join(list(x.split('_')[0])+['_']+list(x.split('_')[2:])) for x in ACr.index.copy().tolist()]))
                A[:,comp].shape
                os.makedirs(rf'topofeatures/{neuro}/{name}/{str_ratio}/{glabel}',exist_ok=True)
                for feature in features:
                    #feature=features[0]
                    def get_df_safe(df,key):
                        if key in df:
                            return df[key]
                        else:
                            return 0
                    def exist_df_key(df,key):
                        return int(key in df)
                    value_per_component=[get_df_safe(ACr,f"{feature.split('_')[0]}_C{x}_{'_'.join(feature.split('_')[1:])}") for x in range(1,11)]
                    num_per_component=[exist_df_key(ACr,f"{feature.split('_')[0]}_C{x}_{'_'.join(feature.split('_')[1:])}") for x in range(1,11)]
                    comps_order=[x for x in range(1,11)]
                    A.shape
                    Afeature=np.zeros((A.shape[0]))
                    for chan in range(Afeature.shape[0]):
                        val_final=0
                        for co,val,num in zip(comps_order,value_per_component,num_per_component):
                            print(co,val,num)
                            w= A[chan,co-1]
                            val_final+=np.abs(w)*val

                        Afeature[chan]=val_final/sum(num_per_component)

                    #A_thresholded=(np.abs(A[:, comp]) > np.abs(A[:, comp]).mean()).astype(int)

                    fig3 = us.single_topomap(Afeature, ch_names, show_names=False, label=feature,show=False,cmap='seismic',title=feature)
                    fig3.savefig(f'topofeatures/{neuro}/{name}/{str_ratio}/{glabel}/{feature}.png')
                    plt.close('all')

