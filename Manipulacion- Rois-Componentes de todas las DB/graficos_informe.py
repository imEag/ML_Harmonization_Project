import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import defaultdict


# In[] Features
palette = ["#8AA6A3","#127369","#10403B","#45C4B0"]
# Crear una instancia de Tkinter (necesaria para los cuadros de diálogo)
root = Tk()
root.withdraw()  # Ocultar la ventana principal de Tkinter

# Abre un cuadro de diálogo para seleccionar un archivo
file_path = askopenfilename(title="Seleccionar archivo", filetypes=[("Archivos de texto", "*.txt")])

# Cargar el archivo seleccionado en un DataFrame
if file_path:
    df = pd.read_csv(file_path, delimiter='\t', header=None)  # Ajustar el delimitador según sea necesario

    # Crear un DataFrame vacío
    categories_df = pd.DataFrame()

    # Extraer y contar las categorías usando listas de comprensión
    categories_df['Feature'] = [t.split('_')[0] for t in df[0]]
    #categories_df['IC'] = [t.split('_')[1] if 'age' not in t else None for t in df[0]]
    categories_df['ROI'] = [t.split('_')[1] if 'age' not in t and 'sex' not in t else None for t in df[0]]
    categories_df['Mband'] = [t.split('_')[2] if len(t.split('_')) == 4 else None for t in df[0]]
    categories_df['Band'] = [t.split('_')[2] if len(t.split('_')) >= 3 and 'age' not in t and 'sex' not in t and not t.split('_')[2].startswith('M') else t.split('_')[3] if 'age' not in t and 'sex' not in t else None for t in df[0]]

    # Crear subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    fig.suptitle("Discriminant analysis of the most relevant features using Decition tree without neuroHarmonize in ROIs", fontsize=15, x=0.55)  # 54x10
    # Graficar cada columna en un subplot
    for i, col in enumerate(categories_df.columns):
        ax = axs[i]
        if i == 0:  # Cambiar el color de la segunda barra del primer gráfico en (1, 1)
            ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[0], palette[1], palette[1], palette[0], palette[0], palette[0]])
        elif i == 1:  
            ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[1], palette[1], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0]])
        elif i == 2:  
            ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[1], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0]])
        elif i == 3:  
            ax.bar(categories_df[col].value_counts().index, categories_df[col].value_counts().values, color=[palette[0], palette[0], palette[1], palette[0], palette[1], palette[0], palette[1], palette[0],palette[0], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0], palette[0]])
        else:
            pass
        ax.set_title(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)


    
    plt.tight_layout()
    plt.show()
    print('ok')

# In[] Generaciones

import matplotlib.pyplot as plt

data = [
    ("Generation 1", "ExtraTreesClassifier", 97),
    ("Generation 2", "ExtraTreesClassifier", 97),
    ("Generation 3", "ExtraTreesClassifier", 97),
    ("Generation 4", "ExtraTreesClassifier", 97),
    ("Generation 5", "ExtraTreesClassifier", 97)
]

generations = [item[0] for item in data]
classifiers = [item[1] for item in data]
accuracies = [item[2] for item in data]

fig, ax = plt.subplots(figsize=(10, 6))

# Define colors based on classifier
colors = [palette[2] if classifier == "ExtraTreesClassifier" else palette[3] for classifier in classifiers]

ax.bar(generations, accuracies, color=colors)
ax.axhline(y=97, color='r', linestyle='--', label='97% Accuracy')
ax.annotate('97%', xy=('Generation 5', 97), xytext=('Generation 5', 101), color='r', fontsize=12)
ax.set_xlabel('Generation')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy by Generation without neuroHarmonize')
ax.set_ylim(0, 100)  # Set y-axis limit

# Adding legend for color explanation
extra_trees_patch = plt.Line2D([0], [0], marker='s', color='w', label='ExtraTreesClassifier', markerfacecolor=palette[2], markersize=10)
other_patch = plt.Line2D([0], [0], marker='s', color='w', label='Other Classifier', markerfacecolor=palette[3], markersize=10)
ax.legend(handles=[extra_trees_patch, other_patch])

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# In[] 
data = [
    ("Generation 1.0", "DecisionTreeClassifier", 71),
    ("Generation 1.1", "XGBClassifier", 73),
    ("Generation 2.0", "DecisionTreeClassifier", 71),
    ("Generation 2.1", "XGBClassifier", 73),
    ("Generation 2.2", "RandomForestClassifier", 75),
    ("Generation 2.3", "RandomForestClassifier", 75),
    ("Generation 3.0", "DecisionTreeClassifier", 71),
    ("Generation 3.1", "GradientBoostingClassifier", 78),
    ("Generation 4.0", "XGBClassifier", 73),
    ("Generation 4.1", "GradientBoostingClassifier", 78),
    ("Generation 5.0", "GradientBoostingClassifier", 76),
    ("Generation 5.1", "GradientBoostingClassifier", 78)
]

generations = [item[0] for item in data]
classifiers = [item[1] for item in data]
accuracies = [item[2] for item in data]

unique_classifiers = list(set(classifiers))

fig, ax = plt.subplots(figsize=(10, 6))

# Define colors based on classifier
colors = [palette[i] for i, cls in enumerate(unique_classifiers)]

ax.bar(generations, accuracies, color=colors)
ax.axhline(y=78, color='r', linestyle='--', label='78% Accuracy')
ax.annotate('78%', xy=('Generation 1.0', 78), xytext=('Generation 1.0', 79), color='r', fontsize=12)
ax.set_xlabel('Generation')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy by Generation without neuroHarmonize')
ax.set_ylim(0, 105)  # Set y-axis limit

# Adding legend for color explanation
handles = [plt.Line2D([0], [0], color=colors[i], label=cls, linewidth=3) for i, cls in enumerate(unique_classifiers)]
ax.legend(handles=handles, loc='upper right')


plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# In[] Tamaño del efecto para las 4 bases de datos en controles
import pandas as pd
import numpy as np
import itertools

A = 'BIOMARCADORES'
B = 'DUQUE'
C = 'SRM'
D = 'CHBMP'

group_combinations = list(itertools.combinations([A, B, C, D], 2))

ez = []

for combination in group_combinations:
    group1, group2 = combination
    effect_size = data_DB.groupby([space, 'Band']).apply(
        lambda data_DB: pg.compute_effsize(data_DB[data_DB['database'] == group1][metric],
                                           data_DB[data_DB['database'] == group2][metric])
    )
    ez.append(effect_size)

ez_df = pd.concat(ez, axis=0)
ez_df = ez_df.reset_index()
ez_df.rename(columns={0: 'effect size'})
ez_df['A'] = A
ez_df['B'] = B
ez_df['C'] = C
ez_df['D'] = D
ez_df['Prueba'] = 'effect size'

# Coefficient of variation (cv)
std = data_DB.groupby([space, 'Band']).apply(
    lambda data_DB: np.std(np.concatenate((data_DB[data_DB['database'] == A][metric],
                                            data_DB[data_DB['database'] == B][metric]), axis=0))
).to_frame()
std = std.reset_index()
std.rename(columns={0: 'std'})
std['A'] = A
std['B'] = B
std['C'] = C
std['D'] = D
std['Prueba'] = 'std'

table_concat = pd.concat([ez_df, std], axis=0)
table_concat = table_concat.reset_index()
table = pd.pivot_table(table_concat, columns=['Prueba'],
                       index=[space, 'Band', 'A', 'B','C','D'])

table.to_csv(r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Correcciones_Evaluador\Tamaño del efecto\output_table.csv')  # Save the table to a CSV file