import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ejemplo de uso
neuro = 'neuroHarmonize'
name = 'G1'
space = 'ic'
path = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\dataframes"
path_save = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\Resultados"
class_names = ['ACr', 'HC']
fbest_model1 = None
str_ratio = '10to1'

path_plot = os.path.join(path_save, f'graphics/ML/{neuro}/{name}_{str_ratio}_{space}/distributions')
#file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
file_path = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\Resultados\graphics\ML\neuroHarmonize\G1_10to1_ic\Entrenado con modelo 10to1 incremental\Data_integration_corr.feather"
data = pd.read_feather(file_path)
#{'ACr': 0, 'HC': 1}
data['group'] = data['group'].replace({0: 'ACr', 1: 'HC'})

df = pd.DataFrame(data)

## Columnas específicas
#selected_columns = list(set([
#    'power_C5_Beta3', 'power_C10_Beta3', 'crossfreq_C1_Mbeta3_Beta3', 'cohfreq_C4_Theta',
#    'power_C1_Beta3', 'power_C7_Beta2', 'power_C10_Theta', 'power_C9_Beta3', 'sl_C1_Alpha-1',
#    'entropy_C3_Alpha-1', 'sl_C4_Delta', 'entropy_C10_Theta', 'power_C9_Alpha-2', 'entropy_C9_Theta',
#    'crossfreq_C4_Mbeta3_Beta3', 'power_C9_Alpha-1', 'power_C10_Delta', 'power_C1_Gamma',
#    'crossfreq_C7_Malpha-1_Alpha-1', 'entropy_C7_Alpha-1'
#]))
#
## Filtrar el DataFrame con las columnas seleccionadas
#df = df[['group'] + selected_columns]

# Separar el DataFrame en dos grupos según 'group'
group1 = df[df['group'] == 'ACr']
group2 = df[df['group'] == 'HC']

# Lista para almacenar los tamaños de efecto (Cohen's d)
effect_sizes = []

# Eliminar columnas no numéricas excepto 'group'
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calcular Cohen's d para cada columna numérica de características
for col in numeric_cols:
    mean_group1 = group1[col].mean()
    mean_group2 = group2[col].mean()
    std_pooled = np.sqrt((group1[col].std() ** 2 + group2[col].std() ** 2) / 2)
    cohens_d = (mean_group1 - mean_group2) / std_pooled
    effect_sizes.append((col, cohens_d))

# Mostrar resultados
for col, effect_size in effect_sizes:
    print(f"Effect size (Cohen's d) for '{col}': {effect_size:.2f}")

# Extraer nombres de columnas y tamaños de efecto
cols = [col for col, _ in effect_sizes]
effect_sizes_values = [effect_size for _, effect_size in effect_sizes]

# Crear una carpeta para guardar las gráficas si no existe
os.makedirs(path_plot, exist_ok=True)

# Diccionario para cambiar los nombres de las características
name_mapping = {
    'power': 'Relative power',
    'cohfreq': 'Coherence',
    'crossfreq': 'Cross-frequency',
    'sl': 'Synchronization Likelihood',
    'entropy': 'Entropy'
}

# Graficar distribuciones para cada característica seleccionada
for col in numeric_cols:
    # Calcular Cohen's d
    mean_group1 = group1[col].mean()
    mean_group2 = group2[col].mean()
    std_pooled = np.sqrt((group1[col].std() ** 2 + group2[col].std() ** 2) / 2)
    cohens_d = (mean_group1 - mean_group2) / std_pooled

    # Obtener la parte del nombre antes y después del primer guion bajo
    parts = col.split('_', 1)
    if len(parts) > 1:
        prefix, rest_of_name = parts
    else:
        prefix = parts[0]
        rest_of_name = ''
    new_prefix = name_mapping.get(prefix, prefix)
    new_col_name = f'{new_prefix} {rest_of_name}'
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    sns.kdeplot(group1[col], shade=True, color="red", label='ACr', alpha=0.5)
    sns.kdeplot(group2[col], shade=True, color="blue", label='HC', alpha=0.5)
    plt.title(new_col_name, fontsize=20)
    plt.xlabel(f'Effect size between ACr and HC',fontsize=20)
    plt.ylabel('Density',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(title=f"Cohen's d = {cohens_d:.2f}",fontsize=20,title_fontsize=20)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(os.path.join(path_plot, f'distribution_{col}.png'))
    
    # Mostrar la gráfica
    #plt.show()
    plt.close()

# Graficar tamaños de efecto en gráfico de barras horizontales
#plt.figure(figsize=(10, 6))
#plt.barh(cols, effect_sizes_values, color=['red' if d < 0 else 'blue' for d in effect_sizes_values])
#plt.xlabel("Cohen's d")
#plt.title("Tamaño del efecto (Cohen's d) entre grupos ACr y HC")
#plt.grid(True, axis='x')
#plt.show()
#
## Crear un DataFrame para los tamaños de efecto
#df_effects = pd.DataFrame(effect_sizes, columns=['Feature', "Cohen's d"])
#
## Graficar distribución de tamaños de efecto en gráfico de violin
#plt.figure(figsize=(12, 8))
#sns.violinplot(x="Cohen's d", y='Feature', data=df_effects, palette='viridis')
#plt.xlabel("Cohen's d")
#plt.ylabel('Feature')
#plt.title("Distribución del tamaño del efecto (Cohen's d) entre grupos ACr y HC")
#plt.grid(True, axis='x')
#plt.show()
#
## Preparar datos para el gráfico de radar
#categories = [col for col, _ in effect_sizes]
#values = [effect_size for _, effect_size in effect_sizes]
#
## Agregar el primer valor al final para cerrar el círculo
#values += values[:1]
#categories += categories[:1]
#
## Ángulos del gráfico
#angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
#
## Hacer que el gráfico sea circular
#values += values[:1]
#angles += angles[:1]
#
## Graficar gráfico de radar
#plt.figure(figsize=(8, 8))
#ax = plt.subplot(111, polar=True)
#ax.fill(angles, values, color='blue', alpha=0.25)
#ax.plot(angles, values, color='blue', linewidth=1, linestyle='solid')
#ax.set_yticklabels([])
#plt.title("Tamaño del efecto (Cohen's d) entre grupos ACr y HC")
#plt.xticks(angles[:-1], categories, size=10)
#plt.show()
#
## Crear un gráfico por cada métrica
#metrics = set(col.split('_')[0] for col in numeric_cols)
#
#for metric in metrics:
#    metric_cols = [col for col in numeric_cols if col.startswith(metric)]
#    num_cols = len(metric_cols)
#    num_rows = (num_cols + 2) // 3
#
#    plt.figure(figsize=(15, 5 * num_rows))
#    for i, col in enumerate(metric_cols, 1):
#        plt.subplot(num_rows, 3, i)
#        sns.kdeplot(group1[col], shade=True, color="blue", label='ACr', alpha=0.5)
#        sns.kdeplot(group2[col], shade=True, color="red", label='HC', alpha=0.5)
#        plt.title(f'Distribution of {col}')
#        plt.xlabel(col)
#        plt.ylabel('Density')
#        plt.legend()
#
#    plt.suptitle(f'Distribution of {metric}')
#    plt.tight_layout(rect=[0, 0, 1, 0.96])
#    plt.show()
#