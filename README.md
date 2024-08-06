# Data_analysis_ML_Harmonization_Proyect

Orden de archivos:

1. ProcessingEEG.py 
2. Dataframes_potencias_Componentes_Demograficos.py 
3. Borrar_datos_atipicos_potencias.py 
4. Dataframes_SL_Coherencia_Entropy_Cross.py 
5. neuroharmonaze_G1CTR.py 
6. unirfeatherharmonize.py 
7. Graficos_power_sl_coherencia_entropia_cross.py
8. graficos_data_harmonized.py
9. ML_models_G1_ic_sovaharmony.py * Mirar notas
10. ML_models_G1_ic_neuroHarmonize.py * Mirar notas

NOTAS
* Los pasos 7,8,9 y 10 pueden ejecutarse simultáneamente sin necesidad de esperar los resultados de los pasos anteriores
* El código de ML integrado y optimizado se encuentra en el script: training_script.py y utiliza las funciones de training_functions.py

# Archivos adicionales para papers = (2 ICA - 54x10 y 58x25) y (Diversificación de dataset 2to1, 5to1 y 10to1)

1. neuroharmonize - V2.py => Se necesita el archivo "Data_complete_ic.feather". Repositirorio eeg_harmonization
2. unirfeatherharmonize - V2.py => Se necesita la ruta de la carpeta "complete" que contiene los complete para cada metrica. Repositorio Data_analysis_ML_Harmonization_Proyect
3. training_script3.py => Se necesita la ruta de la carpeta "integrate" que contiene los integrate para cada metrica. Repositorio Data_analysis_ML_Harmonization_Proyect

# Archivos adicionales de graficos de interes (independientes de los todos los scripts

1. Topofeatures.py edificion del codigo original topo.py del repositorio eeg_harmonization
2. Graficos_Paper.py => Se ejecuta drante el debbug de (3.) para graficar y analizar los resultados por medio de graficas. 

