import pandas as pd 
import itertools
import seaborn as sns
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import warnings
from Funciones import dataframe_long_roi,dataframe_long_components,dataframe_long_cross_ic,dataframe_long_cross_roi
from Funciones import columns_SL_roi,columns_coherence_roi,columns_entropy_rois,columns_powers_rois
from Funciones import columns_SL_ic,columns_coherence_ic,columns_entropy_ic,columns_powers_ic,columns_cross_ic
warnings.filterwarnings("ignore")
import os
import tkinter as tk
from tkinter.filedialog import askdirectory

tk.Tk().withdraw() # part of the import if you are not using other tkinter functions


def graphics(data,type,path,name_band,id,id2,A,B,space,id_cross=None,num_columns=4,save=True,plot=True,palette=None,l=None):
    '''Function to make graphs of the given data '''
    data['database'].replace({'BIOMARCADORES':'UdeA 1','DUQUE':'UdeA 2'}, inplace=True)
    new_path = rf'{path}\Graficos_{type}'
    os.makedirs(new_path,exist_ok=True)
    max=data[type].max()
    sns.set(rc={'figure.figsize':(15,12)})
    sns.set_theme(style="white")
    if id=='ic':
        col='Component'
    else:
        col='ROI'
    axs=sns.catplot(x='group',y=type,data=data,dodge=True, kind="box",col=col,col_wrap=num_columns,palette=palette,fliersize=1.5,linewidth=0.5,legend=False)
    #plt.yticks(np.arange(0,round(max),0.1))
    axs.set(xlabel=None)
    axs.set(ylabel=None)
    axs.set_titles(size=20)
    if id_cross==None:
        axs.fig.suptitle(type+' in '+r'$\bf{'+name_band.replace('-','')+r'}$'+ ' in the ICs of normalized data given by the databases',fontsize=20,x=0.55)
    else:
        axs.fig.suptitle(type+' in '+id_cross.replace('-','')+' of ' +r'$\bf{'+name_band.replace('-','')+r'}$'+ ' in the ICs of normalized data given by the databases',fontsize=20,x=0.55)
    if id=='IC':
        axs.add_legend(loc='upper right',bbox_to_anchor=(.59,.95),ncol=4,title="",fontsize=20)#title="Database"
        #top=0.85,bottom=0.121, right=0.986,left=0.05, hspace=0.138, wspace=0.062
        axs.fig.subplots_adjust(top=0.917,bottom=0.067,left=0.072,right=0.987,hspace=0.248,wspace=0.092) 
        axs.fig.text(0.5, 0.04, 'Group', ha='center', va='center',fontsize=20)
        axs.fig.text(0.01, 0.5,  type, ha='center', va='center',rotation='vertical',fontsize=20) 
    else:
        axs.add_legend(loc='upper right',bbox_to_anchor=(.7,.95),ncol=4,title="",fontsize=20)#title="Database"
        axs.fig.subplots_adjust(top=0.917,bottom=0.067,left=0.072,right=0.987,hspace=0.248,wspace=0.092) # adjust the Figure in rp
        axs.fig.text(0.5, 0.04, 'Group', ha='center', va='center',fontsize=20)
        axs.fig.text(0.015, 0.5,  type, ha='center', va='center',rotation='vertical',fontsize=20)
    if plot:
        plt.show()
    if save==True:
        #path2 = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Correcciones_Evaluador\Armonizados'+path[-18:]
        path2 = rf'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_54x10\Graficos\{l}'
        verific = '{path}\{id}\Graficos_{type}'.format(path=path2,name_band=name_band,id=id,type=type)
        if not os.path.exists(verific):
            os.makedirs(verific)  
        if id_cross==None:
            path_complete='{path}\{id}\Graficos_{type}\{name_band}_{type}_{id}.png'.format(path=path2,name_band=name_band,id=id,type=type)
            #path_complete= fr'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados\graphics\unmatched\{name_band}_{type}_{id}.png'
        else:
            path_complete='{path}\{id}\Graficos_{type}\{name_band}_{id_cross}_{type}_{id}.png'.format(path=path2,name_band=name_band,id=id,type=type,id_cross=id_cross) 
            #path_complete= fr'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados\graphics\unmatched\{name_band}_{id_cross}_{type}_{id}.png'
        plt.savefig(path_complete)
        plt.close()
        return path_complete
    else:
        return None
    

def text_format(val,value):
    if value==0.2: #Cambie el 0.05 por 0.2 y el lightgreen por lightblue
        color = 'lightblue' if val <0.2 else 'white'
    if value==0.7:
        color = 'lightgreen' if np.abs(val)>=0.7 else 'white'
    if value==0.0:
        color = 'lightblue' if np.abs(val)<=0.05 else 'white'
#    elif value==0.8:
#        if val >=0.7 and val<0.8:
#            color = 'salmon'
#        elif val >=0.8:
#            color = 'lightblue' 
#        else:
#            color='white'

    return 'background-color: %s' % color

def stats_pair(data,metric,space,A,B):
    print(metric,space)
    data_DB=data.copy()
    if metric!='Cross Frequency':
        ez=data_DB.groupby([space,'Band']).apply(lambda data_DB:pg.compute_effsize(data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric])).to_frame()
        ez=ez.rename(columns={0:'effect size'})
        ez['A']=A
        ez['B']=B
        ez['Prueba']='effect size'
        #cv
        std=data_DB.groupby([space,'Band']).apply(lambda data_DB:np.std(np.concatenate((data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric]),axis=0))).to_frame()
        std=std.rename(columns={0:'cv'})
        std['A']=A
        std['B']=B
        std['Prueba']='cv'

        table_concat=pd.concat([ez,std],axis=0)
        table=pd.pivot_table(table_concat,values=['effect size','cv'],columns=['Prueba'],index=[space,'Band','A', 'B'])
    else:
        ez=data_DB.groupby([space,'Band','M_Band']).apply(lambda data_DB:pg.compute_effsize(data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric])).to_frame()
        ez=ez.rename(columns={0:'effect size'})
        ez['A']=A
        ez['B']=B
        ez['Prueba']='effect size'
        #cv
        std=data_DB.groupby([space,'Band','M_Band']).apply(lambda data_DB:np.std(np.concatenate((data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric]),axis=0))).to_frame()
        std=std.rename(columns={0:'cv'})
        std['A']=A
        std['B']=B
        std['Prueba']='cv'

        table_concat=pd.concat([ez,std],axis=0)
        table=pd.pivot_table(table_concat,values=['effect size','cv'],columns=['Prueba'],index=[space,'Band','M_Band','A', 'B'])
    
    table=table.reset_index()
    table=table.style.applymap(text_format,value=0.7,subset=['effect size']).applymap(text_format,value=0.0,subset=['cv'])
    
    #dfi.export(table, path_complete)
    return table

def stats_pair_database(data,metric,space):
    print(metric,space)
    data_DB=data.copy()
    A = 'BIOMARCADORES'
    B = 'DUQUE'
    C = 'SRM'
    D = 'CHBMP'

    group_combinations = list(itertools.combinations([A, B, C, D], 2))
    if metric!='Cross Frequency':
        ez = []
        std = []
        for combination in group_combinations:
            group1, group2 = combination
            effect_size = data_DB.groupby([space, 'Band']).apply(lambda data_DB: pg.compute_effsize(data_DB[data_DB['database'] == group1][metric],
                                                data_DB[data_DB['database'] == group2][metric]))
            ez.append(effect_size)

        ez_df = pd.concat(ez, axis=0)
        ez_df = ez_df.reset_index()
        ez_df.rename(columns={0: 'effect size'})
        ez_df['A'] = A
        ez_df['B'] = B
        ez_df['C'] = C
        ez_df['D'] = D
        ez_df['Prueba'] = 'effect size'

        for combination in group_combinations:
            group1, group2 = combination
            std_value = data_DB.groupby([space, 'Band']).apply(
                lambda data_DB: np.std(np.concatenate((data_DB[data_DB['database'] == group1][metric],
                                                        data_DB[data_DB['database'] == group2][metric]),
                                                        axis=0))
            )
            std.append(std_value)
        
        std = pd.concat(std, axis=0)
        std = std.reset_index()
        std.rename(columns={0: 'std'})
        std['A'] = A
        std['B'] = B
        std['C'] = C
        std['D'] = D
        std['Prueba'] = 'std'

        table_concat = pd.concat([ez_df, std], axis=0)
        table_concat = table_concat.reset_index()
        table = pd.pivot_table(table_concat, columns=['Prueba'],index=[space, 'Band', 'A', 'B','C','D'])
        table=table.reset_index()
    else:
        ez = []
        std = []
        for combination in group_combinations:
            group1, group2 = combination
            effect_size = data_DB.groupby([space, 'Band','M_Band']).apply(lambda data_DB: pg.compute_effsize(data_DB[data_DB['database'] == group1][metric],
                                                data_DB[data_DB['database'] == group2][metric]))
            ez.append(effect_size)

        ez_df = pd.concat(ez, axis=0)
        ez_df = ez_df.reset_index()
        ez_df.rename(columns={0: 'effect size'})
        ez_df['A'] = A
        ez_df['B'] = B
        ez_df['C'] = C
        ez_df['D'] = D
        ez_df['Prueba'] = 'effect size'

        for combination in group_combinations:
            group1, group2 = combination
            std_value = data_DB.groupby([space, 'Band','M_Band']).apply(
                lambda data_DB: np.std(np.concatenate((data_DB[data_DB['database'] == group1][metric],data_DB[data_DB['database'] == group2][metric]),axis=0)))
            std.append(std_value)
        
        std = pd.concat(std, axis=0)
        std = std.reset_index()
        std.rename(columns={0: 'std'})
        std['A'] = A
        std['B'] = B
        std['C'] = C
        std['D'] = D
        std['Prueba'] = 'std'

        table_concat = pd.concat([ez_df, std], axis=0)
        table_concat = table_concat.reset_index()
        table = pd.pivot_table(table_concat, columns=['Prueba'],index=[space, 'Band','M_Band', 'A', 'B','C','D'])
        table=table.reset_index()
        
    
    #table=table.style.applymap(text_format,value=0.7,subset=['effect size']).applymap(text_format,value=0.0,subset=['std'])
    
    #dfi.export(table, path_complete)
    return table

def effect_size_inside_DB(data_i,metric,space,A,B):
    data=data_i.copy()
    if metric!='Cross Frequency':
        groupby=[space,'Band','database']
        l_index=['database',space,'Band','A', 'B']
    else:
        groupby=[space,'Band','M_Band','database']
        l_index=['database',space,'Band','M_Band','A', 'B']

    databases=data['database'].unique().tolist()
    databases.sort()
    db_copy=databases.copy()
    for i,db in enumerate(databases):
        groups=data[data['database']==databases[i]]['group'].unique()
        if len(groups)==1:
             db_copy.remove(db)

    tablas={}
    for DB in db_copy:
        data_DB=data[data['database']==DB]
        combinaciones=[(A, B)]
        test_ez={}
        test_std={}
        for i in combinaciones:
            #Effect size
            ez=data_DB.groupby(groupby).apply(lambda data_DB:pg.compute_effsize(data_DB[data_DB['group']==i[0]][metric],data_DB[data_DB['group']==i[1]][metric]))
            ez=ez.astype(float, errors = 'raise')
            ez=ez.to_frame()
            ez=ez.rename(columns={0:'effect size'})
            ez['A']=i[0]
            ez['B']=i[1]
            ez['Prueba']='effect size'
            test_ez['effsize-'+i[0]+'-'+i[1]]=ez
            #cv
            std=data_DB.groupby(groupby).apply(lambda data_DB:np.std(np.concatenate((data_DB[data_DB['group']==i[0]][metric],data_DB[data_DB['group']==i[1]][metric]),axis=0))).to_frame()
            std=std.rename(columns={0:'cv'})
            std['A']=i[0]
            std['B']=i[1]
            std['Prueba']='cv'
            test_std['cv-'+i[0]+'-'+i[1]]=std
            
        table_ez=pd.concat(list(test_ez.values()),axis=0)
        table_ez.reset_index( level = [0,1],inplace=True )
        table_std=pd.concat(list(test_std.values()),axis=0)
        table_std.reset_index( level = [0,1],inplace=True )
        table_concat=pd.concat([table_ez,table_std],axis=0)
        table_x=pd.pivot_table(table_concat,values=['effect size','cv'],columns=['Prueba'],index=l_index)
        table_x.columns=['effect size','cv']
        tablas[DB]=table_x
    table=pd.concat(list(tablas.values()),axis=0)
    print(table)
    table=table.reset_index()
    table=table.style.applymap(text_format,value=0.7,subset=['effect size']).applymap(text_format,value=0.0,subset=['cv'])
    
    #dfi.export(table, path_complete)
    return table


def graph_harmonize(path,data_roi_sova,data_roi_harmo,space,A,B):
    columns_All=data_roi_harmo.copy().columns.tolist()
    for i in ['participant_id', 'group', 'visit', 'condition','database','sex','MM_total','FAS_F','FAS_S','FAS_A','education','age']:
        columns_All.remove(i)
    # if space == 'ic':
    #     columns = [columns_All[:64],columns_All[64:128],columns_All[128:192],columns_All[192:256],columns_All[256:]]
    # elif space == 'roi':
    #     columns = [columns_All[:32],columns_All[32:64],columns_All[64:96],columns_All[96:128],columns_All[128:]]
    if space == 'ic':
        #columns = [columns_powers_ic,columns_SL_ic,columns_coherence_ic,columns_entropy_ic,columns_All[256:]]
        columns = [columns_powers_ic,columns_SL_ic,columns_coherence_ic,columns_entropy_ic,columns_cross_ic]
    elif space == 'roi':
        columns = [columns_powers_rois,columns_SL_roi,columns_coherence_roi,columns_entropy_rois,columns_All[128:]]

    for i,value in enumerate([data_roi_sova,data_roi_harmo]):
        d_B_roi=value
        if i==0:
            label="sovaHarmony"
        else:
            label="neuroHarmonize"
        #New dataframes from ROIs
        #Dataframes are saved by ROI and components for graphics.
        if space == 'roi':
            if B == 'G2':
                path_ = fr'{path}\{label}\long\{space}\{A+B}'
            else:
                path_ = fr'{path}\{label}\long\{space}\{A}'
            
            os.makedirs(path_,exist_ok=True)
            #Power
            dataframe_long_roi(d_B_roi,'Power',columns=columns[0],name="data_long_power_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #SL
            dataframe_long_roi(d_B_roi,type='SL',columns=columns[1],name="data_long_sl_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #Coherencia
            dataframe_long_roi(d_B_roi,type='Coherence',columns=columns[2],name="data_long_coherence_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #Entropia
            dataframe_long_roi(d_B_roi,type='Entropy',columns=columns[3],name="data_long_entropy_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #Cross frequency
            dataframe_long_cross_roi(d_B_roi,type='Cross Frequency',columns=columns[4],name="data_long_crossfreq_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
        else:
            if B == 'G2':
                path_ = fr'{path}\{label}\long\{space}\{A+B}'
            else:
                path_ = fr'{path}\{label}\long\{space}\{A}'
            
            os.makedirs(path_,exist_ok=True)
            #Power
            dataframe_long_components(d_B_roi,'Power',columns=columns[0],name="data_long_power_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #SL
            dataframe_long_components(d_B_roi,type='SL',columns=columns[1],name="data_long_sl_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #Coherencia
            dataframe_long_components(d_B_roi,type='Coherence',columns=columns[2],name="data_long_coherence_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #Entropia
            dataframe_long_components(d_B_roi,type='Entropy',columns=columns[3],name="data_long_entropy_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)
            #Cross frequency
            dataframe_long_cross_ic(d_B_roi,type='Cross Frequency',columns=columns[4],name="data_long_crossfreq_{space}_{label}_{g}".format(label=label,g=str(A+B),space=space),path=path_)

    #labels=['_harmonized']
    labels=['sovaHarmony','neuroHarmonize']
    for label in labels:
        if B == 'G2':
            C = A+B
        else:
            C = A
        data_p_roi=pd.read_feather(fr'{path}\{label}\long\{space}\{C}\data_long_power_{space}_{label}_{A+B}.feather'.replace('\\','/'))
        data_sl_roi=pd.read_feather(fr'{path}\{label}\long\{space}\{C}\data_long_sl_{space}_{label}_{A+B}.feather'.replace('\\','/'))
        data_c_roi=pd.read_feather(fr'{path}\{label}\long\{space}\{C}\data_long_coherence_{space}_{label}_{A+B}.feather'.replace('\\','/'))
        data_e_roi=pd.read_feather(fr'{path}\{label}\long\{space}\{C}\data_long_entropy_{space}_{label}_{A+B}.feather'.replace('\\','/'))
        data_cr_roi=pd.read_feather(fr'{path}\{label}\long\{space}\{C}\data_long_crossfreq_{space}_{label}_{A+B}.feather'.replace('\\','/')) 
        datos_roi={'Power':data_p_roi,'SL':data_sl_roi,'Coherence':data_c_roi,'Entropy':data_e_roi,'Cross Frequency':data_cr_roi}
        bands= data_sl_roi['Band'].unique()
        bandsm= data_cr_roi['M_Band'].unique()  

        #path_excel = askdirectory()
        path_excel = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_54X10\Tamaño del efecto'
        if B == 'G2':
            path_excel_ = fr'{path_excel}\{label}\{space}\{A+B}'
            file = rf'tabla_effectsize_{space}_{A+B}_{label}.xlsx'
        else:
            path_excel_ = fr'{path_excel}\{label}\{space}\{A}'
            file = rf'tabla_effectsize_{space}_{A}_{label}.xlsx'
        os.makedirs(path_excel_,exist_ok=True)
        writer = pd.ExcelWriter(path_excel_+'/'+file,mode='w')
    
        for metric in datos_roi.keys():
            d_roi=datos_roi[metric]
            if space == 'roi':
                table=stats_pair(d_roi,metric,'ROI',A,B)
                #table=stats_pair_database(d_roi,metric,'ROI')
            else:
                table=stats_pair(d_roi,metric,'Component',A,B)
                #table=stats_pair_database(d_roi,metric,'Component')
            table.to_excel(writer ,sheet_name=metric)
        writer.save()
        writer.close()

        #filename2 = r"{path}\Graficos_armonizacion_sova_harmo\tabla_effectsize_inside_DB_{space}_{group}{label}_03_04_2022.xlsx".format(path=path,label=label,group=str(A+B),space=space)
        #writer2 = pd.ExcelWriter(filename2,mode='w')
        #for metric in datos_roi.keys():
        #    d_roi=datos_roi[metric]
        #    if space == 'roi':
        #        table2=effect_size_inside_DB(d_roi,metric,'ROI',A,B)
        #        table2.to_excel(writer2 ,sheet_name=metric)

        #    else:
        #        table2=effect_size_inside_DB(d_roi,metric,'Component',A,B)
        #        table2.to_excel(writer2 ,sheet_name=metric)

        ##writer2.save()     
        #writer2.close() 
        #print(filename2)
        #path_graph = askdirectory()
        path_graph = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_54x10\Graficos'
        if B == 'G2':
            path_graph = fr'{path_graph}\{label}\{space}\{A+B}'
        else:
            path_graph = fr'{path_graph}\{label}\{space}\{A}'
        os.makedirs(path_graph,exist_ok=True)

        #colors=["#127369","#10403B","#8AA6A3","#45C4B0"]
        colors = ['#708090','darkcyan']
        for metric in datos_roi.keys():
            for band in bands:
                d_roi=datos_roi[metric]
                d_banda_roi=d_roi[d_roi['Band']==band]
                if metric!='Cross Frequency':  
                    print(str(band)+' '+str(metric)) 
                    if space == 'roi':
                        graphics(d_banda_roi,metric,path_graph,band,'roi',label,A=A,B=B,space=space,num_columns=2,save=True,plot=False,palette=colors,l=label)
                    else:
                        graphics(d_banda_roi,metric,path_graph,band,'ic',label,A=A,B=B,space=space,num_columns=2,save=True,plot=False,palette=colors,l=label)
            
                else:
                    #pass
                    for bandm in list(d_banda_roi['M_Band'].unique()):  
                        print(str(band)+' '+str(metric)+' '+str(bandm)) 
                        if d_banda_roi[d_banda_roi['M_Band']==bandm]['Cross Frequency'].iloc[0]!=None:
                            if space == 'roi':
                                graphics(d_banda_roi[d_banda_roi['M_Band']==bandm],'Cross Frequency',path_graph,band,'roi',label,A=A,B=B,space=space,id_cross=bandm,num_columns=2,save=True,plot=False,palette=colors,l=label)
                            else:
                                graphics(d_banda_roi[d_banda_roi['M_Band']==bandm],'Cross Frequency',path_graph,band,'ic',label,A=A,B=B,space=space,id_cross=bandm,num_columns=2,save=True,plot=False,palette=colors,l=label)

def main():
    #path=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' #Cambia dependieron de quien lo corra
    #path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD'
    #path = askdirectory() 
    #path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_54x10\Datosparaorganizardataframes'
    path='/media/gruneco-server/ADATA HD650/BIOMARCADORES/derivatives/data_long/IC'
    # IC
    ##data_ic_sova_G1G2=pd.read_feather(fr'{path}\sovaHarmony\integration\ic\G1G2\Data_complete_ic_sovaharmony_G1G2.feather')
    ##data_ic_harmo_G1G2=pd.read_feather(fr'{path}\neuroHarmonize\integration\ic\G1G2\Data_complete_ic_neuroHarmonize_G1G2.feather')
    #graph_harmonize(path,data_ic_sova_G1G2,data_ic_harmo_G1G2,'ic','G1','G2')
    #print(2)
    #print('Data_complete_ic_sovaharmony_G1\n','Data_complete_ic_neuroHarmonize_G1')
    #data_ic_sova_CTR=pd.read_feather(fr'{path}\sovaHarmony\integration\ic\G1\Data_complete_ic_sovaharmony_G1.feather')
    data_power=pd.read_feather(fr'{path}\data_CE_irasa_long_BIOMARCADORES_54x10_components.feather'.replace('\\','/'))#data_ic_harmo_CTR=pd.read_feather(fr'{path}\neuroHarmonize\integration\ic\G1\Data_complete_ic_neuroHarmonize_G1.feather')

    #graph_harmonize(path,data_ic_sova_CTR,data_ic_harmo_CTR,'ic','Control','Control')
    graph_harmonize(path,data_power,data_ic_harmo_CTR,'ic','G1','Control')
    #print(4)
    #print('Data_complete_ic_sovaharmony_DTA\n','Data_complete_ic_neuroHarmonize_DTA')
    ##data_ic_sova_DTA=pd.read_feather(fr'{path}\sovaHarmony\integration\ic\DTA\Data_complete_ic_sovaharmony_DTA.feather')
    ##data_ic_harmo_DTA=pd.read_feather(fr'{path}\neuroHarmonize\integration\ic\DTA\Data_complete_ic_neuroHarmonize_DTA.feather')
    #graph_harmonize(path,data_ic_sova_DTA,data_ic_harmo_DTA,'ic','DTA','Control')
    #print(6)
    ##data_ic_sova_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_DCL.feather'.format(path=path))
    ##data_ic_harmo_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_DCL.feather'.format(path=path))
    ##graph_harmonize(path,data_ic_sova_DCL,data_ic_harmo_DCL,'ic','DCL','Control')

    # ROI
    ##print('Data_complete_roi_sovaharmony_G2G1\n','Data_complete_roi_neuroHarmonize_G2G1')
    ##data_roi_sova_G1G2=pd.read_feather(fr'{path}\sovaHarmony\integration\roi\G1G2\Data_complete_roi_sovaharmony_G1G2.feather')
    ##data_roi_harmo_G1G2=pd.read_feather(fr'{path}\neuroHarmonize\integration\roi\G1G2\Data_complete_roi_neuroHarmonize_G1G2.feather')
    #graph_harmonize(path,data_roi_sova_G1G2,data_roi_harmo_G1G2,'roi','G1','G2')
    #print(8)
    ##print('Data_complete_roi_sovaharmony_G1\n','Data_complete_roi_neuroHarmonize_G1')
    #data_roi_sova_CTR=pd.read_feather(fr'{path}\sovaHarmony\integration\roi\G1\Data_complete_roi_sovaharmony_G1.feather')
    #data_roi_harmo_CTR=pd.read_feather(fr'{path}\neuroHarmonize\integration\roi\G1\Data_complete_roi_neuroHarmonize_G1.feather')
    #graph_harmonize(path,data_roi_sova_CTR,data_roi_harmo_CTR,'roi','G1','Control')
    #print(10)
    #print('Data_complete_roi_sovaharmony_DTA\n','Data_complete_roi_neuroHarmonize_DTA')
    #data_roi_sova_DTA=pd.read_feather(fr'{path}\sovaHarmony\integration\roi\DTA\Data_complete_roi_sovaharmony_DTA.feather')
    #data_roi_harmo_DTA=pd.read_feather(fr'{path}\neuroHarmonize\integration\roi\DTA\Data_complete_roi_neuroHarmonize_DTA.feather')
    #graph_harmonize(path,data_roi_sova_DTA,data_roi_harmo_DTA,'roi','DTA','Control')
    #print(12)
    #data_roi_sova_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_DCL.feather'.format(path=path))
    #data_roi_harmo_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_DCL.feather'.format(path=path))
    #graph_harmonize(path,data_roi_sova_DCL,data_roi_harmo_DCL,'roi','DCL','Control')


if __name__ == "__main__":
    main()


#new_name = 'Data_yady_'+space+'_'+neuro
#path_integration = fr'{path}\{neuro}\integration\{space}'
#os.makedirs(path_integration,exist_ok=True)
#data.reset_index(drop=True).to_feather('{path}\{name}.feather'.format(path=path_integration,name=new_name))