import pandas as pd 
import seaborn as sns
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import warnings
from Funciones import dataframe_long_roi,dataframe_long_components,dataframe_long_cross_ic,dataframe_long_cross_roi
from Funciones import columns_SL_roi,columns_coherence_roi,columns_entropy_rois,columns_powers_rois
from Funciones import columns_SL_ic,columns_coherence_ic,columns_entropy_ic,columns_powers_ic
warnings.filterwarnings("ignore")

def graphics(data,type,path,name_band,id,id2,A,B,space,id_cross=None,num_columns=4,save=True,plot=True):
    '''Function to make graphs of the given data '''
    max=data[type].max()
    sns.set(rc={'figure.figsize':(15,12)})
    sns.set_theme(style="white")
    if id=='IC':
        col='Component'
    else:
        col='ROI'
    axs=sns.catplot(x='group',y=type,data=data,dodge=True, kind="box",col=col,col_wrap=num_columns,palette='winter_r',fliersize=1.5,linewidth=0.5,legend=False)
    #plt.yticks(np.arange(0,round(max),0.1))
    axs.set(xlabel=None)
    axs.set(ylabel=None)
    if id_cross==None:
        axs.fig.suptitle(type+' in '+r'$\bf{'+name_band+r'}$'+ ' in the ICs of normalized data given by the databases')
    else:
        axs.fig.suptitle(type+' in '+id_cross+' of ' +r'$\bf{'+name_band+r'}$'+ ' in the ICs of normalized data given by the databases')
    if id=='IC':
        #axs.add_legend(loc='upper right',bbox_to_anchor=(.59,.95),ncol=4,title="Database")
        axs.fig.subplots_adjust(top=0.85,bottom=0.121, right=0.986,left=0.05, hspace=0.138, wspace=0.062) 
        axs.fig.text(0.5, 0.04, 'Group', ha='center', va='center')
        axs.fig.text(0.01, 0.5,  type, ha='center', va='center',rotation='vertical') 
    else:
        #axs.add_legend(loc='upper right',bbox_to_anchor=(.7,.95),ncol=4,title="Database")
        axs.fig.subplots_adjust(top=0.85,bottom=0.121, right=0.986,left=0.06, hspace=0.138, wspace=0.062) # adjust the Figure in rp
        axs.fig.text(0.5, 0.04, 'Group', ha='center', va='center')
        axs.fig.text(0.015, 0.5,  type, ha='center', va='center',rotation='vertical')
    if plot:
        plt.show()
    if save==True:
        if id_cross==None:
            path_complete='{path}\Graficos_armonizacion_sova_harmo\Graficos_{type}\{id}\{name_band}_{type}_{id}_{id2}_{space}_{group}.png'.format(path=path,name_band=name_band,id=id,id2=id2,type=type,group=str(A+B),space=space)  
        else:
            path_complete='{path}\Graficos_armonizacion_sova_harmo\Graficos_{type}\{id}\{name_band}_{id_cross}_{type}_{id}_{id2}_{space}_{group}.png'.format(path=path,name_band=name_band,id=id,id2=id2,type=type,id_cross=id_cross,group=str(A+B),space=space)
        plt.savefig(path_complete)
    plt.close()
    return path_complete

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
        columns = [columns_powers_ic,columns_SL_ic,columns_coherence_ic,columns_entropy_ic,columns_All[256:]]
    elif space == 'roi':
        columns = [columns_powers_rois,columns_SL_roi,columns_coherence_roi,columns_entropy_rois,columns_All[128:]]

    for i,value in enumerate([data_roi_sova,data_roi_harmo]):
        d_B_roi=value
        if i==0:
            label="_sova"
        else:
            label="_harmonized"
        #New dataframes from ROIs
        #Dataframes are saved by ROI and components for graphics.
        if space == 'roi':
            #Power
            dataframe_long_roi(d_B_roi,'Power',columns=columns[0],name="data_long_power_{space}_without_oitliers{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #SL
            dataframe_long_roi(d_B_roi,type='SL',columns=columns[1],name="data_long_sl_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #Coherencia
            dataframe_long_roi(d_B_roi,type='Coherence',columns=columns[2],name="data_long_coherence_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #Entropia
            dataframe_long_roi(d_B_roi,type='Entropy',columns=columns[3],name="data_long_entropy_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #Cross frequency
            dataframe_long_cross_roi(d_B_roi,type='Cross Frequency',columns=columns[4],name="data_long_crossfreq_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        else:
            #Power
            dataframe_long_components(d_B_roi,'Power',columns=columns[0],name="data_long_power_{space}_without_oitliers{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #SL
            dataframe_long_components(d_B_roi,type='SL',columns=columns[1],name="data_long_sl_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #Coherencia
            dataframe_long_components(d_B_roi,type='Coherence',columns=columns[2],name="data_long_coherence_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #Entropia
            dataframe_long_components(d_B_roi,type='Entropy',columns=columns[3],name="data_long_entropy_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
            #Cross frequency
            dataframe_long_cross_ic(d_B_roi,type='Cross Frequency',columns=columns[4],name="data_long_crossfreq_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)

    #labels=['_harmonized']
    labels=['_sova','_harmonized']
    for label in labels:
        data_p_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_power_{space}_without_oitliers{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
        data_sl_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_sl_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
        data_c_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_coherence_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
        data_e_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_entropy_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
        data_cr_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_crossfreq_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space)) 
        datos_roi={'Power':data_p_roi,'SL':data_sl_roi,'Coherence':data_c_roi,'Entropy':data_e_roi,'Cross Frequency':data_cr_roi}
        bands= data_sl_roi['Band'].unique()
        bandsm= data_cr_roi['M_Band'].unique()  

        filename = r"{path}\Graficos_armonizacion_sova_harmo\tabla_effectsize{label}.xlsx".format(path=path,label=label)
        filename = r"{path}\Graficos_armonizacion_sova_harmo\tabla_effectsize_{space}_{group}{label}.xlsx".format(path=path,label=label,group=str(A+B),space=space)
        writer = pd.ExcelWriter(filename,mode='w')
        
        for metric in datos_roi.keys():
            d_roi=datos_roi[metric]
            if space == 'roi':
                table=stats_pair(d_roi,metric,'ROI',A,B)
            else:
                table=stats_pair(d_roi,metric,'Component',A,B)
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

        #for metric in datos_roi.keys():
        #    for band in bands:
        #        d_roi=datos_roi[metric]
        #        d_banda_roi=d_roi[d_roi['Band']==band]
        #        if metric!='Cross Frequency':  
        #            print(str(band)+' '+str(metric)) 
        #            if space == 'roi':
        #                path_roi=graphics(d_banda_roi,metric,path,band,'ROI',label,A=A,B=B,space=space,num_columns=2,save=True,plot=False)
        #            else:
        #                path_roi=graphics(d_banda_roi,metric,path,band,'IC',label,A=A,B=B,space=space,num_columns=2,save=True,plot=False)
        #    
        #        else:
        #            #pass
        #            for bandm in list(d_banda_roi['M_Band'].unique()):  
        #                print(str(band)+' '+str(metric)+' '+str(bandm)) 
        #                if d_banda_roi[d_banda_roi['M_Band']==bandm]['Cross Frequency'].iloc[0]!=None:
        #                    if space == 'roi':
        #                        path_roi=graphics(d_banda_roi[d_banda_roi['M_Band']==bandm],'Cross Frequency',path,band,'ROI',label,A=A,B=B,space=space,id_cross=bandm,num_columns=2,save=True,plot=False)
        #                    else:
        #                        path_roi=graphics(d_banda_roi[d_banda_roi['M_Band']==bandm],'Cross Frequency',path,band,'IC',label,A=A,B=B,space=space,id_cross=bandm,num_columns=2,save=True,plot=False)


def main():
    #path=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' #Cambia dependieron de quien lo corra
    path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD'
    
    # IC
    data_ic_sova_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_G2G1.feather'.format(path=path))
    data_ic_harmo_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_G2G1.feather'.format(path=path))
    #graph_harmonize(path,data_ic_sova_G1G2,data_ic_harmo_G1G2,'ic','G1','G2')
    print(2)
    print('Data_complete_ic_sovaharmony_G1\n','Data_complete_ic_neuroHarmonize_G1')
    data_ic_sova_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_G1.feather'.format(path=path))
    data_ic_harmo_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_G1.feather'.format(path=path))
    #graph_harmonize(path,data_ic_sova_CTR,data_ic_harmo_CTR,'ic','G1','Control')
    print(4)
    print('Data_complete_ic_sovaharmony_DTA\n','Data_complete_ic_neuroHarmonize_DTA')
    data_ic_sova_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_DTA.feather'.format(path=path))
    data_ic_harmo_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_DTA.feather'.format(path=path))
    #graph_harmonize(path,data_ic_sova_DTA,data_ic_harmo_DTA,'ic','DTA','Control')
    print(6)
    #data_ic_sova_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_DCL.feather'.format(path=path))
    #data_ic_harmo_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_DCL.feather'.format(path=path))
    #graph_harmonize(path,data_ic_sova_DCL,data_ic_harmo_DCL,'ic','DCL','Control')

    # ROI
    print('Data_complete_roi_sovaharmony_G2G1\n','Data_complete_roi_neuroHarmonize_G2G1')
    data_roi_sova_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_G2G1.feather'.format(path=path))
    data_roi_harmo_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_G2G1.feather'.format(path=path))
    #graph_harmonize(path,data_roi_sova_G1G2,data_roi_harmo_G1G2,'roi','G1','G2')
    print(8)
    print('Data_complete_roi_sovaharmony_G1\n','Data_complete_roi_neuroHarmonize_G1')
    data_roi_sova_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_G1.feather'.format(path=path))
    data_roi_harmo_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_G1.feather'.format(path=path))
    graph_harmonize(path,data_roi_sova_CTR,data_roi_harmo_CTR,'roi','G1','Control')
    print(10)
    print('Data_complete_roi_sovaharmony_DTA\n','Data_complete_roi_neuroHarmonize_DTA')
    data_roi_sova_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_DTA.feather'.format(path=path))
    data_roi_harmo_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_DTA.feather'.format(path=path))
    graph_harmonize(path,data_roi_sova_DTA,data_roi_harmo_DTA,'roi','DTA','Control')
    print(12)
    #data_roi_sova_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_DCL.feather'.format(path=path))
    #data_roi_harmo_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_DCL.feather'.format(path=path))
    #graph_harmonize(path,data_roi_sova_DCL,data_roi_harmo_DCL,'roi','DCL','Control')


if __name__ == "__main__":
    main()