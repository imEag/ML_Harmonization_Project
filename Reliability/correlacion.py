import numpy as np
import pandas as pd 
import collections
import scipy.io
from tokenize import group
import pingouin as pg
from scipy import stats
import dataframe_image as dfi
import warnings
import seaborn as sns
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

datosic=pd.read_feather(r"Reliability\Data_csv_Powers_Componentes-Channels\data_ic_without_age.feather") 
datosrois=pd.read_feather(r"Reliability\Data_csv_Powers_Componentes-Channels\data_roi_without_age.feather") 
#path=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' 
path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_54x10' 
N_BIO=pd.read_excel('{path}\Datosparaorganizardataframes\Demograficosbiomarcadores.xlsx'.format(path=path))
N_BIO = N_BIO.rename(columns={'Codigo':'Subject','Edad en la visita':'age','Sexo':'sex','Escolaridad':'education','Visita':'Session'})
N_BIO=N_BIO.drop(['MMSE', 'F', 'A', 'S'], axis=1)
N_BIO['Subject']=N_BIO['Subject'].replace({'_':''}, regex=True)#Quito el _ y lo reemplazo con '' 
datosic=pd.merge(datosic,N_BIO)
datosrois=pd.merge(datosrois,N_BIO)
datosrois=datosrois.drop(columns=['index','level_0'])
datosrois=datosrois.groupby(['Stage','Roi','Bands','Subject','Session']).mean().reset_index()

datosic['log(Age)'] = np.log2(datosic['age'])
datosrois['log(Age)'] = np.log2(datosrois['age'])

est_ic=datosic.groupby(['Components','Bands']).apply(lambda datosic:pg.rm_corr(data=datosic, x='age', y='Powers', subject='Subject'))
est_roi=datosrois.groupby(['Roi','Bands']).apply(lambda datosic:pg.rm_corr(data=datosic, x='age', y='Powers', subject='Subject'))

p_ic=est_ic[est_ic['pval'] < 0.05]
p_roi=est_roi[est_roi['pval'] < 0.05]

est_ic_log=datosic.groupby(['Components','Bands']).apply(lambda datosic:pg.rm_corr(data=datosic, x='log(Age)', y='Powers', subject='Subject'))
est_roi_log=datosrois.groupby(['Roi','Bands']).apply(lambda datosic:pg.rm_corr(data=datosic, x='log(Age)', y='Powers', subject='Subject'))

p_ic_l=est_ic_log[est_ic_log['pval'] < 0.05]
p_roi_l=est_roi_log[est_roi_log['pval'] < 0.05]

# writer = pd.ExcelWriter('C:\\Users\\valec\\OneDrive - Universidad de Antioquia\\Articulo análisis longitudinal\\Resultados correlacion\\correlacion.xlsx')
# est_ic.to_excel(writer,sheet_name='Component')
# p_ic.to_excel(writer,sheet_name='Component',startcol=10)
# est_ic_log.to_excel(writer,sheet_name='Component(log(age))')
# p_ic_l.to_excel(writer,sheet_name='Component(log(age))',startcol=10)
# est_roi.to_excel(writer,sheet_name='ROI')
# p_roi.to_excel(writer,sheet_name='ROI',startcol=10)
# est_roi_log.to_excel(writer,sheet_name='ROI(log(age))')
# p_roi_l.to_excel(writer,sheet_name='ROI(log(age))',startcol=10)
# writer.save()
# writer.close() 

path_im=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados correlacion\imagenes'

components=datosic['Components'].unique()
bandas=datosic['Bands'].unique()
rois=datosrois['Roi'].unique()

def plot_correlacion(data,x,path,title):
    max=data[x].max()
    min=data[x].min()
    sns.set(rc={'figure.figsize':(15,12)})
    sns.set_theme(style="white")
    g = pg.plot_rm_corr(data=data, x=x, y='Powers', subject='Subject',kwargs_facetgrid=dict(height=8, aspect=1.5,
                                          palette='Spectral'))
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.fig.subplots_adjust(top=0.885,bottom=0.121, right=0.956,left=0.125, hspace=0.138, wspace=0.062) # adjust the Figure in rp
    if x=='age':
        x='Age'
    g.fig.suptitle(title,fontsize=25)
    g.fig.text(0.5, 0.04, x, ha='center', va='center',fontsize=25)
    g.fig.text(0.03, 0.5, 'Relative power', ha='center', va='center',rotation='vertical',fontsize=25)
    plt.xticks(np.arange(20,65,5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(path)
    plt.close()

for band in bandas:
    #Para ROI
    for roi in rois:
        data_roi_band=datosrois[datosrois.Roi.isin([roi])&datosrois.Bands.isin([band])]
        path_r='{path}\{roi}_{band}_age.png'.format(path=path_im,roi=roi,band=band)
        title= 'Repeated measures correlation for \nROI {roi} in the {band} band'.format(roi=roi,band=band)
        plot_correlacion(data_roi_band,'age',path_r,title)
        # path_r='{path}\{roi}_{band}_log(age).png'.format(path=path_im,roi=roi,band=band)
        # title= 'Repeated measures correlation for ROI {roi} in {band} band with log(age)'.format(roi=roi,band=band)
        # plot_correlacion(data_roi_band,'log(Age)',path_r,title)
        print(band,roi)
        #Para componentes
    for comp in components:
        data_ic_band=datosic[datosic.Components.isin([comp])&datosic.Bands.isin([band])]
        path_c='{path}\{comp}_{band}_age.png'.format(path=path_im,comp=comp,band=band)
        title= 'Repeated measures correlation for \ncomponent {comp} in the {band} band'.format(comp=comp,band=band)
        plot_correlacion(data_ic_band,'age',path_c,title)
        # path_c='{path}\{comp}_{band}_log(Age).png'.format(path=path_im,comp=comp,band=band)
        # title= 'Repeated measures correlation for component {comp} in {band} band with log(age)'.format(comp=comp,band=band)
        # plot_correlacion(data_ic_band,'log(Age)',path_c,title)   
    
        
        
print('valelinda')
    


