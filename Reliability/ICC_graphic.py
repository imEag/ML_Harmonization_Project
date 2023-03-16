import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

def barplot_icc_nB_1G(icc_data,x_value,group,plot=False,save=False):
    filter_band=icc_data[icc_data['Group']==group]
    sns.set(font_scale = 0.9)
    sns.set_theme(style="white")
    ax=sns.catplot(x=x_value,y='ICC',data=filter_band,hue='Stage',palette='winter_r',kind='bar',col='Bands',col_wrap=4,legend=False)
    ax.fig.suptitle('ICC(3,k) for '+ x_value +' in frequency bands of '+group+' group')
    #ax.add_legend(loc='upper center',bbox_to_anchor=(.5,0.94),ncol=2)
    ax.fig.subplots_adjust(top=0.9,bottom=0.133, right=0.936,left=0.062, hspace=0.143, wspace=0.11) # adjust the Figure in rp
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.fig.text(0.5, 0.07, x_value, ha='center', va='center')
    ax.fig.text(0.03, 0.5,  'ICC', ha='center', va='center',rotation='vertical')
    if save==True:
        plt.savefig('Reliability\ICC_Graphics\ICC_{name_group}_{tipo}.png'.format(name_group=group,tipo=x_value))
        plt.close()
    if plot:
        plt.show()

def barplot_icc_comp_nG(icc_data,x_value,title,plot=False,save=False):
    sns.set(font_scale = 0.9)
    sns.set_theme(style="white")
    ax=sns.catplot(x=x_value,y='ICC',data=icc_data,hue='Group',palette='winter_r',kind='bar',col='Bands',col_wrap=4,legend=False)
    #ax=sns.catplot(x=x_value,y='ICC',data=icc_data,palette='winter_r',kind='bar',col='Bands',col_wrap=4,legend=False)
    ax.fig.suptitle(title)
    #ax.add_legend(loc='upper center',bbox_to_anchor=(.5,0.94),ncol=2)
    ax.fig.subplots_adjust(top=0.9,bottom=0.133, right=0.936,left=0.062, hspace=0.143, wspace=0.11) # adjust the Figure in rp
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.fig.text(0.5, 0.07, x_value, ha='center', va='center')
    ax.fig.text(0.03, 0.5,  'ICC', ha='center', va='center',rotation='vertical')
    if save==True:
        #plt.savefig('Reliability\ICC_Graphics\ICC_Comparacióngrupos_{tipo}.png'.format(tipo=x_value))
        plt.savefig('Reliability\ICC_Graphics\ICC_todojunto_{tipo}.png'.format(tipo=x_value))
        plt.close()
    if plot:
        plt.show()

def plot_ICC_nG(icc_data,x_value,title,kind,plot=False,save=False):
    sns.set(font_scale = 0.9)
    sns.set_theme(style="white")
    ax=sns.catplot(x=x_value,y='ICC',data=icc_data,hue='Group',palette='winter_r',kind=kind,legend=False)
    #ax=sns.catplot(x=x_value,y='ICC',data=icc_data,palette='winter_r',kind=kind,legend=False)
    ax.fig.suptitle(title)
    #ax.add_legend(loc='upper center',bbox_to_anchor=(.5,0.94),ncol=2,title='Group')
    ax.fig.subplots_adjust(top=0.9,bottom=0.168, right=0.950,left=0.107, hspace=0.143, wspace=0.11)# adjust the Figure in rp
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.fig.text(0.5, 0.07, x_value, ha='center', va='center')
    ax.fig.text(0.03, 0.5,  'ICC', ha='center', va='center',rotation='vertical')
    if save==True:
        #plt.savefig('Reliability\ICC_Graphics\ICC_Comparacióngrupos_sinsepararbandas_{tipo}.png'.format(tipo=x_value))
        plt.savefig('Reliability\ICC_Graphics\ICC_todojunto_sinsepararbandas_{tipo}.png'.format(tipo=x_value))
        plt.close()
    if plot:
        plt.show()


def barplot_icc_bandsx(icc_data,x_value, title,plot=False,save=False):
    sns.set(font_scale = 0.9)
    sns.set_theme(style="white")
    ax=sns.catplot(x='Bands',y='ICC',data=icc_data,hue='Group',palette='winter_r',kind='bar',legend=False)
    #ax=sns.catplot(x='Bands',y='ICC',data=icc_data,palette='winter_r',kind='bar',legend=False)
    ax.fig.suptitle(title)
    #ax.add_legend(loc='upper center',bbox_to_anchor=(.5,0.94),ncol=2, title='Group')
    ax.fig.subplots_adjust(top=0.9,bottom=0.168, right=0.950,left=0.107, hspace=0.143, wspace=0.11) # adjust the Figure in rp
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.fig.text(0.5, 0.07, 'Bands', ha='center', va='center')
    ax.fig.text(0.03, 0.5,  'ICC', ha='center', va='center',rotation='vertical')
    if save==True:
        #plt.savefig('Reliability\ICC_Graphics\ICC_Comparacióngrupos_bandasx_{tipo}.png'.format(tipo=x_value))
        plt.savefig('Reliability\ICC_Graphics\ICC_todojunto_bandasx_{tipo}.png'.format(tipo=x_value))
        plt.close()
    if plot:
        plt.show()
  
icc_data_Roi=pd.read_excel(r'Reliability\ICC_values_csv\ICC_rois_sin_separar.xlsx')
icc_data_Comp=pd.read_excel(r'Reliability\ICC_values_csv\ICC_components_sin_separar.xlsx')

barplot_icc_comp_nG(icc_data_Roi[icc_data_Roi['Stage']=='Preprocessed data'],'ROI','ICC(3,k) for ROIs in frequency bands',plot=True,save=True)
barplot_icc_comp_nG(icc_data_Comp[icc_data_Comp['Stage']=='Preprocessed data'],'Neural Component','ICC(3,k) for ICs in frequency bands', plot=True,save=True)

barplot_icc_bandsx(icc_data_Roi[icc_data_Roi['Stage']=='Preprocessed data'],'ROI','ICC(3,k) for ROIs in frequency bands',plot=True,save=True)
barplot_icc_bandsx(icc_data_Comp[icc_data_Comp['Stage']=='Preprocessed data'],'Neural Component','ICC(3,k) for ICs in frequency bands',plot=True,save=True)

plot_ICC_nG(icc_data_Roi[icc_data_Roi['Stage']=='Preprocessed data'],'ROI','ICC(3,k) for ROIs','bar',plot=True,save=True)
plot_ICC_nG(icc_data_Comp[icc_data_Comp['Stage']=='Preprocessed data'],'Neural Component','ICC(3,k) for ICs','bar',plot=True,save=True)

icc_data_Roi=pd.read_csv(r'Reliability\ICC_values_csv\icc_values_ROIS_G2-CTR.csv',sep=';')
icc_data_Comp=pd.read_csv(r'Reliability\ICC_values_csv\icc_values_Components_G2-CTR.csv',sep=';')

#Para cambiar los nombres de los grupos en las graficas del articulo
# icc_data_Roi['Group'].replace({'G2':'Group 2','CTR':'Group 1'}, inplace=True)
# icc_data_Comp['Group'].replace({'G2':'Group 2','CTR':'Group 1'}, inplace=True)
# barplot_icc_comp_nG(icc_data_Roi[icc_data_Roi['Stage']=='Normalized data'],'Roi','ICC(3,k) for ROIs in frequency bands',plot=True,save=True)
# barplot_icc_comp_nG(icc_data_Comp[icc_data_Comp['Stage']=='Normalized data'],'Components','ICC(3,k) for ICs in frequency bands', plot=True,save=True)

# barplot_icc_bandsx(icc_data_Roi[icc_data_Roi['Stage']=='Normalized data'],'Roi','ICC3k for ROIs in frequency bands',plot=True,save=True)
# barplot_icc_bandsx(icc_data_Comp[icc_data_Comp['Stage']=='Normalized data'],'Components','ICC(3,k) for ICs in frequency bands',plot=True,save=True)

barplot_icc_nB_1G(icc_data_Roi,'Roi','G2',plot=True,save=True)
barplot_icc_nB_1G(icc_data_Roi,'Roi','CTR',plot=True,save=True)
barplot_icc_nB_1G(icc_data_Comp,'Components','G2',plot=True,save=True)
barplot_icc_nB_1G(icc_data_Comp,'Components','CTR',plot=True,save=True)

def icc_mean(data):
    Stage=data['Stage'].unique()
    bands=data['Bands'].unique()
    for st in Stage:
        print(st+':')
        d_stage=data[data['Stage']==st]
        print('Promedio general: '+str(d_stage['ICC'].mean())+' ± ' +str(d_stage['ICC'].std())+'\n')
        for band in bands:
            d_band=d_stage[d_stage['Bands']==band]
            print('Promedio '+band+': '+str(d_band['ICC'].mean())+' ± ' +str(d_band['ICC'].std()))
        print('\n')

print('Promedios por bandas de ROIs\n')
icc_mean(icc_data_Roi)
print('Promedios por bandas de Componentes\n')
icc_mean(icc_data_Comp)