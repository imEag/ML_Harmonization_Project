"""
Some miscellaneous utilities for prepy.
@revision: Yorguin Mantilla

"""

# Imports
import numpy as np
import logging
from datetime import datetime

import scipy.signal as signal
import matplotlib.pyplot as plt
import mne.io
import json
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
#from matplotlib.colors import DivergingNorm
from matplotlib.colors import TwoSlopeNorm
from scipy.io import loadmat
import os
import copy
from mne.viz import plot_topomap
from mne.utils import (_clean_names, _time_mask, verbose, logger, warn, fill_doc,
                     _validate_type, _check_sphere, _check_option, _is_numeric)
from mne.viz.utils import  (_setup_vmin_vmax, _prepare_trellis,
                    _check_delayed_ssp, _draw_proj_checkbox, figure_nobar,
                    plt_show, _process_times, DraggableColorbar,
                    _validate_if_list_of_axes, _setup_cmap, _check_time_unit)
from mne.viz.topomap import _prepare_topomap_plot,_make_head_outlines,_add_colorbar,_hide_frame
#from mne.channels.channels import _get_ch_type
from mne.channels.layout import (
    _find_topomap_coords, find_layout, _pair_grad_sensors, _merge_ch_data)

#_picks_to_idx _get_ch_type _setup_cmap
#_prepare_topomap_plot
#_make_head_outlines
#_prepare_trellis
#_merge_ch_data
#_setup_vmin_vmax
#_add_colorbar


# def super_topomap(cols = 5):

#     rows = A.shape[0]//cols
#     res = A.shape[0]%cols
#     superfig = plt.figure()
#     A,W,ch_names = get_spatial_filter()
#     for i in range(A.shape[1]):
#         if res == 0:
#             fig1,ax = plt.subplots(rows,cols,i+1)
#         else:
#             fig1,ax = plt.subplots(rows+1,cols,i+1)

#         fig = our_topomap('comp'+str(i))
#         x,y = fig.get_data()
#         plt.plot(x,y)
#         fig.add_axes(ax)

#     return superfig
def cfg_logger(log_path):
    """Configures the logger of the pipeline.
    Parameters
    ----------
    log_path : string
        Directory of the log file without filename.
    Returns
    -------
    dalogger : logging.Logger instance
        The logger object to be used by the pipeline.
    currentDT : instance of datetime.datetime
        The current date and time.
    Examples
    --------
    >>> log, date = cfg_logger(log_path)
    """
    for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    currentDT = datetime.now()
    currentDT.strftime("%Y-%m-%d %H:%M:%S")
    log_name = os.path.join(log_path, 'sovaflow__' + currentDT.strftime("%Y-%m-%d__%H_%M_%S") + '.log')
    logging.basicConfig(filename= log_name, level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    dalogger=logging.getLogger(__name__)
    return dalogger, currentDT

def our_topomap(comp=None,colormap='seismic'):
    """Gets the topomap of a component or a set
    of components given by the default spatial
    filter.
    Parameters
    ----------
    comp : string of list of string
        Components required in the following notation
        compX where X is the number of the component
        counting from 1.
    colormap: string, default='seismic'
        String of the colormap to use to plot
        the topomap.
        See matplotlib colormaps documentation.
    Returns
    -------
    figs : matplotlib.pyplot.figure or list of it
        The figure(s) with the topomap required
    """

    ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    ch_names = [chn_name_mapping(x) for x in ch_names]
    dummy = np.zeros((len(ch_names),1))
    sfreq = 250
    montage_kind = 'standard_1020'
    raw = createRaw(dummy,sfreq,'eeg',ch_names)
    raw.set_montage(montage_kind)
    A,W,ch_names = get_spatial_filter()
    if comp is not None and type(comp) != list:
        label = comp
        comp = comp.replace('comp','')
        comp = int(comp)
        A_ = np.expand_dims(A[:,comp-1],axis=-1)
        W_ = np.expand_dims(W[comp-1,:],axis=0)
        figs = topomap(A_,W_,A_.shape[0],raw.info,cmap=colormap,show=False,labels=as_list(label))
    elif type(comp) == list :
        if len(comp) == 0:
            return None
        labels = comp
        comp = [int(c.replace('comp','')) -1  for c in labels]
        if len(comp)==1:
            comp = comp[0]
            A_ = np.expand_dims(A[:,comp-1],axis=-1)
            W_ = np.expand_dims(W[comp-1,:],axis=0)
        else:
            A_ = A[:,comp]
            W_ = W[comp,:]
        figs = topomap(A_,W_,A_.shape[0],raw.info,cmap=colormap,show=False,labels=as_list(labels))
    else:
        labels = ['comp'+str(i) for i in range(A.shape[1])]
        figs = topomap(A,W,A.shape[0],raw.info,cmap=colormap,show=False,labels=labels)

    return figs




def single_topomap(A,ch_names,info=None,show_names=False,cmap='seismic',title=None,label=None,show=False):
    """Gets the topomap of all the components in
    a given spatial filter.
    Parameters
    ----------
    A : np.ndarray
        The mixing matrix of the spatial filter.
        Slice the matrix to select a single component.
    ch_names : list of str, default=None
        The names of the sensors in the same order
        as A. If None the names the ones of the
        default montage of the software.
    info : mne.Info instance, default=None
        The metadata of the eeg recording needed
        to create a mne.io.raw objetct. If 
        None it defaults to a 10-20 montage.
    show_names: bool, default=False
        Whether to show the names of the sensors
        on the topomap or not.
    cmap: string, default='seismic'
        String of the colormap to use to plot
        the topomap.
        See matplotlib colormaps documentation.
    title: string, default=None
        The desired title of the topomap.
    label: string, default=None
        The label printed above of each
        component in the topomap.
    show: bool, default=False
        Whether to plot or not.
    Returns
    -------
    fig : matplotlib.pyplot.figure or list of it
        The figure(s) with the topomap required


    Example:
    >>>import models.eegflow.utils as us
    >>>import matplotlib.pyplot as plt
    >>>ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    >>>ch_names = [us.chn_name_mapping(x) for x in ch_names]
    >>>dummy = np.zeros((len(ch_names),1000))
    >>>sfreq = 250
    >>>montage_kind = 'standard_1020'
    >>>raw = us.createRaw(dummy,sfreq,'eeg',ch_names)
    >>>raw.set_montage(montage_kind)
    >>>A,W,ch_names = us.get_spatial_filter()
    >>>comp = 25
    >>>fig3 = us.single_topomap(A[:,comp-1],show=True,label='1')
    >>>plt.show()
    """
    if info is None:
        info = generate_info(ch_names)
    fig, ax = plt.subplots()
    im, _ =  plot_topomap(A,info,cmap=cmap,axes=ax,names=info.ch_names,show=show)
    fig.colorbar(im, ax=ax)
    if title is not None:
        fig.suptitle(title)
    if label is not None:
        fig.axes[0].set_title(label)
    if show:
        plt.show()
    return fig

def generate_info(ch_names):
    dummy = np.zeros((len(ch_names),10)) # this will be about 600kb
    sfreq = 250
    montage_kind = 'standard_1005'
    
    try:
        raw.set_montage(montage_kind)
    except:
        raw = createRaw(dummy,sfreq,'eeg',[chn_name_mapping(x) for x in ch_names])
        raw.set_montage(montage_kind)
    info = raw.info
    return info

def topomap(A,W,ch_names=None,info=None,cmap='plasma',show=False,title=None,labels=None,ncols=None):
    """Gets the topomap of all the components in
    a given spatial filter.
    Parameters
    ----------
    A : np.ndarray
        The mixing matrix of the spatial filter.
    W : np.ndarray
        The unmixing matrix of the spatial filter.
    ch_names : names of the channels
    info : mne.Info instance
        The metadata of the eeg recording needed
        to create a mne.io.raw objetct.
    cmap: string, default='plasma'
        String of the colormap to use to plot
        the topomap.
        See matplotlib colormaps documentation.
    show: bool, default=False
        Whether to plot or not.
    title: string, default=None
        The desired title of the topomap.
    labels: list of strings, default=None
        The labels of each of the components.
        If None components will be numbered
        from 1 and labeled as compX.
    Returns
    -------
    figs : matplotlib.pyplot.figure or list of it
        The figure(s) with the topomap required

    Note:
        You may or may not need to tranpose A and W,
        just check which gets the obvious correct result
    Example:
    >>>import models.eegflow.utils as us
    >>>import matplotlib.pyplot as plt
    >>>ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    >>>ch_names = [us.chn_name_mapping(x) for x in ch_names]
    >>>dummy = np.zeros((len(ch_names),1000))
    >>>sfreq = 250
    >>>montage_kind = 'standard_1020'
    >>>raw = us.createRaw(dummy,sfreq,'eeg',ch_names)
    >>>raw.set_montage(montage_kind)
    >>>A,W,ch_names = us.get_spatial_filter()
    >>>comp = 25
    >>>A_ = np.expand_dims(A[:,comp-1],axis=-1)
    >>>W_ = np.expand_dims(W[comp-1,:],axis=0)
    >>>figSingle = us.topomap(A_,W_,A_.shape[0],raw.info,cmap='seismic',show=True)
    >>>figMany = us.topomap(A,W,A.shape[0],raw.info,cmap='seismic',show=True)
    """
    if info is None:
        info = generate_info(ch_names)

    ica = ICA(random_state=97, method = 'fastica')
    ica.info = info
    ica.n_components_= A.shape[0]
    ica.unmixing_matrix_ = W
    ica.pca_components_ = np.eye(A.shape[0]) #transformer.whitening_#np.linalg.pinv(transformer.whitening_)
    ica.mixing_matrix_ = A
    ica._update_ica_names()

    if labels is None:
        labels = [str('comp'+str(i+1)) for i in range(A.shape[1])]


    ica._ica_names = labels
    #figs = ica.plot_components(cmap=cmap,show=False)
    ncols = int(np.floor(np.sqrt(A.shape[1])))
    
    figs = plot_ica_components(ica,p=A.shape[1],ncols=ncols,cmap=cmap)
    if title is not None:
        for fig in figs:
            fig.suptitle(title)
    else:
        for fig in figs:
            fig.suptitle('')

    # Reverse labels and pop consecutively
    
    # labels.reverse()
    # for fig in figs:
    #     for ax in fig.axes:
    #         #ax.set_label(labels.pop())
    #         idx0 = figs.index(fig)
    #         idx = fig.axes.index(ax)
    #         figs[idx0].axes[idx].set_title(labels.pop(),y=1.08)
    #     fig.subplots_adjust(top=0.88, bottom=0.)
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     fig.canvas.draw()
    if show:
        plt.show()
    return figs

def createRaw(signal,sfreq,ch_types='eeg',ch_names=None):
    """Creates an mne.io.raw object given some parameters.
    Parameters
    ----------
    signal : np.ndarray
        Signal with the shape of (channels,samples)
    sfreq : float
        The sampling frequency of the signal.
    ch_types: str or list of strings, default='eeg'
        The types of all or each of the channels.
        See mne.create_info documentation.
    ch_names : list of str, default=None
        The names of the sensors/channels.
    Returns
    -------
    signals : mne.io.raw instance of the signal
    """
    if signal.ndim == 1:
        signal = np.reshape(signal,(1,len(signal)),order='F')
    if ch_names is None:
        ch_names = signal.shape[0]
    signals = mne.io.RawArray(signal, mne.create_info(ch_names=ch_names, ch_types=ch_types,sfreq=sfreq),verbose=False)
    return signals

def scrollPlot(signal,sfreq,block=True,ch_types='eeg',scale='auto',returnSig = False,color='k',title=None,show=True,remove_dc=False):
    """Makes an mne scroll plot for a given signal.
    Parameters
    ----------
    signal : np.ndarray
        Signal with the shape of (channels,samples)
    sfreq : float
        The sampling frequency of the signal.
    block : bool, default=True
        Whether the plot should or should not
        block the execution of the program.
    ch_types: str or list of strings, default='eeg'
        The types of all or each of the channels.
        See mne.create_info documentation.
    scale : string, default='auto'
        The default scale applied to all of the
        signals. See mne.viz.plot_raw documentation.
        If None it will use the default scales of that
        documentation.
    returnSig : bool, default=False
        Whether to also return the signal or not
        (besides the plot figure)
    ch_names : list of str, default=False
        The names of the sensors/channels.
    color : char, default='k'
        The color for ALL types of signals in the plot.
        See mne.viz.plot_raw documentation.
    title : string, default=None
        Title of the plot windows
    show: bool, default=True
        Whether to plot or not.
    remove_dc: bool, default=False
        Whether to remove the dc value
        of the signals when plotting.
        (Helps them to stay around a center line).
    Returns
    -------
    
    if returnSig is True:
        signals,fig
    else:
        fig

    signals : mne.io.raw instance of the signal
    fig : matplotlib.pyplot figure instance of the signal plot

    Notes:
    If you change the sampling frequency you can see it more detailed in time since
    
    It has a constant width for seconds.
    A good idea is to change it to a divisor or a multiple of the real sfreq

    DO NOTE sfreq changes subsecuent sfreq of the object so if you apply 
    a filter after it will be wrong if you dont use the real sfreq.
    """
    scalings = dict(mag=scale, grad=scale, eeg=scale, eog=scale, ecg=scale,emg=scale, ref_meg=scale, misc=scale, stim=scale,      resp=scale, chpi=scale, whitened=scale)
    if scale is None:
        scalings = None
    colors = dict(mag=color, grad=color, eeg=color, eog=color, ecg=color, emg=color, ref_meg=color, misc=color, stim=color, resp=color, chpi=color)
    signals = createRaw(signal,sfreq=sfreq,ch_types=ch_types)
    #signals = mne.io.RawArray(signal, mne.create_info(ch_names=signal.shape[0], ch_types=ch_types,sfreq=sfreq))
    fig = signals.plot(block=block,scalings=scalings,color=colors,title=None,show=show,remove_dc=remove_dc)
    if(returnSig):
        return signals,fig
    else:   
        return fig

def get_harmonic_signal(freqs,weights,phases,s_rate=1000,start=0,stop=10,withtime=False,time=None):
    """
    Function that returns a harmonic signal given its components.
    Parameters:
        freqs: list of floats
                the frequencies [Hz] of the components
        weights: list of floats
                the weights  of the components
        phases: list of floats
                the phases of the components
        s_rate: float, default=1000
                the sampling rate
        start: float, default=0
                start time
        stop: float, default=10
                stop time
        withtime: bool, default=False
                Whether to return or not
                the time vector of the signal.
        time : np.ndarray, default=None
                1d time vector. If None
                it is created from the start,
                stop and s_rate parameters.
    Returns:
        if withtime:
            time,harmonic_signal
        else:
            harmonic_signal
        harmonic_signal: numpy.ndarray
                the harmonic signal generated
        
    Example:
    >>>s_rate = 1000
    >>>length = s_rate
    >>>freqs_sin = np.array([10,10])
    >>>weights_sin = np.array([1,1])
    >>>phases_sin = np.array([0,90])
    >>>start = 0
    >>>stop = 10
    >>>sin = get_harmonic_signal(freqs_sin,weights_sin,phases_sin,s_rate,start,stop)
    """
    freqs = as_list(freqs)
    weights = as_list(weights)
    phases = as_list(phases)
    if time is None:
        time = np.arange(start, stop, 1/s_rate)
    time = np.squeeze(time)
    harmonic_signal = np.zeros(time.shape[0])
    rad_phases = [phase * np.pi/180 for phase in phases]
    rad_freqs = [2 * np.pi * freq for freq in freqs]
    sinusoid = np.zeros((len(freqs),time.shape[0]))
    for i in range(0,len(freqs)):
        sinusoid[i,:] = weights[i] * np.sin(rad_freqs[i] * time + rad_phases[i])

    harmonic_signal = np.sum(sinusoid,axis=0)
    if withtime == True:
        return time,harmonic_signal
    else:
        return harmonic_signal

def time_compare(timeA,timeB,label_A='python',label_B='matlab'):
    """
    Compares two times and says which is faster. Designed for
    comparing python and matlab functions.
    Parameters:
        timeA : float
        timeB : float
            The times to compare.
        label_A : string, default='python'
        label_B : string, default='matlab'
            The labels associated with those times.
    Returns:
        None
    """

    timing = get_differences(timeA,timeB)
    print('Time Difference Absolute: '+str(timing[0])+ 's')
    print('Time Difference Relative: '+str(timing[1]))
    if timing[0] < 0:
        print(label_A + ' is faster')
    elif timing[0] > 0:
        print(label_B + ' is faster')
    else:
        print('Equally Fast, this is rare!')

def get_differences(data_A,data_B,hor_axis=None,unsigned = True,title='', marker='',hor_axis_label='samples',plot = False,label_B='data_B',label_A='data_A',max_errors = 3,relative_method='offset',max_tolerance=1,width=100,absolute_method='difference'):
    """
    Auxiliar function to show differences between two sources.

    Parameters:
        data_A: numpy.ndarray
            from source A
        data_B: numpy.ndarray
            from source B
        hor_axis: numpy.ndarray
            horizontal axis for comparison, defaults to None
        unsigned: bool
            Whether the difference is unsigned or not.
            Basically if and absolute value operation is applied
            to it.
        title: string
            title to add to plots
        marker: string
            marker for graphs
        hor_axis_label: string
            label for x axis, defaults to'samples'
        plot: boolean
                Whether to plot or not the errors.
        label_B: string, default='data_B'
        label_A: string, default='data_A'
            labels identifying each source.
        max_errors: natural number
                Max number of errors to plot
        relative_method: string : default: 'offset'
            One of  'offset'| 'rpd'| 'true'| 'square'| 'max'| 'mean'| 'epsilon'
            See code formulas for details.
        max_tolerance: 0<=float<=1, default=1
            Percentage of the greatest difference that will be
            considered as an offeding difference.
            ie = 0.7 for 70%
        width : int, default = 100
            width (in samples) of the plot (which is centered at the middle)
            thus the plot will be from ``center - width/2`` to ``center + width/2`` 
        absolute_method: string, 'square'|'difference'
            controls whether the difference is squared ('square') or not ('absolute')
            note that 'absolute' has nothing to do with the sign of the difference,
            that is controlled by the ``unsigned`` parameter
        
    Returns:
        diff,diff_rel : numpy.ndarray tuple
            absolute,relative differences
            differences between A and B, relative is to B in percentage
    """

    # If you remove the abs you get some other interesting graphs
#    data_A = np.squeeze(data_A)
#    data_B = np.squeeze(data_B) , no because this will also work for matrices
    if absolute_method == 'difference':
        diff = data_A - data_B
    elif absolute_method == 'square':
        diff = data_A - data_B
        diff = diff**2

    # Offset the data to avoid divisions by zero
    if relative_method == 'offset':
        max = np.amax([np.abs(data_A),np.abs(data_B)])
        offset = 2
        off_B = data_B + max + offset
        off_A = data_A + max + offset
        diff_off = off_A - off_B
        diff_rel = np.divide(diff_off,np.abs(off_B))
        diff_rel = diff_rel*100
    elif relative_method == 'rpd': #Relative Percent Difference , https://stats.stackexchange.com/questions/86708/how-to-calculate-relative-error-when-the-true-value-is-zero
        diff_rel = 2.0*(np.divide(data_A-data_B,np.abs(data_A)+np.abs(data_B))) # between 2 and -2 #*100
    elif relative_method == 'true':
        diff_rel = np.divide(data_A-data_B,np.abs(data_B)) 
        diff_rel = diff_rel*100
    elif relative_method == 'square':
        diff_rel = np.divide(data_A-data_B,np.max(abs(data_B), axis=None)) #*100? # wouldnt the mean be better and multiply by 50?
        diff_rel = diff_rel**2
    elif relative_method == 'max':
        diff_rel = np.divide(data_A-data_B,np.max(abs(data_B), axis=None))*100 # wouldnt the mean be better and multiply by 50?
    elif relative_method == 'mean':
        diff_rel = np.divide(data_A-data_B,np.mean(abs(data_B), axis=None))*50 # wouldnt the mean be better and multiply by 50?
    elif relative_method == 'epsilon':
        diff_rel = np.divide(data_A-data_B,np.max(np.finfo(float).eps,np.max(np.abs(data_B))))
        diff_rel *= 100
        

    if (unsigned):
        diff = np.abs(diff)
        diff_rel = np.abs(diff_rel)

    if plot == True:
        fig, axis = plt.subplots(2,1,constrained_layout=False)
        fig.suptitle('Error for ' + title)


        if(hor_axis is None):
            try:
                hor_axis = range(len(diff))
            except:
                hor_axis = None
            
        plt.subplot(211)
        if hor_axis is None:
            if marker == '':
                plt.plot(diff)
            else:
                plt.plot(diff,marker)    
        else:
            if marker == '':
                plt.plot(hor_axis,diff)
            else:
                plt.plot(hor_axis,diff,marker)

        plt.title(title + ' absolute')
        plt.ylabel('absolute error')
        plt.xlabel(hor_axis_label)
            
        plt.subplot(212)

        if hor_axis is None:
            if marker == '':
                plt.plot(diff_rel)
            else:
                plt.plot(diff_rel,marker)    
        else:
            if marker == '':
                plt.plot(hor_axis,diff_rel)
            else:
                plt.plot(hor_axis,diff_rel,marker)

            
        plt.title(title + ' relative')
        plt.ylabel('relative percentage error')
        plt.xlabel(hor_axis_label)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        # Plot signals in the neighborhood of the greatest absolute error
        max_abs = np.amax(abs(diff))
        if max_abs != 0:
            max_indexes = np.argwhere(abs(diff) >= max_tolerance*np.amax(abs(diff))) # this could return a lot of indexes if periodic
            
            try:
                max_indexes = list(max_indexes)
                #max_indexes = max_indexes[0] # If it returns too many indexes you could try this
                max_indexes = max_indexes[:max_errors]
            except:
                print("Error trying to obtain indexes")
            
            
            signals = [data_B,data_A]
            labels = [label_B,label_A]
            markers = ['r-','b-']
            
            
            # To also plot errors
            #signals_errors = [diff,diff_rel]
            #labels_errors = ['absolute error','relative error']
            
            plot_in_neighborhoods(None,signals,labels,markers,max_indexes,title=title + ' Greatest absolute error',width=width)
            # To also plot errors
            #plot_in_neighborhoods(None,signals_errors,labels_errors,markers,max_indexes,title=title + ' Greatest absolute error')

        # Plot signals in the neighborhood of the greatest relative error
        max_rel = np.amax(abs(diff_rel))
        if max_rel != 0:
            max_indexes = np.argwhere(abs(diff_rel) >= max_tolerance*np.amax(abs(diff_rel))) #this could return a lof of indexes if periodic
            
            try:
                max_indexes = list(max_indexes)
                #max_indexes = max_indexes[0] # If it returns too many indexes you could try this
                max_indexes = max_indexes[:max_errors]
            except:
                print("Error trying to obtain indexes")

            signals = [data_B,data_A]
            
            labels = [label_B,label_A]
            markers = ['r-','b-']
            
            #signals_errors = [diff,diff_rel]
            #labels_errors = ['absolute error','relative error']
            
            plot_in_neighborhoods(None,signals,labels,markers,max_indexes,title=title + ' Greatest relative error',width=width)
            # To also plot errors
            #plot_in_neighborhoods(None,signals_errors,labels_errors,markers,max_indexes,title=title + ' Greatest relative error')
        
    return diff,diff_rel

def relative_eeg_power(input_signal, s_rate, segment_length, overlap_length):
    """
    Function to calculate relative power in each eeg band.

    Parameters:
        input_signal: numpy.ndarray
                    Series of measurement values.
        s_rate: float
                sampling rate in Hz.
        segment_length: int
                    Length of each segment.

        overlap_length: int
                    Number of points to overlap between segments.
    
    Returns:
        numpy.ndarray tuple with relative powers: delta,theta, alpha, beta, gamma
    """
    f, Pxx = signal.welch(input_signal,s_rate,'hanning', segment_length, overlap_length)

    # Power in all bands

    total_power = np.sum(Pxx[f <= 50 ])
    delta_power = np.sum(Pxx[f <= 4])
    theta_power = np.sum(Pxx[(f > 4) & (f <= 8) ])
    alpha_power = np.sum(Pxx[(f > 8) & (f <= 13) ])
    beta_power = np.sum(Pxx[(f > 13) & (f <= 30) ])
    gamma_power = np.sum(Pxx[(f > 30) & (f <= 50) ])
    
    delta_power_relative = delta_power/total_power
    theta_power_relative = theta_power/total_power
    alpha_power_relative = alpha_power/total_power
    beta_power_relative = beta_power/total_power
    gamma_power_relative = gamma_power/total_power
    return [delta_power_relative, theta_power_relative, alpha_power_relative, beta_power_relative, gamma_power_relative]

def change_row_to_column(data):
    """
    Helper routine to transform 1d arrays into column vectors that are needed
    by other routines in Chronux.

    I will probably try to change the other functions so this wont be needed.
    EEGLAB needs to transpose it too to use chronux utilities.
    
    Usage: data=change_row_to_column(data)
    
    Inputs:
        data -- required. numpy.darray
        
        If data is a matrix, it is assumed that it is of the
        form samples x channels/trials and it is returned without change. 
        
        If it is a vector, it is transformed to a column vector. 
        
        If it is a struct array of dimension 1, it is again returned as a column vector. 
        
        If it is a struct array with multiple dimensions, it is returned without change.

        Note that the routine only looks at the first field of a struct array.
    
    Ouputs:
        data (in the form samples x channels/trials)
    """

    data = np.squeeze(data) # this is a reassigment so we dont actually modify data.

    if (data.ndim == 1):
        data = data.reshape((1,len(data)))
        return data.transpose()
    else:
        return data

def plot_in_neighborhood(hor_axis,signals,labels,markers,center,width=1000,title=''):
    # Plot signals in the neighborhood
    fig, ax = plt.subplots()
    center = int(center)

    left = center - int(width/2)
    right = center + int(width/2)
    
    try:
        length = len(signals[0])
    except:
        length = 0

    markers = ['r-','b-']
    if (left < 0):
        width_left = 0
        left = center - width_left
        #markers = ['r-','b-']#['r.','b.']
    if (right > length):
        width_right = length - center
        right = center + width_right
        #markers = ['r-','b-']#['r.','b.']
    

    if (left == right):
        markers = ['r.','b.']#['r.','b.']
        for i in np.arange(len(signals)):
            if markers is None:
                ax.plot(signals[i],label=labels[i])
            else:
                ax.plot(signals[i],markers[i],label=labels[i])
        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
        plt.title('Signals in the neighborhood of ' +  str(center) + ' - '+ title)
        plt.show()
        return

    if(hor_axis is None):
        try:
            hor_axis = np.arange(len(signals[0]))
        except:
            hor_axis = None

    for i in np.arange(len(signals)):
        if hor_axis is None:
            if markers is None:
                ax.plot(signals[i][left:right],label=labels[i])
            else:
                ax.plot(signals[i][left:right],markers[i],label=labels[i])
        else:
            if markers is None:
                ax.plot(hor_axis[left:right],signals[i][left:right],label=labels[i])
            else:
                ax.plot(hor_axis[left:right],signals[i][left:right],markers[i],label=labels[i])
        
    legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
    plt.title('Data in the neighborhood of ' +  str(center) + ' - '+ title)
    plt.show()

def plot_in_neighborhoods(hor_axis,signals,labels,markers,centers,width=1000,title=''):
    #centers = np.squeeze(centers)
    for i in centers:
        center = int(i)
        plot_in_neighborhood(None,signals,labels,markers,center,width,title)

def get_matrix_differences(data_A,data_B,plot=False,title='',unsigned=True,relative_method='offset',label_A='data_A',label_B='data_B',max_errors=3,axis=-1,cmap='seismic',max_tolerance=1,width=100,absolute_method='difference'):
    # If you remove the abs you get some other interesting graphs
    #data_A = np.squeeze(data_A)
    #data_B = np.squeeze(data_B)
    # dont apply to matrices np type , it will go wrong

    diff, diff_rel = get_differences(data_A,data_B,relative_method=relative_method,plot=False,unsigned=unsigned,absolute_method=absolute_method)

    if plot:
        fig, ax = plt.subplots()
        ax.set_title('Absolute Difference (' + absolute_method + ') - ' + title, fontsize=14)
        if unsigned:
            im = ax.imshow(diff,aspect='auto',extent=[0, (diff.shape[1]), diff.shape[0], 0],cmap=plt.get_cmap(cmap))
        else:
            im = ax.imshow(diff,aspect='auto',norm=TwoSlopeNorm(0),extent=[0, (diff.shape[1]), diff.shape[0], 0],cmap=plt.get_cmap(cmap))

        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.set_label('abs units', fontsize=14)
        plt.show()
        fig, ax = plt.subplots()
        ax.set_title('Relative Difference ('+relative_method+') - ' + title, fontsize=14)
        if unsigned:
            im = ax.imshow(diff_rel,aspect='auto',extent=[0, (diff.shape[1]), diff.shape[0], 0],cmap=plt.get_cmap(cmap))
        else:
            im = ax.imshow(diff_rel,norm=TwoSlopeNorm(0),aspect='auto',extent=[0, (diff.shape[1]), diff.shape[0], 0],cmap=plt.get_cmap(cmap))
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.set_label(relative_method, fontsize=14)
        plt.show()
        average_diff = np.median(diff,axis=axis)
        average_diff_rel = np.median(diff_rel,axis=axis)
        max_average_diff = np.amax(np.abs(average_diff))
        max_average_diff_rel = np.amax(np.abs(average_diff_rel))
        if max_average_diff!= 0:
            max_indexes = np.argwhere(abs(average_diff) >= max_tolerance*max_average_diff) # this could return a lot of indexes if periodic
            try:
                 max_indexes = list(max_indexes)
                 #max_indexes = max_indexes[0] # If it returns too many indexes you could try this
                 max_indexes = max_indexes[:max_errors]
            except:
                 print("Error trying to obtain indexes")

            if axis == 1 or axis == -1:
                for i in max_indexes:
                    A = np.squeeze(data_A[i,:])
                    B = np.squeeze(data_B[i,:])
                    get_differences(A,B,plot=True,title='row:'+ str(i),relative_method=relative_method,label_A=label_A,label_B=label_B,unsigned=unsigned,width=width)
            
            if axis == 0 :
                for i in max_indexes:
                    A = np.squeeze(data_A[:,i])
                    B = np.squeeze(data_B[:,i])
                    get_differences(A,B,plot=True,title='column:'+ str(i),relative_method=relative_method,label_A=label_A,label_B=label_B,unsigned=unsigned,width=width)

        if max_average_diff_rel!= 0:
            max_indexes = np.argwhere(abs(average_diff_rel) >= max_tolerance*max_average_diff_rel) # this could return a lot of indexes if periodic
            try:
                 max_indexes = list(max_indexes)
                 #max_indexes = max_indexes[0] # If it returns too many indexes you could try this
                 max_indexes = max_indexes[:max_errors]
            except:
                 print("Error trying to obtain indexes")

            if axis == 1 or axis == -1:
                for i in max_indexes:
                    A = np.squeeze(data_A[i,:])
                    B = np.squeeze(data_B[i,:])
                    get_differences(A,B,plot=True,title='row:'+ str(i),relative_method=relative_method,label_A=label_A,label_B=label_B,unsigned=unsigned,width=width)
            
            if axis == 0 :
                for i in max_indexes:
                    A = np.squeeze(data_A[:,i])
                    B = np.squeeze(data_B[:,i])
                    get_differences(A,B,plot=True,title='column:'+ str(i),relative_method=relative_method,label_A=label_A,label_B=label_B,unsigned=unsigned,width=width)

    return diff,diff_rel

def chn_name_mapping(ch_name):
    """
    Map channel names to fit standard naming convention.
    This code is from NeuroDataDesign Standford pyprep Implementation

    Parameters:
        ch_name: string
            channel name to fit

    Returns_
        ch_name: string
            channel name fitted
    """
    ch_name = ch_name.strip('.')
    ch_name = ch_name.upper()
    if 'Z' in ch_name:
        ch_name = ch_name.replace('Z', 'z')
    
    if 'FP' in ch_name:
        ch_name = ch_name.replace('FP', 'Fp')
    
    return ch_name

def as_list(x):
    """
    Represents x as a list if it isn't a list already.
    Parameters:
        x : any object
    Returns:
        list representation of x
    """
    if type(x) is list:
        return x
    elif type(x) is np.ndarray:
        return x.tolist()
    else:
        return [x]

def get_spatial_filter(name='62x19'):
    """
    Returns the default spatial filter of the module.

    Parameters:
        None
    
    Returns:
        A,W tuple of np.ndarrays
        Mixing and Demixing Matrices of the default spatial filter of the module.
    """
    # How sure are we that the order of the channels of matlab is the same as of python?
    #mat_contents = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)),'spatial_filter__'+name+'.mat'))
    mat_contents = loadmat(r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Datos_MsC_Veronica\biomarcadorespruebaICAYorguin\spatial_filter__54x10.mat")
    W = mat_contents['W']
    A = mat_contents['A']
    ch_names = [x[0] for x in mat_contents['ch_names'][0,:].tolist()]
    return A,W,ch_names 


# def plot_ica_components(ica, picks=None, ch_type=None, res=64,
#                         vmin=None, vmax=None, cmap='RdBu_r',
#                         sensors=True, colorbar=False, title=None,
#                         show=False, outlines='head', contours=6,
#                         image_interp='linear',
#                         inst=None, plot_std=True, topomap_args=None,
#                         image_args=None, psd_args=None, reject='auto',
#                         sphere=None,p=20,ncols=5):
#     """
#     Copy of the mne.viz.plot_ica_components function but with 2 additional parameters: p,ncols
#     This allows the control of the distribution of topomaps in the plot.
#     _____________________________________________________________________
#     Project unmixing matrix on interpolated sensor topography.
#     Parameters
#     ----------
#     ica : instance of mne.preprocessing.ICA
#         The ICA solution.
#         If None all are plotted in batches of 20.
#     ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
#         The channel type to plot. For 'grad', the gradiometers are
#         collected in pairs and the RMS for each pair is plotted.
#         If None, then channels are chosen in the order given above.
#     res : int
#         The resolution of the topomap image (n pixels along each side).
#     vmin : float | callable | None
#         The value specifying the lower bound of the color range.
#         If None, and vmax is None, -vmax is used. Else np.min(data).
#         If callable, the output equals vmin(data). Defaults to None.
#     vmax : float | callable | None
#         The value specifying the upper bound of the color range.
#         If None, the maximum absolute value is used. If callable, the output
#         equals vmax(data). Defaults to None.
#     cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
#         Colormap to use. If tuple, the first value indicates the colormap to
#         use and the second value is a boolean defining interactivity. In
#         interactive mode the colors are adjustable by clicking and dragging the
#         colorbar with left and right mouse button. Left mouse button moves the
#         scale up and down and right mouse button adjusts the range. Hitting
#         space bar resets the range. Up and down arrows can be used to change
#         the colormap. If None, 'Reds' is used for all positive data,
#         otherwise defaults to 'RdBu_r'. If 'interactive', translates to
#         (None, True). Defaults to 'RdBu_r'.
#         .. warning::  Interactive mode works smoothly only for a small amount
#                       of topomaps.
#     sensors : bool | str
#         Add markers for sensor locations to the plot. Accepts matplotlib
#         plot format string (e.g., 'r+' for red plusses). If True (default),
#         circles  will be used.
#     colorbar : bool
#         Plot a colorbar.
#     title : str | None
#         Title to use.
#     show : bool
#         Show figure if True.
#     contours : int | array of float
#         The number of contour lines to draw. If 0, no contours will be drawn.
#         When an integer, matplotlib ticker locator is used to find suitable
#         values for the contour thresholds (may sometimes be inaccurate, use
#         array for accuracy). If an array, the values represent the levels for
#         the contours. Defaults to 6.
#     image_interp : str
#         The image interpolation to be used. All matplotlib options are
#         accepted.
#     inst : Raw | Epochs | None
#         To be able to see component properties after clicking on component
#         topomap you need to pass relevant data - instances of Raw or Epochs
#         (for example the data that ICA was trained on). This takes effect
#         only when running matplotlib in interactive mode.
#     plot_std : bool | float
#         Whether to plot standard deviation in ERP/ERF and spectrum plots.
#         Defaults to True, which plots one standard deviation above/below.
#         If set to float allows to control how many standard deviations are
#         plotted. For example 2.5 will plot 2.5 standard deviation above/below.
#     topomap_args : dict | None
#         Dictionary of arguments to ``plot_topomap``. If None, doesn't pass any
#         additional arguments. Defaults to None.
#     image_args : dict | None
#         Dictionary of arguments to ``plot_epochs_image``. If None, doesn't pass
#         any additional arguments. Defaults to None.
#     psd_args : dict | None
#         Dictionary of arguments to ``psd_multitaper``. If None, doesn't pass
#         any additional arguments. Defaults to None.
#     reject : 'auto' | dict | None
#         Allows to specify rejection parameters used to drop epochs
#         (or segments if continuous signal is passed as inst).
#         If None, no rejection is applied. The default is 'auto',
#         which applies the rejection parameters used when fitting
#         the ICA object.
#     p : int
#         Number of topomaps per plot figure
#     ncols: int
#         Number of columns for topomaps in a figure
    
#     Returns
#     -------
#     fig : instance of matplotlib.figure.Figure or list
#         The figure object(s).
#     Notes
#     -----
#     When run in interactive mode, ``plot_ica_components`` allows to reject
#     components by clicking on their title label. The state of each component
#     is indicated by its label color (gray: rejected; black: retained). It is
#     also possible to open component properties by clicking on the component
#     topomap (this option is only available when the ``inst`` argument is
#     supplied).
#     """

#     if ica.info is None:
#         raise RuntimeError('The ICA\'s measurement info is missing. Please '
#                            'fit the ICA or add the corresponding info object.')

#     topomap_args = dict() if topomap_args is None else topomap_args
#     topomap_args = copy.copy(topomap_args)
#     if 'sphere' not in topomap_args:
#         topomap_args['sphere'] = sphere
#     if picks is None:  # plot components by sets of p
#         ch_type = _get_ch_type(ica, ch_type)
#         n_components = ica.mixing_matrix_.shape[1]
#         figs = []
#         for k in range(0, n_components, p):
#             picks = range(k, min(k + p, n_components))
#             fig = plot_ica_components(ica, picks=picks, ch_type=ch_type,
#                                       res=res, vmax=vmax,
#                                       cmap=cmap, sensors=sensors,
#                                       colorbar=colorbar, title=title,
#                                       show=show, outlines=outlines,
#                                       contours=contours,
#                                       image_interp=image_interp, inst=inst,
#                                       plot_std=plot_std,
#                                       topomap_args=topomap_args,
#                                       image_args=image_args,
#                                       psd_args=psd_args, reject=reject,
#                                       sphere=sphere,ncols=ncols)
#             figs.append(fig)
#         return figs
#     else:
#         picks = _picks_to_idx(ica.info, picks)
#     ch_type = _get_ch_type(ica, ch_type)

#     cmap = _setup_cmap(cmap, n_axes=len(picks))
#     data = np.dot(ica.mixing_matrix_[:, picks].T,
#                   ica.pca_components_[:ica.n_components_])

#     data_picks, pos, merge_channels, names, _, sphere, clip_origin = \
#         _prepare_topomap_plot(ica, ch_type, sphere=sphere)
#     outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)

#     data = np.atleast_2d(data)
#     data = data[:, data_picks]

#     # prepare data for iteration
#     fig, axes, _, _ = _prepare_trellis(len(data), ncols=ncols)
#     if title is None:
#         title = 'Components'
#     fig.suptitle(title)

#     titles = list()
#     for ii, data_, ax in zip(picks, data, axes):
#         kwargs = dict(color='gray') if ii in ica.exclude else dict()
#         titles.append(ax.set_title(ica._ica_names[ii], fontsize=12, **kwargs))
#         if merge_channels:
#             data_, names_ = _merge_ch_data(data_, ch_type, names.copy())
#         vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
#         im = plot_topomap(
#             data_.flatten(), pos, vmin=vmin_, vmax=vmax_, res=res, axes=ax,
#             cmap=cmap[0], outlines=outlines, contours=contours,
#             image_interp=image_interp, show=False, sensors=sensors)[0]
#         im.axes.set_label(ica._ica_names[ii])
#         if colorbar:
#             cbar, cax = _add_colorbar(ax, im, cmap, title="AU",
#                                       side="right", pad=.05, format='%3.2f')
#             cbar.ax.tick_params(labelsize=12)
#             cbar.set_ticks((vmin_, vmax_))
#         _hide_frame(ax)
#     del pos
#     tight_layout(fig=fig)
#     fig.subplots_adjust(top=0.88, bottom=0.)
#     fig.canvas.draw()
#     # add title selection interactivity
#     def onclick_title(event, ica=ica, titles=titles):
#         # check if any title was pressed
#         title_pressed = None
#         for title in titles:
#             if title.contains(event)[0]:
#                 title_pressed = title
#                 break
#         # title was pressed -> identify the IC
#         if title_pressed is not None:
#             label = title_pressed.get_text()
#             ic = int(label[-3:])
#             # add or remove IC from exclude depending on current state
#             if ic in ica.exclude:
#                 ica.exclude.remove(ic)
#                 title_pressed.set_color('k')
#             else:
#                 ica.exclude.append(ic)
#                 title_pressed.set_color('gray')
#             fig.canvas.draw()
#     fig.canvas.mpl_connect('button_press_event', onclick_title)

#     # add plot_properties interactivity only if inst was passed
#     if isinstance(inst, (BaseRaw, BaseEpochs)):
#         def onclick_topo(event, ica=ica, inst=inst):
#             # check which component to plot
#             if event.inaxes is not None:
#                 label = event.inaxes.get_label()
#                 if label.startswith('ICA'):
#                     ic = int(label[-3:])
#                     ica.plot_properties(inst, picks=ic, show=True,
#                                         plot_std=plot_std,
#                                         topomap_args=topomap_args,
#                                         image_args=image_args,
#                                         psd_args=psd_args, reject=reject)
#         fig.canvas.mpl_connect('button_press_event', onclick_topo)

#     plt_show(show)
#     return fig


def save_json(data,filename,indent='    '):
    with open(filename, 'w') as outfile:
        json.dump(data,outfile,indent=indent)