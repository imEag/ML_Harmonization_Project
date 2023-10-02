from sovaharmony.processing import harmonize

BIOMARCADORES = {
    'name':'BIOMARCADORES',
    'input_path':r'C:\Users\elian\OneDrive - Universidad de Antioquia\biomarcadoresprueba',
    'layout':{'extension':'.vhdr', 'task':'OE','suffix':'eeg', 'return_type':'filename'},
    'args':{'line_freqs':[60]},
    'group_regex':'(.+).{3}',
    'events_to_keep':None,
    'run-label':'restEC'
}

harmonize(BIOMARCADORES)