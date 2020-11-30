from hdf_operations.hdf2files import HDF2files
#from hdf2files import HDF2files

import os

location = os.path.join('datasets', 'ee285f-public', 'caltech_ucsd_birds', 'birds.hdf5')
txt_dir_name = 'text_c10'

converter = HDF2files(path=location, main_dir_name='caltech_ucsd_birds', img_dir_name='caltech_ucsd_birds', txt_dir_name=txt_dir_name)
converter.convert()

