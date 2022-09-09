import matplotlib.pyplot as plt
from nilearn import plotting
plotting.plot_img('/media/takina/DATA/tesis_sebas/nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/imagesTs/prostate_30_0000.nii.gz')
# plotting.plot_img('/media/takina/DATA/tesis_sebas/nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTs/la_001_0000.nii.gz')
# plotting.plot_img('/home/takina/OUTPUT_DIRECTORY_HEART/la_001.nii.gz')
plotting.plot_img('/home/takina/OUTPUT_DIRECTORY/prostate_30.nii.gz')
plotting.show()