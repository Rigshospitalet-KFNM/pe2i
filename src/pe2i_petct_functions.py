import os
import re
import glob
import random
import datetime
from typing import List, Optional
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd
import nibabel as nib
import scipy
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
matplotlib.rc('text', usetex=True)
from functools import partial
from itertools import islice
from sklearn.linear_model import LinearRegression
from tensorflow import keras  # type: ignore
import tensorflow as tf
import keras_contrib
import dotenv # type: ignore
dotenv.load_dotenv()

import shutil

from pe2i_environment import environment as env
# tf.config.list_physical_devices('GPU')
# tf.config.list_physical_devices('CPU')
import ants
from ants import ANTsImage
from ants.internal import get_lib_fn, get_pointer_string, process_arguments
from nilearn.image import smooth_img, resample_to_img # type: ignore
from nipype.interfaces.niftyseg import LabelFusion # type: ignore
from nipype.interfaces.niftyreg import RegAladin, RegResample, RegTransform # type: ignore
from pylatex import Figure, Command, NoEscape, Tabular, Document, Package,Section, SubFigure, MultiColumn
from pylatex.utils import bold, NoEscape, escape_latex
from pylatex.base_classes import Environment
from pylatex.table import Tabularx
import pydicom
from pydicom import Dataset
from pydicom.uid import PositronEmissionTomographyImageStorage, generate_uid
from pydicom.dataset import Dataset, FileMetaDataset
from dicomnode.dicom.dicom_factory import DicomFactory, CopyOrElseElement,\
    Blueprint, InstanceCopyElement, StaticElement, CopyElement, FunctionalElement,\
    InstanceEnvironment, SeriesElement
from dicomnode.dicom.blueprints import get_time, get_today, add_UID_tag
from HD_CTBET.run import run_hd_ctbet # type: ignore


STATIC_FILES = env.STATIC_PATH # path to static files
FWHM = 2.35482 # converting sigma of 1 to FWHM 2.35482*1 mm
CEREBELLUM_INDEX = 4 # label for Cerebellum cortex
MID_POINTS = [0] # for graphical visualisation
DEF_MIDS = [0.07] # shifting PET colormap parameter

# Colormap to be used when displaying images.
_PETRainbowCMAP = matplotlib.colors.LinearSegmentedColormap(
    'PET-Rainbow',
    {
        u'blue': [(0.0, 0.0, 0.0),
                  (0.07, 0.4667, 0.4667),
                  (0.15, 0.6667, 0.6667),
                  (0.2, 0.8667, 0.8667),
                  (0.25, 0.8667, 0.8667),
                  (0.3, 0.8667, 0.8667),
                  (0.35, 0.8667, 0.8667),
                  (0.4, 0.8333, 0.8333),
                  (0.45, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (0.55, 0.0, 0.0),
                  (0.6, 0.0, 0.0),
                  (0.65, 0.0, 0.0),
                  (0.7, 0.0, 0.0),
                  (0.75, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (0.85, 0.1, 0.1),
                  (0.9, 0.2, 0.2),
                  (0.95, 0.3, 0.3),
                  (1.0, 1.0, 1.0)],
        u'green': [(0.0, 0.0, 0.0),
                   (0.15, 0.0, 0.0),
                   (0.2, 0.0, 0.0),
                   (0.25, 0.4667, 0.4667),
                   (0.3, 0.8, 0.8),
                   (0.35, 0.8667, 0.8667),
                   (0.4, 0.8667, 0.8667),
                   (0.45, 0.86, 0.86),
                   (0.5, 0.8633, 0.8633),
                   (0.55, 0.8667, 0.8667),
                   (0.6, 0.8667, 0.9667),
                   (0.65, 0.9667, 1.0),
                   (0.7, 1.0, 0.9333),
                   (0.75, 0.8, 0.8),
                   (0.8, 0.6, 0.6),
                   (0.85, 0.1, 0.1),
                   (0.9, 0.2, 0.2),
                   (0.95, 0.3, 0.3),
                   (1.0, 1.0, 1.0)],
        u'red': [(0.0, 0.0, 0.0),
                 (0.15, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.25, 0.0, 0.0),
                 (0.3, 0.0, 0.0),
                 (0.35, 0.0, 0.0),
                 (0.4, 0.0, 0.0),
                 (0.45, 0.0, 0.0),
                 (0.5, 0.0, 0.0),
                 (0.55, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.65, 0.7333, 0.7333),
                 (0.7, 0.9333, 0.9333),
                 (0.75, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (0.85, 1.0, 1.0),
                 (0.9, 0.8667, 0.8667),
                 (0.95, 0.8, 0.8),
                 (1.0, 1.0, 1.0)]},
    256)

# defining slices to include in cropped image for visualisation purposes
FIRST_DIM = (22, 234)
SECOND_DIM = (4, 216)
FIRST_DIM_CROPPED = slice(*FIRST_DIM)
SECOND_DIM_CROPPED = slice(*SECOND_DIM)


class MidPointNorm(matplotlib.colors.Normalize):
    """
    Class defining normalization of colors to be used when plotting using matplotlib.
    The class allows one to define the minimum and maximum used in normalization, as usual,
    while it also allows one to define a "mid-point" -- this allows one to have values
    between the provided minimum and maximum mapped to something other than 0.5 in the colormap.

    Parameters:
    -----------
    vmin (numeric):
        Minimum used in the normalization.

    vmax (numeric):
        Maximum used in the normalization.

    midpoints (numeric):
        Midpoints used in the normalization.

    clip (bool):
        Whether to allow clipping in normalization (standard matplotlib argument).

    defmids (list):
        What point in the colorscale the provided midpoints should be mapped to (normally, this would be 0.5, but one may "stretch" the colormap using this).

    Attributes:
    -----------
    vmin (numeric):
        Minimum used in the normalization.

    vmax (numeric):
        Maximum used in the normalization.

    midpoints (numeric):
        Midpoints used in the normalization.

    defmids (list):
        What point in the colorscale the provided midpoints should be mapped to (normally, this would be 0.5, but one may "stretch" the colormap using this).

    """
    def __init__(self, vmin=None, vmax=None, midpoints=None, clip=False, defmids=[0.5]):
        self.midpoints = midpoints
        self.defmids = defmids
        self._vmin = vmin
        self._vmax = vmax
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin] + self.midpoints + \
            [self.vmax], [0] + self.defmids + [1]
        result = np.ma.array(
            np.interp(value, x, y), mask=result.mask, copy=False)
        return (result[0]
                if is_scalar
                else result)


class ColorBox(Environment):
    """
    Create a colored box environment in LaTeX using the colorbox command.

    Parameters:
    -----------
    color (str):
        The name of the color to be used for the box.

    Attributes:
    -----------
    color (str):
        The name of the color to be used for the box.

    """
    def __init__(self, color):
        super().__init__(arguments=NoEscape(r'\colorbox{' + color + r'}'))
        self.content_separator = ''


def swap_dims(self, modality, name):
    """
    Reorients a given NIfTI image from radiological to neurological orientation if necessary
    and saves the reoriented image to a specified output path. This ensures consistency in image orientation,
    particularly for PET scans, which often require neurological orientation for further processing.

    Parameters:
    -----------
    modality : nibabel.Nifti1Image
        The NIfTI image object to be processed and potentially reoriented.

    name : str
        The base name for the output file. The full output file name will include the suffix `_swap.nii.gz`.

    Returns:
    --------
    modality_path : str
        The file path to the saved, reoriented NIfTI image.

    Raises:
    -------
    ValueError
        If the `modality` parameter is not a `nibabel.Nifti1Image` object.
    IOError
        If the function fails to save the reoriented NIfTI image to the specified path.

    Notes:
    ------
    - The function first checks the image's orientation using the affine transformation and determines whether it
      is in radiological orientation (i.e., the first axis is labeled 'R' for right).
    - If the orientation is radiological, it flips the image data along the left-right axis to convert it to neurological orientation.
    - The flipped image is saved with the suffix `_swap.nii.gz` in the current working directory.
    - This function logs the process, indicating when an orientation change is made and confirming the successful saving of the image.

    Example:
    --------
    >>> modality = nib.load("example_image.nii.gz")
    >>> output_path = swap_dims(self, modality, "example_image")
    >>> print(f"Reoriented image saved at: {output_path}")
    """

    # Validate input type
    if not isinstance(modality, nib.Nifti1Image):
        raise ValueError("Input modality must be a Nifti1Image object.")

    # Construct the full output path for the new NIfTI image
    modality_path = os.getcwd() + '/' + name + '_swap.nii.gz'

    # Convert the input image to the closest canonical orientation
    img = nib.as_closest_canonical(modality)

    # Check if the first dimension is in the 'R' (Right) direction
    if nib.aff2axcodes(modality.affine)[0] == 'R':
        # Log that we're changing from radiological to neurological orientation
        self.logger.info('Changing to neurological orientation')

        # Get the image data as a numpy array
        img_data = img.get_fdata()

        # Flip the data along the first axis (left-right flip)
        img_data_swap = np.flip(img_data, 0)

        # Create a new NIfTI image with the flipped data and the same affine transformation
        img = nib.Nifti1Image(img_data_swap, img.affine)

    # Save the new NIfTI image to the specified path
    nib.save(img, modality_path)

    # Check if the file is saved successfully
    if os.path.exists(modality_path):
        self.logger.info(f'NIfTI image saved successfully at: {modality_path}')
    else:
        self.logger.error(f'Failed to save NIfTI image at: {modality_path}')
        raise IOError(f"Failed to save NIfTI image at {modality_path}")

    return modality_path

def convert_LAC_to_HU(self, dd_path):
    """
    Converts a Deep Dixon (DD) NIfTI image from Linear Attenuation Coefficient (LAC) units to Hounsfield Units (HU)
    and saves the converted image to a new file.

    Parameters:
    -----------
    dd_path : str
        File path to the input NIfTI image in LAC units.

    Returns:
    --------
    converted_path : str
        File path to the converted NIfTI image in HU units.

    Notes:
    ------
    - LAC values below 1016.7 are converted using a linear formula (`LAC / 0.95 - 1000`).
    - LAC values equal to or above 1016.7 are converted using a different formula: `(LAC / 10000 - 0.0471) / 0.000051 - 1000`.
    - The converted image is saved with the name `DD_swap_HU.nii.gz` in the current working directory.
    """

    # Log the conversion process
    self.logger.info('converting Deep Dixon from LAC to HU')

    # Define the output file path
    converted_path = f'{os.getcwd()}/DD_swap_HU.nii.gz'

    # Load the input NIfTI image
    dd_nib = nib.load(dd_path)
    dd = dd_nib.get_fdata() # Get the image data as a NumPy array

    # Create a copy of the image data for conversion
    dd_copy = dd.copy()

    # Initialize an array of zeros with the same shape as the input data for storing HU values
    dd_hu =np.zeros_like(dd_copy)

    # Convert LAC values to HU using two different formulas based on the value threshold (1016.7)
    hu_lower = dd_copy/0.95-1000 # Conversion for values below 1016.7
    hu_upper = (dd_copy/10000-0.0471)/0.000051-1000 # Conversion for values >= 1016.7

    # Apply the appropriate conversion formula to each element in the array
    dd_hu[dd_copy<1016.7] = hu_lower[dd_copy<1016.7]
    dd_hu[dd_copy>=1016.7] = hu_upper[dd_copy>=1016.7]

    # Create a new NIfTI image with the converted data and the same affine transformation as the original image
    im = nib.Nifti1Image(dd_hu, dd_nib.affine)
     # Save the new NIfTI image to the specified path
    nib.save(im, converted_path)

    # Return the file path of the converted image
    return converted_path

#import torch.nn.functional as F

#def softmax(x):
#    return F.softmax(x, 1)

#import nnunet.utilities
#nnunet.utilities.softmax_helper = softmax

def run_skullstripping(self, input_modality_path):
    """
    Perform skull stripping on a anatomical scan using the `hd_ctbet` method to remove non-brain tissues.

    Parameters:
    -----------
    input_modality_path : str
        File path to the input NIfTI image (anatomical scan) to be skull-stripped.

    Returns:
    --------
    output_filename : str
        File path to the resulting skull-stripped NIfTI image (`anatomical_swap_BET.nii.gz`).

    Notes:
    ------
    - The skull stripping is performed using the `hd_ctbet` function in `fast` mode.
    - The operation is executed on the CPU without test-time augmentation (`do_tta=False`).
    - The output file is saved in the current working directory.
    """

    # Log the beginning of the skull stripping process
    self.logger.info('Running skullstripping')

    # Define the output filename for the skull-stripped image
    output_filename = os.getcwd() + '/anatomical_swap_BET.nii.gz'
    if Path(output_filename).is_file():
        return output_filename
    # Call the hd_ctbet function to perform skull stripping

    script = "\n".join([
        "from HD_CTBET.run import run_hd_ctbet",
        f"run_hd_ctbet({input_modality_path!r}, {output_filename!r}, mode='fast', device='cpu', do_tta=False)",
    ])
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True
    )


    #run_hd_ctbet(str(input_modality_path), str(output_filename), mode='fast', device='cpu', do_tta =False)

    return output_filename


def process_anatomical(self, brain_path):
    """
    Preprocess a anatomical scan by applying thresholding to limit HU values and smoothing to reduce noise
    before performing segmentation.

    Parameters:
    -----------
    brain_path : str
        File path to the NIfTI image of the brain anatomical scan to be preprocessed.

    Returns:
    --------
    brain_sm_th_path : str
        File path to the preprocessed and saved NIfTI image (`brain_preprocessed.nii.gz`).

    Notes:
    ------
    - The thresholding operation keeps values in the range of 0-100 Hounsfield units (HU).
      Values outside this range are set to 0.
    - Smoothing is performed on the thresholded image using a specified Full Width at Half Maximum (FWHM) value.
    - The output file is saved as `brain_preprocessed.nii.gz` in the current working directory.
    """
    # Generate the output file path for the preprocessed image
    brain_sm_th_path = os.getcwd() + '/brain_preprocessed.nii.gz'
    self.logger.info('Applying thresholding and smoothing')

    # Load the NIfTI image
    brain_nib = nib.load(brain_path)

    # Convert the NIfTI image data to a NumPy array
    brain_data = brain_nib.get_fdata()

    # Apply thresholding to keep values in the range of 0-100 HU
    brain_th =  brain_data.copy()
    brain_th[(brain_th<0)|(brain_th>100)] = 0

    # Convert the thresholded NumPy array back to a NIfTI image
    brain_th_nib = nib.Nifti1Image(brain_th, brain_nib.affine)

    # Apply smoothing to the thresholded image using the specified FWHM
    brain_sm_th = smooth_img(brain_th_nib, FWHM)

    # Save the preprocessed image
    nib.save(brain_sm_th, brain_sm_th_path)
    self.logger.info(f'saved {brain_sm_th_path }')

    # Return the path to the preprocessed image
    return brain_sm_th_path


def cerebellum_mask(self, input_file):
    """
    Generate a cerebellum mask by segmenting the cerebellum gray matter using the LabelFusion tool
    with the STEPS algorithm. The process uses pre-defined templates and classifier settings for
    accurate segmentation.

    Parameters:
    -----------
    input_file : str
        The file path to the input NIfTI image (e.g., a brain MRI or anatomical scan) that needs cerebellum
        segmentation.

    Returns:
    --------
    str
        The file path to the generated cerebellum mask NIfTI image (`cerebellum.nii.gz`).

    Exceptions:
    -----------
    Raises `FileNotFoundError` if the LabelFusion process fails to generate the output file.

    Notes:
    ------
    - The segmentation uses the STEPS algorithm, which is a machine learning-based method for
      segmentation.
    - The function uses predefined static files (atlas and templates) for the segmentation.
    """
    self.logger.info('Segmenting cerebellum gray matter mask')
    # Initialize LabelFusion with the necessary inputs
    lf = LabelFusion()
    lf.inputs.file_to_seg = input_file
    lf.inputs.in_file = STATIC_FILES / 'atlas4_swap.nii.gz'
    lf.inputs.template_file = STATIC_FILES / 'templates_swap.nii.gz'
    lf.inputs.classifier_type = 'STEPS'  # Use the STEPS algorithm for segmentation
    lf.inputs.kernel_size = 5            # Set the kernel size for the algorithm
    lf.inputs.template_num = 8           # Use 8 templates in the process
    lf.inputs.mrf_value = 0.5            # Set the MRF (Markov Random Field) value for regularization
    lf.inputs.out_file = os.getcwd() + '/cerebellum.nii.gz'
    # Run the LabelFusion proces
    lf.run()
    # Check if the output file was successfully created
    out_file = lf.inputs.out_file
    if not Path(out_file).exists():
        self.logger.error(f"Output file {out_file} was not created.")
        raise FileNotFoundError(f"Segmentation failed: Output file {out_file} does not exist.")

    self.logger.info(f"Segmentation successful. Output file: {out_file}")
    # Return the path to the output file
    return out_file


# def resampling(self, pet_nii, anatomical_nii, brain_nii):
#     """
#     Resample and register PET and anatomical scans to a brain template, ensuring that all steps 
#     are performed only if the corresponding output files do not already exist.

#     This function performs the following operations:
#     1. Registers the anatomical brain modality to an average brain template.
#     2. Resamples the anatomical brain modality and anatomical modality to match the brain template.
#     3. Registers and resamples the PET image to the anatomical modality and brain template.

#     Parameters:
#     -----------
#     pet_nii : nib.Nifti1Image
#         The NIfTI image representing the PET scan.
#     anatomical_nii : nib.Nifti1Image
#         The NIfTI image representing the anatomical scan.
#     brain_nii : nib.Nifti1Image
#         The NIfTI image representing the brain scan.

#     Returns:
#     --------
#     petrsltemplate_path : str
#         File path to the PET image resampled to the brain template.
#     anatomicalrsl_path : str
#         File path to the anatomical image resampled to the brain template.
#     brainrsl_path : str
#         File path to the brain image resampled to the brain template.
#     trans_pet : pathlike object 
#         The transformation matrix file that defines the transformation from PET space to CT space
#     trans_anatomical : pathlike object 
#         The transformation matrix file that defines the transformation from CT space to MNI space.
#     Exceptions:
#     -----------
#     Raises IOError if any of the registration or resampling steps fail to create the expected output files.

#     Notes:
#     ------
#     The function uses the `reg_aladin` tool for image registration and the `reg_resample` tool for resampling.
#     """

#     # Define file paths for templates and output files
#     template_path = STATIC_FILES / 'avg_template_swap.nii.gz'
#     brainreg_path = os.getcwd() + '/brain_reg_avg.nii.gz'
#     brainrsl_path = os.getcwd() + '/brain_rsl_avg.nii.gz'
#     trans_anatomical = os.getcwd() + '/brain_to_avg.txt'
#     anatomicalrsl_path = os.getcwd() + '/anatomical_rsl_avg.nii.gz'
#     petreg_path = os.getcwd() + '/pet_rsl_anatomical.nii.gz'
#     petrsltemplate_path = os.getcwd() + '/pet_rsl_avg.nii.gz'
#     trans_pet = os.getcwd() + '/pet_to_anatomical.txt'

#     # Step 1: Register anatomical brain to the average template if the transformation doesn't exist
#     self.logger.info(f'Registering anatomical brain to template')
#     reg_aladin(ref_file=template_path, 
#                 flo_file=brain_nii,
#                 aff_file=trans_anatomical,
#                 res_file=brainreg_path,
#                 verbosity='none')
    
#     # Verify if registration was successful
#     if not Path(trans_anatomical).is_file():
#         self.logger.error(f"Failed to save anatomical brain registration at {trans_anatomical}")
#         raise IOError(f"Anatomical brain registration not saved: {trans_anatomical}")

#     # Step 2: Resample anatomical brain to the template if not already resampled  
#     self.logger.info(f'Resampling anatomical brain to template')
#     reg_resample(ref_file=template_path, 
#                     flo_file=brain_nii,
#                     trans_file=trans_anatomical,
#                     out_file=brainrsl_path,
#                     interpol='LIN',
#                     pad_val=-1024,
#                     verbosity='none')
    
#     # Check if brain resampling was successful
#     if not Path(brainrsl_path).is_file():
#         self.logger.error(f"Failed to save resampled anatomical brain at {brainrsl_path}")
#         raise IOError(f"Resampled anatomical brain not saved: {brainrsl_path}")
        
#     # Step 3: Resample anatomical modality to the template if not already resampled
#     self.logger.info('Resampling anatomical to template')
#     reg_resample(ref_file=template_path,
#                     flo_file=anatomical_nii,
#                     trans_file=trans_anatomical,
#                     out_file=anatomicalrsl_path,
#                     interpol='LIN',
#                     pad_val=-1024,
#                     verbosity='none')
    
#     # Check if anatomical resampling was successful
#     if not Path(anatomicalrsl_path).is_file():
#         self.logger.error(f"Failed to save resampled anatomical at {anatomicalrsl_path}")
#         raise IOError(f"Resampled anatomical not saved: {anatomicalrsl_path}")

#     # Step 4: Register PET to anatomical modality if the transformation doesn't exist
#     self.logger.info('Registering PET to anatomical')
#     reg_aladin(ref_file=anatomical_nii, 
#                 flo_file=pet_nii,
#                 aff_file=trans_pet,
#                 res_file=petreg_path,
#                 verbosity='none')
    
#     # Verify if PET registration was successful
#     if not Path(trans_pet).is_file():
#         self.logger.error(f"Failed to save PET registration at {trans_pet}")
#         raise IOError(f"PET registration not saved: {trans_pet}")

#     # Step 5: Resample PET to the brain template if not already resampled
#     self.logger.info('Resampling PET to template')
#     reg_resample(ref_file=template_path,
#                     flo_file=petreg_path,
#                     trans_file=trans_anatomical,
#                     out_file=petrsltemplate_path,
#                     interpol='LIN',
#                     pad_val=0,
#                     verbosity='none')
    
#     # Check if PET resampling to template was successful
#     if not Path(petrsltemplate_path).is_file():
#         self.logger.error(f"Failed to save resampled PET at {petrsltemplate_path}")
#         raise IOError(f"Resampled PET not saved: {petrsltemplate_path}")

#     return petrsltemplate_path, anatomicalrsl_path, brainrsl_path, trans_pet, trans_anatomical


def get_predition(logger, brain_path, pet_path):
    """
    Obtain the segmentation prediction for basal ganglia using a trained deep learning model.

    This function normalizes the input anatomical and PET images, loads a pre-trained model, and generates 
    a segmentation prediction for the basal ganglia. It ensures TensorFlow resources are properly 
    released after prediction.

    Parameters:
    -----------
    logger : Logger object
        Logger for logging the process information.
    brain_path : str
        File path to the anatomical brain modality NIfTI image.
    pet_path : str
        File path to the PET NIfTI image.

    Returns:
    --------
    prediction_image : numpy array
        The segmentation prediction image of basal ganglia.
    
    Exceptions:
    -----------
    Logs an exception if there is an error while clearing the TensorFlow session.

    Notes:
    ------
    - Normalization of input images is performed before prediction.
    - The model expects normalized anatomical and PET images as inputs and predicts three label values (0, 2, 3).
    - TensorFlow session is cleared to manage memory usage after prediction.
    """

    logger.info('Getting predition.')
    
    # Normalize anatomical and PET images
    input_files = normalize(logger, brain_path, pet_path) 
    
    try:
        # Define the path to the trained model
        MODEL_PATH  = STATIC_FILES / 'new_model_other.keras'
        
        # Run prediction using the model
        prediction_image = run_prediction(logger, 
                                          model_file=MODEL_PATH,  # Path to the trained model
                                          labels=(0, 2, 3),       # The label values (0, 2, 3) that the model will predict
                                          input_data=input_files  # The normalized input data
                                          )
    finally:
        try:
            # Clear the TensorFlow session to free up resources
            tf.keras.backend.clear_session()
        except Exception as e:
            # Log any exceptions encountered during session clearing
            logger.info(e)

    return prediction_image


def registration_helper(
    fixed,
    moving,
    type_of_transform="SyN-adjusted",
    outprefix="",
    grad_step=0.2,
    flow_sigma=3.,
    total_sigma=0.,
    aff_metric="mattes",
    aff_sampling=32,
    syn_metric="mattes",
    syn_sampling=32,
    reg_iterations=(40, 20, 0),
    write_composite_transform=False,
    verbose=False,
    **kwargs
):
    """
    Register a pair of images either through the full or simplified
    interface to the ANTs registration method.

    ANTsR function: `antsRegistration`

    Arguments
    ---------
    fixed : ants.ANTsImage
        fixed image to which we register the moving image.

    moving : ants.ANTsImage
        moving image to be mapped to fixed space.

    type_of_transform : string
        A linear or non-linear registration type. Mutual information metric by default.
        See Notes below for more.

    initial_transform : list of strings (optional)
        transforms to prepend. If None, a translation is computed to align the image centers of mass.

    outprefix : string
        output will be named with this prefix.

    grad_step : scalar
        gradient step size (not for all tx)

    flow_sigma : scalar
        smoothing for update field
        At each iteration, the similarity metric and gradient is calculated. 
        That gradient field is also called the update field and is smoothed 
        before composing with the total field (i.e., the estimate of the total 
        transform at that iteration). This total field can also be smoothed 
        after each iteration.

    total_sigma : scalar
        smoothing for total field

    aff_metric : string
        the metric for the affine part (GC, mattes, meansquares)

    aff_sampling : scalar
        number of bins for the mutual information metric

    syn_metric : string
        the metric for the syn part (CC, mattes, meansquares, demons)

    syn_sampling : scalar
        the nbins or radius parameter for the syn metric

    reg_iterations : list/tuple of integers
        vector of iterations for syn. we will set the smoothing and multi-resolution parameters based on the length of this vector.

    write_composite_transform : boolean
        Boolean specifying whether or not the composite transform (and its inverse, if it exists) should be written to an hdf5 composite file. This is false by default so that only the transform for each stage is written to file.

    verbose : boolean
        request verbose output (useful for debugging)

    kwargs : keyword args
        extra arguments

    Returns
    -------
    dict containing follow key/value pairs:
        `warpedmovout`: Moving image warped to space of fixed image.
        `warpedfixout`: Fixed image warped to space of moving image.
        `fwdtransforms`: Transforms to move from moving to fixed image.
        `invtransforms`: Transforms to move from fixed to moving image.

    """
    if isinstance(fixed, list) and (moving is None):
        processed_args = process_arguments(fixed)
        libfn = get_lib_fn("antsRegistration")
        reg_exit = libfn(processed_args)
        if (reg_exit != 0):
            raise RuntimeError(f"Registration failed with error code {reg_exit}")
        else:
            return 0

    if not (ants.is_image(fixed) and ants.is_image(moving)):
        raise ValueError("Fixed and moving images must be ants.ANTsImage objects")

    if type_of_transform == "":
        type_of_transform = "SyN-adjusted"

    if isinstance(type_of_transform, (tuple, list)) and (len(type_of_transform) == 1):
        type_of_transform = type_of_transform[0]

    if np.sum(np.isnan(fixed.numpy())) > 0:
        raise ValueError("fixed image has NaNs - replace these")
    if np.sum(np.isnan(moving.numpy())) > 0:
        raise ValueError("moving image has NaNs - replace these")

    if fixed.dimension != moving.dimension:
        raise ValueError("Fixed and moving image dimensions are not the same.")
    # ----------------------------

    args = [fixed, moving, type_of_transform, outprefix]

    mysyn = "SyN[%f,%f,%f]" % (grad_step, flow_sigma, total_sigma)
    itlen = len(reg_iterations)  # NEED TO CHECK THIS
    if itlen == 0:
        synits = reg_iterations
    else:
        synits = "x".join([str(ri) for ri in reg_iterations])

    inpixeltype = fixed.pixeltype
    moving = moving.clone("float")
    fixed = fixed.clone("float")

    warpedfixout = moving.clone()
    warpedmovout = fixed.clone()
    f = get_pointer_string(fixed)
    m = get_pointer_string(moving)
    wfo = get_pointer_string(warpedfixout)
    wmo = get_pointer_string(warpedmovout)

    if type_of_transform == "SyN-adjusted":
        initx = ["[%s,%s,0]" % (f, m)]
        args = [
            "-d",
            str(fixed.dimension),
            "-r",
        ] + initx + [
            "-n",
            "Linear",
            # --- NEW:  Rigid stage,
            "-t",
            "Rigid[0.1]",
            "-m",
            "%s[%s,%s,1,%s]" % (aff_metric, f, m, aff_sampling),
            "-c",
            "[1000x500x250x100,1e-6,10]",
            "-s",
            "4.0x3.0x2.0x1.0",
            "-f",
            "12x8x4x2",
            # --- end new Rigid stage ---
            "-t",
            "Affine[0.1]",  # this is different
            "-m",
            "%s[%s,%s,1,%s]" # this is different
            % (aff_metric, f, m, aff_sampling), 
            "-c",
            "[1000x500x250x100,1e-6,10]", # this is different
            "-s",
            "3.0x2.0x1.0x0.",
            "-f",
            "8x4x2x1", # this is different, and no x flag
            "-t",
            mysyn,
            "-m",
            "%s[%s,%s,1,%s]" % (syn_metric, f, m, syn_sampling),
            "-c",
            "[%s,1e-6,10]" % synits, # this is different
            "-s",
            "2.0x1.0x0.0", # this is different (but can be set)
            "-f",
            "3x2x1", # this is different (but can be set)
            "-u",
            "0", # this is different (cant be changed) (but in new version yes)
            "-z",
            "1", 
            "-o",
            "[%s,%s,%s]" % (outprefix, wmo, wfo),
            "-w", 
            "[0.005, 0.995]" # this is different, no o2nd x flag
        ]
    

    args.append("--float")
    args.append("1")
    args.append("--write-composite-transform")
    args.append(write_composite_transform * 1)
    if verbose:
        args.append("-v")
        args.append("1")
    print(args)
    processed_args = process_arguments(args)
    libfn = get_lib_fn("antsRegistration")
    if verbose:
        print("antsRegistration " + ' '.join(processed_args))
    reg_exit = libfn(processed_args)
    if (reg_exit != 0):
        raise RuntimeError(f"Registration failed with error code {reg_exit}")
    afffns = glob.glob(outprefix + "*" + "[0-9]GenericAffine.mat")
    fwarpfns = glob.glob(outprefix + "*" + "[0-9]Warp.nii.gz")
    iwarpfns = glob.glob(outprefix + "*" + "[0-9]InverseWarp.nii.gz")
    vfieldfns = glob.glob(outprefix + "*" + "[0-9]VelocityField.nii.gz")
    # print(afffns, fwarpfns, iwarpfns)
    if len(afffns) == 0:
        afffns = ""
    if len(fwarpfns) == 0:
        fwarpfns = ""
    if len(iwarpfns) == 0:
        iwarpfns = ""
    if len(vfieldfns) == 0:
        vfieldfns = ""

    alltx = sorted(
        set(glob.glob(outprefix + "*" + "[0-9]*"))
        - set(glob.glob(outprefix + "*VelocityField*"))
    )
    findinv = np.where(
        [re.search("[0-9]InverseWarp.nii.gz", ff) for ff in alltx]
    )[0]
    findfwd = np.where([re.search("[0-9]Warp.nii.gz", ff) for ff in alltx])[
        0
    ]
    if len(findinv) > 0:
        fwdtransforms = list(
            reversed(
                [ff for idx, ff in enumerate(alltx) if idx != findinv[0]]
            )
        )
        invtransforms = [
            ff for idx, ff in enumerate(alltx) if idx != findfwd[0]
        ]
    else:
        fwdtransforms = list(reversed(alltx))
        invtransforms = alltx

    if write_composite_transform:
        fwdtransforms = outprefix + "Composite.h5"
        invtransforms = outprefix + "InverseComposite.h5"

    if not vfieldfns:
        return {
            "warpedmovout": warpedmovout.clone(inpixeltype),
            "warpedfixout": warpedfixout.clone(inpixeltype),
            "fwdtransforms": fwdtransforms,
            "invtransforms": invtransforms,
        }
    else:
        return {
            "warpedmovout": warpedmovout.clone(inpixeltype),
            "warpedfixout": warpedfixout.clone(inpixeltype),
            "fwdtransforms": fwdtransforms,
            "invtransforms": invtransforms,
            "velocityfield": vfieldfns,
        }

def move_to_space(fixed: ANTsImage, moving: ANTsImage, transformlist: List[str], 
                      interpolator: str = 'linear', which_to_invert: Optional[List[bool]] = None) -> ANTsImage:
    kwargs = {
        "fixed": fixed, "moving": moving,
        "transformlist": transformlist, "interpolator": interpolator
    }
    if which_to_invert is not None:
        kwargs["whichtoinvert"] = which_to_invert
    return ants.apply_transforms(**kwargs) # type: ignore

def registration_ants(self, pet_path, anatomical_path, brain_path):

    brain_template_path = STATIC_FILES / 'avg_template_swap.nii.gz'
    brain_template = ants.image_read(str(brain_template_path))
    anatomical = ants.image_read(anatomical_path)
    pet = ants.image_read(pet_path) 
    anatomical_brain = ants.image_read(brain_path)
    #brain to template
    self.logger.info(f'Registering anatomical brain to template')
    brain_to_mni_reg = registration_helper(fixed = brain_template, 
                                        moving = anatomical_brain, 
                                        type_of_transform="SyN-adjusted", # added to ants registration file
                                        grad_step= 0.25,  
                                        flow_sigma=3.0,
                                        total_sigma=0.0,
                                        syn_metric='Mattes',  
                                        reg_iterations=(100, 50, 30), 
                                        outprefix=os.getcwd() + "/SyN")
    
    transform_to_mni = [brain_to_mni_reg['fwdtransforms'][0], 
                    brain_to_mni_reg['fwdtransforms'][1]]
    #ct to template
    self.logger.info('Moving anatomical to template space')
    anatomical_to_mni = move_to_space(fixed=brain_template, moving=anatomical, transformlist=transform_to_mni)
    #pet to ct
    self.logger.info('Resampling PET to anatomical')
    pet_to_anatomical_rsl = ants.resample_image_to_target(image=pet, target=anatomical)
    pet_to_anatomical = ants.registration(fixed=anatomical, moving=pet_to_anatomical_rsl, type_of_transform='Rigid')
    
    #pet to template
    self.logger.info('Moving PET to template space')
    pet_to_mni = move_to_space(fixed=brain_template, moving=pet_to_anatomical['warpedmovout'], transformlist=transform_to_mni)

    pet_to_ct_path = '/home/zuza/test/pettoct_ants.nii.gz'
    pet_to_mni_path = '/home/zuza/test/pettomni_ants.nii.gz'
    anatomical_to_mni_path = '/home/zuza/test/anatomical_tomni_ants.nii.gz'
    brain_to_mni_path = '/home/zuza/test/brain_tomni_ants.nii.gz'
    pet_path2 = '/home/zuza/test/pet_ants.nii.gz'
    ct_path2 = '/home/zuza/test/ct_ants.nii.gz'
    ants.image_write(pet, pet_path2)
    ants.image_write(anatomical, ct_path2)
    ants.image_write(pet_to_mni, pet_to_mni_path)
    ants.image_write(pet_to_anatomical['warpedmovout'], pet_to_ct_path)
    ants.image_write(anatomical_to_mni['warpedmovout'], anatomical_to_mni_path)
    ants.image_write(brain_to_mni_reg['warpedmovout'], brain_to_mni_path)

    return  pet_to_mni_path, anatomical_to_mni_path, brain_to_mni_path

def get_statistics(logger, pet_path, cerebellum_path, prediction):
    """
    Calculates various statistics on PET and prediction data, including median normalization, SBR,
    asymmetry calculations, and putamen/caudate ratios.

    Parameters:
    -----------
    logger : Logger object
        The logger used for logging information.
    pet_path : str
        File path to the PET NIfTI image.
    cerebellum_path : str
        File path to the cerebellum mask NIfTI image.
    prediction : np.array
        Prediction mask (segmentation) containing regions of interest.

    Returns:
    --------
    pet_normalized : np.array
        PET data normalized by the cerebellum cortex median.
    cerebellum_mask : np.array
        The mask of the cerebellum.
    data : dict
        A dictionary containing various statistics:
        - SBR values for putamen, caudate nucleus, striatum, and posterior putamen.
        - Ratios of putamen to caudate nucleus.
        - Asymmetry indices for different brain regions and ratios.

    Notes:
    ------
    - SBR (Specific Binding Ratio) is calculated as the ratio between the mean PET uptake
      in the target region and the median uptake in the cerebellum cortex.
    - Asymmetry index is calculated as the relative difference between left and right regions.
    """
    # Step 1: Load and normalize PET data by cerebellum cortex median
    logger.info('Calculating cerebellum cortex median')
    pet_data = nib.load(pet_path).get_fdata()
    pet_data = np.nan_to_num(pet_data, nan=0.0)
    cerebellum_mask = nib.load(cerebellum_path).get_fdata()

    # Calculate the median from the cerebellum region (where mask == CEREBELLUM_INDEX)
    cerebellum_median = np.median(pet_data[(cerebellum_mask == CEREBELLUM_INDEX)])

    if cerebellum_median == 0:
        logger.warning('Cerebellum median is zero, normalization might be invalid.')

    logger.info(f'Found cerebellum median: {cerebellum_median}')

    # Normalize PET data by the cerebellum median
    pet_normalized = pet_data / cerebellum_median - 1

    # Step 2: Split data and prediction masks into left and right hemispheres
    logger.info('Calculating statistics')
    pet_left = get_split(pet_data, 'left')
    pet_right = get_split(pet_data, 'right')
    prediction_left = get_split(prediction, 'left')
    prediction_right = get_split(prediction, 'right')

    # Step 3: Check each structure/hemisphere mask for emptiness up front.
    # This catches e.g. a surgically removed putamen or caudate nucleus before
    # any arithmetic (median/division/eig) is attempted on it.
    putamen_left_mask = prediction_left == 2
    putamen_right_mask = prediction_right == 2
    caudate_left_mask = prediction_left == 3
    caudate_right_mask = prediction_right == 3
    striatum_left_mask = prediction_left != 0
    striatum_right_mask = prediction_right != 0

    structure_masks = {
        'Putamen left': putamen_left_mask,
        'Putamen right': putamen_right_mask,
        'Caudate Nucleus left': caudate_left_mask,
        'Caudate Nucleus right': caudate_right_mask,
        'Striatum left': striatum_left_mask,
        'Striatum right': striatum_right_mask,
    }
    for label, mask in structure_masks.items():
        if not mask.any():
            logger.warning(f"{label}: structure not found in prediction (e.g. post-surgery); "
                            "will report as 0 in statistics and DICOM header.")

    # Step 4: Get posterior putamen masks, only if the putamen actually has voxels
    # in that hemisphere (skip the PCA-based posterior/anterior split otherwise)
    if putamen_left_mask.any():
        posterior_putamen_mask_left = get_posterior_putamen(prediction, 'left', logger)
    else:
        logger.warning("Putamen left: no voxels found; skipping posterior putamen calculation, reporting as 0.")
        posterior_putamen_mask_left = np.zeros_like(prediction)

    if putamen_right_mask.any():
        posterior_putamen_mask_right = get_posterior_putamen(prediction, 'right', logger)
    else:
        logger.warning("Putamen right: no voxels found; skipping posterior putamen calculation, reporting as 0.")
        posterior_putamen_mask_right = np.zeros_like(prediction)

    # Step 5: Calculate SBR (Specific Binding Ratio) for each region, safely
    putamen_left = get_sbr_safe(pet_left[putamen_left_mask], cerebellum_median, logger, 'Putamen left')
    putamen_right = get_sbr_safe(pet_right[putamen_right_mask], cerebellum_median, logger, 'Putamen right')
    caudate_left = get_sbr_safe(pet_left[caudate_left_mask], cerebellum_median, logger, 'Caudate Nucleus left')
    caudate_right = get_sbr_safe(pet_right[caudate_right_mask], cerebellum_median, logger, 'Caudate Nucleus right')
    striatum_left = get_sbr_safe(pet_left[striatum_left_mask], cerebellum_median, logger, 'Striatum left')
    striatum_right = get_sbr_safe(pet_right[striatum_right_mask], cerebellum_median, logger, 'Striatum right')
    posterior_putamen_left = get_sbr_safe(pet_data[posterior_putamen_mask_left == 2], cerebellum_median, logger, 'Posterior Putamen left')
    posterior_putamen_right = get_sbr_safe(pet_data[posterior_putamen_mask_right == 2], cerebellum_median, logger, 'Posterior Putamen right')

    # Step 6: Calculate ratios of putamen to caudate nucleus, safely
    ratio_left = safe_divide(putamen_left, caudate_left, logger, 'Putamen / Caudate Nucleus left')
    ratio_right = safe_divide(putamen_right, caudate_right, logger, 'Putamen / Caudate Nucleus right')
    ratio_posterior_left = safe_divide(posterior_putamen_left, caudate_left, logger, 'Posterior Putamen / Caudate Nucleus left')
    ratio_posterior_right = safe_divide(posterior_putamen_right, caudate_right, logger, 'Posterior Putamen / Caudate Nucleus right')

    # Step 7: Calculate asymmetry indices, safely
    caudate_asymmetry = get_asymmetry_safe(caudate_right, caudate_left, logger, 'Caudate Nucleus')
    putamen_asymmetry = get_asymmetry_safe(putamen_right, putamen_left, logger, 'Putamen')
    posterior_asymmetry = get_asymmetry_safe(posterior_putamen_right, posterior_putamen_left, logger, 'Posterior Putamen')
    striatum_asymmetry = get_asymmetry_safe(striatum_right, striatum_left, logger, 'Striatum')
    ratio_asymmetry = get_asymmetry_safe(ratio_right, ratio_left, logger, 'Putamen / Caudate Nucleus ratio')
    ratio_posterior_asymmetry = get_asymmetry_safe(ratio_posterior_right, ratio_posterior_left, logger, 'Posterior Putamen / Caudate Nucleus ratio')

    # Step 8: Compile statistics into a dictionary
    data = {
        'Putamen left': putamen_left,
        'Putamen right': putamen_right,
        'Caudate Nucleus left': caudate_left,
        'Caudate Nucleus right': caudate_right,
        'Striatum left': striatum_left,
        'Striatum right': striatum_right,
        'Posterior Putamen left': posterior_putamen_left,
        'Posterior Putamen right': posterior_putamen_right,
        'Putamen / Caudate Nucleus left': ratio_left,
        'Putamen / Caudate Nucleus right': ratio_right,
        'Posterior Putamen / Caudate Nucleus left': ratio_posterior_left,
        'Posterior Putamen / Caudate Nucleus right': ratio_posterior_right,
        'Putamen asymmetry': putamen_asymmetry,
        'Caudate Nucleus asymmetry': caudate_asymmetry,
        'Posterior Putamen asymmetry': posterior_asymmetry,
        'Striatum asymmetry': striatum_asymmetry,
        'Putamen / Caudate Nucleus asymmetry': ratio_asymmetry,
        'Posterior Putamen / Caudate Nucleus asymmetry': ratio_posterior_asymmetry
    }

    logger.info('Statistics calculation complete.')

    return pet_normalized, cerebellum_mask, data, cerebellum_median


def get_split(data, direction):
    """
    Splits image data into left or right hemispheres based on the direction.

    Parameters:
    -----------
    data : numpy array
        Image data to be split.
    direction : str
        Direction of split ('left' or 'right').

    Returns:
    --------
    numpy array
        The split data for the specified hemisphere.
    Raises:
    -------
    ValueError
        If the direction is not 'left' or 'right'.
    """
    if direction.lower() not in ['left', 'right']:
        raise ValueError("Invalid direction. Choose 'left' or 'right'.")

    if direction.lower() == 'left':
        # Return the left hemisphere (assumes image width is 256, splits at index 128)
        return data[:128, :, :]
    else:
        # Return the right hemisphere (second half starting from index 128)
        return data[128:, :, :]


def get_sbr(region_data, cerebellum_median):
    """
    Calculates the Specific Binding Ratio (SBR) by normalizing region data against the cerebellum gray matter median.

    Parameters:
    -----------
    region_data : numpy array
        Data of the brain region of interest.
    cerebellum_median : float
        Median value of the cerebellum gray matter used for normalization.

    Returns:
    --------
    float
        The SBR value for the region.
    """
    # Calculate median of the region and normalize it by cerebellum median, then subtract 1
    return np.median(region_data)/cerebellum_median - 1


def get_asymmetry(right, left):
    """
    Calculates the asymmetry between the right and left hemisphere regions.

    Parameters:
    -----------
    right : float
        Value representing the measurement from the right hemisphere.
    left : float
        Value representing the measurement from the left hemisphere.

    Returns:
    --------
    float
        The asymmetry value.
    """
    # Calculate asymmetry using the formula: (right - left) / (right + left)
    asymmetry = (right - left) / (right + left)
    return asymmetry


def get_sbr_safe(region_data, cerebellum_median, logger=None, label="region"):
    """
    Same as get_sbr, but returns 0.0 (with a logged warning) instead of NaN
    when the region is empty (structure not present in this hemisphere).
    """
    if region_data.size == 0:
        if logger is not None:
            logger.warning(f"{label}: no voxels found (structure absent); reporting SBR as 0.")
        return 0.0
    return get_sbr(region_data, cerebellum_median)


def get_asymmetry_safe(right, left, logger=None, label="region"):
    """
    Same as get_asymmetry, but returns 0.0 (with a logged warning) instead of
    NaN when right + left == 0 (e.g. structure absent bilaterally, or one side
    absent and the other legitimately at zero uptake).
    """
    denom = right + left
    if denom == 0:
        if logger is not None:
            logger.warning(f"{label}: asymmetry undefined (right + left == 0); reporting as 0.")
        return 0.0
    return (right - left) / denom


def safe_divide(numerator, denominator, logger=None, label="ratio"):
    """
    Divides numerator by denominator, returning 0.0 (with a logged warning)
    instead of raising/producing inf or NaN when denominator == 0
    (e.g. caudate absent, so putamen/caudate ratio is undefined).
    """
    if denominator == 0:
        if logger is not None:
            logger.warning(f"{label}: denominator is zero; reporting as 0.")
        return 0.0
    return numerator / denominator


def get_posterior_putamen( prediction_data, direction, logger):
    """
    Identifies the posterior putamen region from prediction data for a specified hemisphere.

    Parameters:
    -----------
    prediction_data : numpy array
        The data containing predicted regions (e.g., segmentation results).
    direction : str
        The hemisphere to process ('left' or 'right').

    Returns:
    --------
    numpy array
        A binary mask indicating the posterior putamen region.
    """
    # Create a copy of the prediction data to manipulate
    prediciton_hemisphere = prediction_data.copy()

    # Zero out the opposite hemisphere to isolate the hemisphere of interest
    if direction == 'left':
        prediciton_hemisphere[128:,:,:] = 0 # Zero out the right hemisphere
    else:
        prediciton_hemisphere[:128, :, :] = 0 # Zero out the left hemisphere

    # Identify voxels labeled as putamen (assumed to have label '2')
    putamen = prediciton_hemisphere == 2

    # Get the coordinates of non-zero voxels in the putamen region
    xo, zo, yo = np.nonzero(putamen)
    
    # Prepare an empty mask up front so we can return it early if needed
    putamen_posterior = np.zeros_like(prediction_data)

    # If the putamen is absent or has too few voxels, skip the
    # PCA-based split and return an empty posterior-putamen mask 
    if xo.size < 2:
        message = (f"Putamen not found (or too few voxels: {xo.size}) in the "
                    f"'{direction}' hemisphere; treating as absent (e.g. post-surgery). "
                    "Returning empty posterior putamen mask.")
        if logger is not None:
            logger.warning(message)
        return putamen_posterior
    
    # Center the coordinates relative to their mean
    x = xo - np.mean(xo)
    y = yo - np.mean(yo)
    z = zo - np.mean(zo)
    coordinates = np.vstack([x, y, z])

    # Calculate the covariance matrix and its eigenvectors for the coordinate distribution
    covariance = np.cov(coordinates)
    _, eigenvectors = np.linalg.eig(covariance)

    # Project the coordinates along the second principal axis (posterior-anterior direction)
    axis_to_use = np.dot(eigenvectors, coordinates)[1]

    # Identify the posterior half of the putamen
    posterior = axis_to_use < (max(axis_to_use) + min(axis_to_use)) / 2

    # Extract the coordinates of the posterior part
    coords_low = np.vstack([np.extract(posterior, xo),
                            np.extract(posterior, zo),
                            np.extract(posterior, yo)])

    # Create a mask for the posterior putamen
    putamen_posterior[tuple(coords_low)] = 2 # Set the posterior putamen voxels to 2 (same as putamen label)

    return putamen_posterior


def run_prediction(logger, model_file, labels, input_data, threshold=0.5):
    """
    Run a prediction on the input data using a pre-trained model and convert the output to an image format.

    Parameters:
    -----------
    model_file : str
        The file path to the pre-trained model.
    labels : list
        A list of labels corresponding to the classes for prediction.
    input_data : np.ndarray
        The input data for which predictions will be made.
    threshold : float, optional (default=0.5)
        The threshold value for converting model outputs to binary class predictions.

    Returns:
    --------
    np.ndarray
        A numpy array representing the predicted image after applying the threshold.

    Exceptions:
    -----------
    Raises an exception if there is an issue with loading the model or running the prediction.
    """
    try:
        # Load the pre-trained model
        model = load_model(logger, model_file)

        # Perform patch-wise prediction using the model
        prediction = patch_wise_prediction(model=model, data=input_data)[np.newaxis]

        # Convert the prediction to an image format based on the threshold and labels
        prediction_image = prediction_to_image(prediction, threshold=threshold, labels=labels)

        # Return the resulting prediction image
        return prediction_image

    except Exception as e:
        # Log the exception and re-raise it
        raise RuntimeError(f"Failed to run prediction: {e}")


def get_pet_dicom(self, pet_nii, ref_pet_dicom, modality_name, series_number):
    self.logger.info('Conversion of normalized PET nifti to DICOM')
    pet_data = pet_nii.get_fdata()
    # pet_data =  pet_data.T
    pet_data = (pet_nii.get_fdata().T)[::-1,:, :] # maybe this not necessery
    ref_pet_dicom.sort(key=lambda dcm: int(dcm.InstanceNumber))

    PET_BLUEPRINT = Blueprint([
        # 0008
        StaticElement(0x0008_0008, 'CS', ['DERIVED', 'SECONDARY']),
        CopyOrElseElement(0x0008_0014, 'LO', 'MISSING'),
        StaticElement(0x0008_0016, 'UI', PositronEmissionTomographyImageStorage),
        FunctionalElement(0x0008_0018, 'UI', add_UID_tag),
        CopyOrElseElement(0x0008_0021, 'DA', "MISSING"), # Series Date
        CopyOrElseElement(0x0008_0022, 'LO', 'MISSING'), # Acquisition Date
        FunctionalElement(0x0008_0023, 'DA', get_today), # Content Date
        CopyOrElseElement(0x0008_0031, 'TM', "MISSING"), # Series Time
        CopyOrElseElement(0x0008_0032, 'LO', 'MISSING'), # Acquisition Time
        FunctionalElement(0x0008_0033, 'TM', get_time), # Content Time
        CopyOrElseElement(0x0008_0050, 'SH', "MISSING"),
        StaticElement(0x0008_0060, 'CS', 'PT'),
        StaticElement(0x0008_0070, 'LO', 'PE2I pipeline'),
        CopyOrElseElement(0x0008_0080, 'LO', 'MISSING'),
        CopyOrElseElement(0x0008_1030, 'LO', "PE2I PIPELINE PRODUCED STUDY"),
        StaticElement(0x0008_103E, 'LO', modality_name),
        StaticElement(0x0008_1090, 'LO', 'PE2I pipeline'),

        # 0010 ____
        CopyOrElseElement(0x0010_0010, 'PN', "Missing Patient Name"),
        CopyOrElseElement(0x0010_0020, 'LO', "Missing Patient ID"),
        CopyOrElseElement(0x0010_0030, 'DA', datetime.date(1970,1,1)),

        # 0018 ____
        # StaticElement(0x0018_0050, 'DS', z_dim),
        CopyElement(0x0018_0050),
        CopyOrElseElement(0x0018_0060, 'LO', ''),
        CopyOrElseElement(0x0018_1000, 'LO', 'MISSING'),
        CopyOrElseElement(0x0018_1020, 'LO', 'MISSING'),
        CopyOrElseElement(0x0018_1181, 'LO', 'MISSING'),
        CopyOrElseElement(0x0018_1242, 'LO', 'MISSING'),
        CopyOrElseElement(0x0018_5100, 'LO', 'MISSING'),
        #CopyOrElseElement(0x0018_5100, Optional=True), # Patient Position

        # 0020 ____
        SeriesElement(0x0020_000E, 'UI', add_UID_tag),
        CopyOrElseElement(0x0020_000D, 'UI', SeriesElement(0x0020_000D, 'UI', add_UID_tag)),
        CopyOrElseElement(0x0020_0010, 'SH', "Missing Study ID"),
        StaticElement(0x0020_0011, 'IS', series_number), # Series Number
        InstanceCopyElement(0x0020_0032, 'DS'),
        # FunctionalElement(0x0020_0032, 'DS', add_patient_position),
        CopyOrElseElement(0x0020_0037, 'LO', 'MISSING'),
        CopyOrElseElement(0x0020_0052, 'LO', 'MISSING'),
        CopyOrElseElement(0x0020_1040, 'LO', 'MISSING'),
        StaticElement(0x0020_4000, 'LO', 'Unit: SBR'),
        #InstanceCopyOrElseElement(0x0020_1041, 'DS'),
        # InstanceCopyElement(0x0020_1041, 'DS'),
        InstanceCopyElement(0x0020_1041, 'DS'),

        # FunctionalElement(0x0020_1041, 'DS', add_slice_location),


        # 0028 ____
        # StaticElement(0x0028_0030, 'DS', [x_dim, y_dim]),
        CopyElement(0x0028_0030),
        CopyOrElseElement(0x0028_0051, 'LO', 'MISSING'),
        CopyOrElseElement(0x0028_1054, 'LO', 'MISSING'),

        # 0054 ____
        CopyOrElseElement(0x0054_0013, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_0081, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_0101, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_0410, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_0414, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_1002, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_1100, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_1105, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_1300, 'LO', 'MISSING'),
        CopyOrElseElement(0x0054_0016, 'LO', 'MISSING'),  #RadiopharmaceuticalInformationSequence
        StaticElement(0x0054_1000, 'CS', ["STATIC", "IMAGE"]),
        StaticElement(0x0054_1001, 'CS', "MLML"),
        StaticElement(0x0054_1102, 'CS', "NONE"),

        # FunctionalElement(0x0054_1330, 'US', add_image_index),
        InstanceCopyElement(0x0054_1330, 'US'),
        # FunctionalElement(0x0054_1330, 'US', add_image_index),
    ])

    # swapped_pet_data = np.swapaxes(pet_data.T, 1, 2)[:,::-1,:]
    # transposed_data = transpose_nifti_coords(swapped_pet_data)

    factory = DicomFactory()

    return factory.build_series(
        pet_data, PET_BLUEPRINT, ref_pet_dicom
        # transposed_data, PET_BLUEPRINT, ref_pet_dicom
    )

############## report part ############
def get_logo(institution: str):
    """
    Retrieves the logo abbreviation based on the institution name.

    Parameters:
    -----------
    institution : str
        The name of the institution.

    Returns:
    --------
    str
        The abbreviation for the logo corresponding to the institution.
    """

    # Map of institutions to their logo abbreviations
    logo_map = {
        'Bispebjerg': 'BBH',
        'Rigshospitalet': 'RH',
        'Nuclearmedicin': 'RH',
        'RH Glostrup': 'RH',
        'AUH': 'AUH'
    }

    # Iterate through the map to find a match
    for key, logo in logo_map.items():
        if key in institution:
            return logo


def get_footnote(institution):
    """
    Retrieves the footnote text based on the institution name.

    Parameters:
    -----------
    institution : str
        The name of the institution.

    Returns:
    --------
    str
        The footnote text corresponding to the institution.
    """

    # Map of institutions to their footnote text
    footnote_map ={
        'Bispebjerg': (r'Bispebjerg og Frederiksberg Hospital\\' +
                       r'Klinisk Fysiologisk / Nuklearmedicinsk Afdeling\\' +
                       r'Indgang 60, Ebba Lunds Vej 44\\' +
                       r'2400 København NV'),
        'Rigshospitalet': (r'Klinik for Klinisk Fysiologi, ' +
                          r'Nuklear medicin og PET\\ Rigshospitalet ' +
                          r'\\ Blegdamsvej 9\\  ' +
                          '2100 København Ø'),
        'RH Glostrup': (r'Klinik for Klinisk Fysiologi, Nuklear medicin og PET\\' +
                     r'Rigshospitalet-Glostrup\\' +
                     r'Valdemar Hansens Vej 1-23\\ ' +
                     '2600 Glostrup')}

    # Iterate through the map to find a match
    for key, footnote in footnote_map.items():
        if key in institution:
            return footnote


def get_name(patient_name):
    """
    Formats the patient's name by reordering the surname and first name.

    Parameters:
    -----------
    patient_name : str
        Patient's name in the format "Surname^FirstName".

    Returns:
    --------
    str
        Formatted name in the format "FirstName Surname".
    """

    # Convert the patient name to a string and split it by '^'
    name_str = str(patient_name)
    name_parts = name_str.split('^')

    # Extract surname and first name
    #surname = name_parts[0]
    #first_name = name_parts[1]

    if len(name_parts) == 0:
        return "Ukendt Navn"
    elif len(name_parts) == 1:
        return name_parts[0]
    else:
        # Return the name in "FirstName Surname" format
        return name_parts[0] + ' ' + name_parts[1]


def get_age(patient_age):
    """
    Converts the patient's age from a string in format 'YXX' to an integer.

    Parameters:
    -----------
    patient_age : str
        Patient's age in format 'YXX' where 'Y' is a single digit year and 'XX' is the age.

    Returns:
    --------
    int
        The patient's age as an integer.
    """

    # Remove the last character (usually 'Y' for year) from the age string
    patient_age = patient_age[:-1]

    # Convert the age to an integer, accounting for single digit years
    if patient_age[0] == '0':
        return int(patient_age[1:])
    else:
        return int(patient_age)


def get_date(study_date):
    """
    Formats the study date from 'YYYYMMDD' to 'DD/MM/YYYY'.

    Parameters:
    -----------
    study_date : str
        The study date in 'YYYYMMDD' format.

    Returns:
    --------
    str
        The study date formatted as 'DD/MM/YYYY'.
    """

    # Format the date string from 'YYYYMMDD' to 'DD/MM/YYYY'
    return f'{study_date[-2:]}/{study_date[4:6]}/{study_date[:4]}'


def get_report_header(doc, institution):
    """
    Adds a header section to the LaTeX document, including a logo and institution information.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object to which the header will be added.
    institution : str
        The name of the institution to determine the logo and footnote.
    """

    # Get the logo filename based on the institution
    logo_file = get_logo(institution)
    icon_path=f'{STATIC_FILES}/{logo_file}.png'

    # Center the header content
    doc.append(Command('centering'))
    ##### TODO changed for test
    # # Add the logo image to the header
    # doc.append(NoEscape(r'\begin{minipage}{0.6\textwidth}'))  # Adjust width as needed
    # doc.append(NoEscape(r'\includegraphics[height=1cm]{%s}' % icon_path))
    # doc.append(NoEscape(r'\end{minipage}'))

    # # Add the footnote about the institution
    # doc.append(NoEscape(r'\begin{minipage}{0.5\textwidth}'))  # Adjust width as needed
    # doc.append(NoEscape(r'{\footnotesize \begin{tabular}{r}' +
    #                        get_footnote(institution) +
    #                        r'\end{tabular}}'))
    # doc.append(NoEscape(r'\end{minipage}'))
    # doc.append(NoEscape(r'\newline'))
    ##### TODO changed for test

    address = get_footnote(institution)

    doc.append(NoEscape(r'\noindent'))
    # Minipage for the logo (left part)
    doc.append(NoEscape(r'\begin{minipage}[b]{0.5\linewidth}'))
    doc.append(NoEscape(r'\includegraphics[height=1.2cm]{%s}' % icon_path))
    doc.append(NoEscape(r'\end{minipage}%'))
    # Minipage for the address (right part)
    doc.append(NoEscape(r'\begin{minipage}[b]{0.5\linewidth}'))
    doc.append(NoEscape(r'\raggedleft')) # Right-aligns all content in this minipage
    doc.append(NoEscape(r'\footnotesize '))
    doc.append(NoEscape(address))
    doc.append(NoEscape(r'\end{minipage}'))
    doc.append(NoEscape(r'\newline'))


def create_document(fname):
    """
    Creates a LaTeX document with specified formatting and options.

    Parameters:
    -----------
    fname : str
        The filename for the LaTeX document.

    Returns:
    --------
    Document
        The LaTeX document object with the specified options.
    """

    # Define the page geometry options
    geometry_options = {'top': '0.5in',
                        'left': '0.45in',
                        'right': '0.45in',
                        'bottom': '0.8in'}

    # Create a new LaTeX document with the specified geometry options
    doc = Document(os.path.join(os.getcwd(), fname),
                   geometry_options=geometry_options, lmodern=False, document_options='12pt')

    # Add the graphicx package for handling images
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('tabularx')) ### TODO new code

    return doc


def get_patient_table(doc, ref, pt_age):
    """
    Adds a table with patient information to the LaTeX document.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object to which the table will be added.
    ref : Dataset
        The DICOM dataset from which patient information will be extracted.
    """

    # Dataset might not have all the data we wish to display
    patient_name = "Missing name"
    if 'PatientName' in ref:
        patient_name = escape_latex(get_name(ref.PatientName))

    patient_id = "Missing ID"
    if "PatientID" in ref:
        patient_id = escape_latex(ref.PatientID)

    patient_age = pt_age

    patient_sex = "Missing sex"
    if "PatientSex" in ref:
         patient_sex = escape_latex(ref.PatientSex)

    study_date = "Missing study date"
    if 'StudyDate' in ref:
        study_date = escape_latex(get_date(ref.StudyDate))

    patient_weight = "Missing weight"
    if "PatientWeight" in ref:
        patient_weight = int(ref.PatientWeight)

    patient_dose = "Missing dosis"
    if "RadiopharmaceuticalInformationSequence" in ref:
        seq = ref.RadiopharmaceuticalInformationSequence[0]
        if "RadionuclideTotalDose" in seq:
            patient_dose = int(seq.RadionuclideTotalDose / 1e6)

    # Create the table data including headers and patient information
    table_data = [
        ['Patient name', 'CPR', 'Age', 'Sex', 'Scan date', 'Weight [kg]', 'Dose [MBq]'],
        [patient_name, patient_id, patient_age, patient_sex, study_date, patient_weight, patient_dose],
    ]

    # # Center the table content
    # doc.append(NoEscape(r'\centering'))

    # # Create the table with column alignment
    # with doc.create(Tabular('lccccccc')) as table:
    #     table.append(NoEscape(r'\hline'))

    #     # Add header row with bold text
    #     table.append(NoEscape(' & '.join([r'\textbf{' + element + '}' for element in table_data[0]]) + r' \\ '))

    #      # Add data rows
    #     for row in table_data[1:]:
    #         table.append(NoEscape(' & '.join(map(str, row)) + r' \\'))

    #     table.append(NoEscape(r'\hline'))
    # TODO this code added
    with doc.create(Tabularx('Xcccccc', width_argument=NoEscape(r'\linewidth'))) as table:
        table.add_hline()
        # Add header row with bold text
        table.add_row(table_data[0], mapper=bold)
        # Add data rows
        for row in table_data[1:]:
            table.add_row(row)
        table.add_hline()


def contours_mask_slice(slice):
    '''
    Computes the contour of a 2D binary image.

    Parameters:
    -----------
    slice : array
        A 2D numpy array representing the binary image.

    Returns:
    --------
    array
        A binary image containing only the contour of the segmentation.
    '''

    # Find contours in the binary image
    contours, _ = cv.findContours(slice.astype(np.uint8).copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw contours
    mask = np.zeros((256, 256), dtype=np.uint8)

    # Draw the contours onto the mask
    cv.drawContours(mask, contours, -1, 1, 4)

    # Return the logical 'and' between the inverse of the original slice and the mask
    return np.logical_and(np.logical_not(slice), mask)


# def get_overlaying_plots(axes, segmentations, image, min_value, max_value, c_map=_PETRainbowCMAP, c_map_contour='plasma_r'):
def get_overlaying_plots(axes, segmentations, image, min_value, max_value, normalization, c_map=_PETRainbowCMAP, c_map_contour='plasma_r'):

    '''
    Superimposes contour of segmentations on the current PET image slice.
    The contours are color-mapped dynamically according to the underlying PET image.

    Parameters:
    -----------
    axes : matplotlib.axes.Axes
        Matplotlib axes object where the contours and image will be plotted.

    segmentations : numpy.ndarray
        Binary or labeled segmentation mask whose contours will be overlaid on the image.

    image : numpy.ndarray
        PET image data to which the segmentations will be overlaid.

    min_value : float
        Minimum value for the color scaling of the PET image.

    max_value : float
        Maximum value for the color scaling of the PET image.

    c_map : matplotlib.colors.Colormap, optional
        Colormap to use for the PET image. Default is '_PETRainbowCMAP'.

    Returns:
    --------
    None
    '''

    # Extract the contours of the segmentations
    segmentations_cont = contours_mask_slice(segmentations)[FIRST_DIM_CROPPED, SECOND_DIM_CROPPED]

    # Mask the PET image where segmentations are not present
    im_ma = np.ma.array(image, mask=np.logical_not(segmentations_cont))

    # Display the PET image
    # kwargs = {'interpolation': 'none', 'vmin': min_value, 'vmax': max_value}
    # axes.imshow(np.rot90(image), cmap=c_map, **kwargs)
    if normalization:
        norm = MidPointNorm(vmin=min_value, vmax=max_value, midpoints=MID_POINTS, defmids=DEF_MIDS)
        axes.imshow(np.rot90(image), aspect='equal', norm=norm, cmap=c_map)
    else:
        kwargs = {'interpolation': 'none', 'vmin': min_value, 'vmax': max_value}
        axes.imshow(np.rot90(image), cmap=c_map, **kwargs)

    # Overlay the masked segmentation contours with an 'autumn' colormap
    kwargs = {'interpolation': 'none', 'vmin': min_value, 'vmax': max_value}
    axes.imshow(np.rot90(im_ma), cmap=c_map_contour, **kwargs)


def get_image_sides(axes):
    '''
    Annotates the left and right sides of an image slice with 'L' and 'R'.

    Parameters:
    -----------
    axes : matplotlib.axes.Axes
        Matplotlib axes object to which the text will be added.

    Returns:
    --------
    None
    '''
    # Add text annotation for the right side
    axes.text(SECOND_DIM[1] * 0.05,
              SECOND_DIM[1] * 0.1,
              'R', color='#f9f9f9', fontsize=25)

    # Add text annotation for the left side
    axes.text(SECOND_DIM[1] * 0.89,
              SECOND_DIM[1] * 0.1,
              'L', color='#f9f9f9', fontsize=25)


def add_colormap_plot(doc, plt, vmin, vmax, step, add_max_tick=False,
                       subfig_width=NoEscape(r'0.9\linewidth'),
                       plot_width=NoEscape(r'0.3\textwidth')):
    """
    Adds a plot showing a colormap to the provided LaTeX document.

    Parameters:
    -----------
    doc : Document
        The LaTeX Document object to which the plot will be added.

    plt : matplotlib.pyplot
        Active matplotlib plot object used for generating the colormap plot.

    vmin : float
        Minimum value for the colormap range.

    vmax : float
        Maximum value for the colormap range.

    step : float
        Step size for tick marks on the colormap plot.

    add_max_tick : bool, optional
        Whether to add the maximum tick value on the colormap plot. Default is False.

    subfig_width : NoEscape
        Width of the subfigure in LaTeX units. Default is '0.9\linewidth'.

    plot_width : NoEscape
        Width of the plot in LaTeX units. Default is '0.3\textwidth'.


    Returns:
    --------
    None
    """
    # Create a subfigure in the LaTeX document
    with doc.create(SubFigure(width=subfig_width,position='t')) as subplot2:
        # Center the plot within the subfigure
        doc.append(Command('centering'))

        # Create a new matplotlib figure and axes
        fig, axes = plt.subplots(1, 1, figsize=(4, 1))

        # Display the colormap as a gradient
        axes.imshow(
            np.rot90(np.outer(np.arange(vmin, vmax, 0.01), np.ones(5))),
            cmap=_PETRainbowCMAP,
            extent=[vmin, vmax, vmin, vmax],
            origin='lower',
            aspect=0.1
        )

        # Set tick marks on the colormap
        ticks = np.arange(vmin, vmax, step)
        if vmax not in ticks and add_max_tick:
            ticks = np.append(ticks, vmax)

        plt.xticks(np.round(ticks, 2))

        # Customize tick label font size
        for label in axes.get_xticklabels():
            label.set_fontsize(13)

        # Set the title of the colormap plot
        axes.title.set_text('Specific Binding Ratio')
        axes.get_yaxis().set_visible(False)

        # Add the plot to the LaTeX document
        subplot2.add_plot(width=plot_width)
        plt.close(fig)  # Close the figure to free memory


def get_slices(masks, indices, num_slices):
    """
    Extracts a subset of slices from a 3D mask array based on specified indices.

    Parameters:
    -----------
    masks : np.ndarray
        3D array representing the mask.
    indices : list of int
        List of indices to include in the mask.
    num_slices : int
        Number of slices to return.

    Returns:
    --------
    list of int
        Selected slice numbers.
    """

    # Create a mask for the specified indices
    if len(indices) == 1:
        mask = (masks == indices[0])
    elif len(indices) == 2:
        mask = (masks == indices[0]) | (masks == indices[1])

    # Set the selected mask regions to 1
    masks[mask] = 1

    # Extract slice numbers from the mask
    slice_numbers = np.unique([x[2] for x in np.argwhere(masks)])
    indices = np.linspace(0, len(slice_numbers) - 1, num=num_slices, dtype=int)
    return [slice_numbers[i] for i in indices]
    # slice_numbers = sorted(np.unique([x[2] for x in np.argwhere(masks)]))

    # # Select and return the slices based on the interval
    # return  slice_numbers[0::int(np.ceil((len(slice_numbers) * 1.0) / num_slices))]


def get_first_plots(doc, pet, mask, slices):
    """
    Generates and adds the first set of plots to the document, including PET images with overlaid segmentation contours.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object to which plots will be added.
    pet : array
        Numpy array containing the PET image data.
    mask : array
        Numpy array containing the segmentation mask data.
    slices : list
        List of slice indices to be used for plotting.

    Returns:
    --------
    None
    """
    # Setting minimum and maximum value for PET data display
    min_value = -1
    max_value = 0.9 * np.max(pet*(mask.astype(bool)))

    # Create a new figure in the LaTeX document
    with doc.create(Figure(position='h!')) as plot:
        doc.append(Command('centering'))
        # Create a subfigure for the plots
        with doc.create(SubFigure(position='t', width=NoEscape(r'0.9\linewidth'))) as subplot1:
            # Create a matplotlib figure with 1 row and 3 columns of subplots
            fig, axes = plt.subplots(
                1, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(15, 5)
            )

            # Adjust subplot margins to remove unwanted spacing around the plots
            margins = {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
            fig.subplots_adjust(**margins)

             # The mask is the segmentation data which will be overlaid on the PET image
            segmentations = mask

            # Loop over slices indices 3 to 5 to generate three plots (one for each slice)
            for i, k in enumerate(range(3, 6)):
                axes[i].axis('off')
                ind = slices[k]
                pet_crop = pet[FIRST_DIM_CROPPED,SECOND_DIM_CROPPED, ind]

                # Generate the overlay plot of the PET image and the segmentation mask
                get_overlaying_plots(axes[i], segmentations[:, :, ind], pet_crop, min_value, max_value, True)
                # Adding letter to distinguish between directions
                get_image_sides(axes[i])

            subplot1.add_plot()
            plt.close(fig)  # Close the figure to free memory

        doc.append(NoEscape(r'\par \vfill'))

        # Add a color map to the LaTeX document
        add_colormap_plot(doc, plt, vmin=0, vmax=max_value, step=1 if max_value < 5 else 2)

    doc.append(NoEscape(r'\vspace*{-0.3cm}'))


def plot_collapse_pet(img_pet, seg, axial_slices, normalization):
    """
    Creates a plot of an axially collapsed PET image by averaging over specified slices.

    Parameters:
    -----------
    img_pet : array
        PET image given as a numpy array.
    seg : array
        Numpy array with relevant segmentations used for cropping.
    axial_slices : slice
        Slices over which the PET image should be collapsed.
    vmin : int
        Minimum value used for normalization when plotting.
    vmax : int
        Maximum value used for normalization when plotting.

    Returns:
    --------
    None
    """
    # Compute mean over the specified slices
    img_pet_collapse = (img_pet[: , :,axial_slices]).mean(axis=2)

    # Find coordinates of relevant structures in the segmentation
    coords = np.argwhere((seg == 2 ) | (seg ==3))

    # Define cropping bounds with a 10-pixel margin
    xmin = np.amin(coords[:, 0]) - 10
    xmax = np.amax(coords[:, 0]) + 10
    ymin = np.amin(coords[:, 2]) - 10
    ymax = np.amax(coords[:, 2]) + 10

    # Adjust x and y dimensions to be the same if necessary
    xrang = xmax - xmin
    yrang = ymax - ymin

    if xrang > yrang:
        xslices = slice(xmin, xmax)
        yslices = slice(ymin - int(np.floor((xrang - yrang) / 2.0)), ymax + int(np.ceil((xrang - yrang) / 2.0)))
    else:
        yslices = slice(ymin, ymax)
        xslices = slice(xmin - int(np.floor((yrang - xrang) / 2.0)), xmax + int(np.ceil((yrang - xrang) / 2.0)))

    # Create and configure the plot
    fig, axes = plt.subplots(1, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(15, 15))
    margins = {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
    fig.subplots_adjust(**margins)

    # Showing of collapsed PET images
    axes.axis('off')
    axes.imshow(np.rot90(img_pet_collapse[xslices, yslices]), norm=normalization, aspect='equal', cmap=_PETRainbowCMAP)
    axes.text(2, 5, 'R', color='#f9f9f9', fontsize=45)
    axes.text(xrang - 5, 5, 'L', color='#f9f9f9', fontsize=45)

    return fig

def add_average_plot(doc, norm_pet, mask, title, subfig_width, colormap_width, min_value, max_value, slices):
    """
    Adds an average plot to the document, showing a collapsed PET image over specified slices.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object to which the plot will be added.
    norm_pet : array
        Normalized PET image data.
    mask : array
        Numpy array with segmentation mask data.
    title : str
        Title for the plot.
    subfig_width : str
        Width of the subfigure in LaTeX.
    colormap_width : str
        Width of the colormap plot in LaTeX.
    min_value : float
        Minimum value for normalization when plotting.
    max_value : float
        Maximum value for normalization when plotting.
    slices : list
        List of slice indices to be used for plotting.

    Returns:
    --------
    None
    """
    # Set up a normalization for the PET data using the given min_value and max_value
    normalization = MidPointNorm(min_value, max_value, midpoints=MID_POINTS, defmids=DEF_MIDS)
    with doc.create(SubFigure(width=NoEscape(subfig_width))) as subplot:
        doc.append(Command('centering'))
        doc.append(NoEscape(r'{\small\textbf{' + title + r'}}\\'))
        doc.append(NoEscape(r'\vspace{0.2cm}'))

        # Generate the collapsed PET image plot
        fig = plot_collapse_pet(
            norm_pet,
            mask,
            slice(slices[6]-1, slices[0]+1),
            normalization
        )

        subplot.add_plot(width=NoEscape(subfig_width))
        plt.close(fig)  # Close the figure to free memory
        
        doc.append(NoEscape(r'\par \vfill'))

        # Determine if a maximum tick should be added to the color map based on the plot title
        if title == 'Absolute scale':
            add_max_tick = True
        else:
            add_max_tick = False

        # Add the color map to the LaTeX document
        add_colormap_plot(doc, plt, vmin=0, vmax=max_value, step=1, add_max_tick=add_max_tick, subfig_width=subfig_width, plot_width=colormap_width)
        doc.append(NoEscape(r'\newline'))
        doc.append(NoEscape(r' {\small Average intensity of top 6 axial slices. }  '))
        doc.append(NoEscape(r'\vspace{-0.2cm}'))


def plot_average(doc, pet, mask, slices):
    """
    Generates average plots for PET images in both relative and absolute scales.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object to which plots will be added.
    pet : array
        Numpy array containing the PET image data.
    mask : array
        Numpy array containing the segmentation mask data.
    slices : list
        List of slice indices to be used for plotting.

    Returns:
    --------
    None
    """

    doc.append(NoEscape(r'\vspace{-0.2cm}'))

    # Define the minimum value for normalization
    min_value = - 1

    # Calculate the maximum value for the relative scale
    avgmax = 0.9 * np.max(pet[:, :, (slices[6]-1):(slices[0]+1)].mean(axis=2))

    # Set a fixed maximum value for the absolute scale
    max_value_absolute = 4

    with doc.create(Figure(position='!htp')):
        doc.append(Command('centering'))
        subfig_width = r'8cm'
        colormap_width = r'5cm'

        # Generate the relative scale plot and add it to the document
        add_average_plot(doc, pet, mask, 'Relative scale', subfig_width, colormap_width, min_value, max_value=avgmax, slices=slices)

        # Generate the absolute scale plot and add it to the document
        add_average_plot(doc, pet, mask, 'Absolute scale', subfig_width, colormap_width, min_value, max_value=max_value_absolute, slices=slices)


def plot_nine_pet(doc, pet, mask, pet_desc, slices):
    """
    Plots a 3x3 grid of PET images with overlays of segmentations.

    Parameters:
    -----------
    doc : Document
        LaTeX document to append the plot.
    pet : array
        PET image as a numpy array.
    mask : array
        Segmentation mask for the PET image.
    pet_desc : str
        Description of the PET study.
    slices : list
        List of slice indices to use for plotting.

    Returns:
    --------
    None
    """

    # Define intensity range for PET display
    # min_value = -0.2
    min_value = - 1
    max_value = 0.9 * np.max(pet*(mask.astype(bool)))
    doc.append(NoEscape(r'\vspace*{-0.3cm}'))

    # Create figure for PET plots
    with doc.create(Figure(position='h!')) as plot:
        doc.append(Command('centering'))
        with doc.create(SubFigure(position='t', width=NoEscape(r'0.8\linewidth'))) as subplot1:
            # Create a 3x3 grid for displaying PET images
            fig, axes = plt.subplots(3, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(15, 15))

            # Adjust margins to minimize whitespace
            margins = {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
            fig.subplots_adjust(**margins)

            segmentations = mask

            # Iterate through 3x3 grid positions to plot PET slices
            for k, (i, j) in enumerate([(i, j) for i in range(0, 3) for j in range(0, 3)]):
                axes[i, j].axis('off')  # Turn off axis for each subplot
                ind = slices[k]  # Select slice index

                pet_crop = pet[FIRST_DIM_CROPPED, SECOND_DIM_CROPPED, ind]  # Crop the PET image

                # Plot PET image with segmentation overlay
                get_overlaying_plots(axes[i, j], segmentations[:, :, ind], pet_crop, min_value, max_value, normalization=True)
                get_image_sides(axes[i, j])  # Add R/L labels

            subplot1.add_plot()  # Add plot to the LaTeX document
            plt.close(fig) # Close the figure to free memory
            
        doc.append(NoEscape(r'\par \vfill'))

        # Add study description to the LaTeX document
        doc.append(NoEscape(r'{\scriptsize{' + pet_desc + r'}}\\'))

        # Add colormap to the LaTeX document
        add_colormap_plot(doc, plt, vmin=0, vmax=max_value, step=1 if max_value < 5 else 2)


def plot_ct(doc, anatomical, mask, anatomical_desc, slices, MR=False):
    """
    Plots a 3x3 grid of anatomical images with overlays of segmentations.

    Parameters:
    -----------
    doc : Document
        LaTeX document to append the plot.
    anatomical : array
        anatomical image as a numpy array.
    mask : array
        Segmentation mask for the anatomical image.
    anatomical_desc : str
        Description of the anatomical study.
    slices : list
        List of slice indices to use for plotting.

    Returns:
    --------
    None
    """

    # Define intensity range for anatomical display
    min_value = 0
    max_value = 100


    doc.append(NoEscape(r'\vspace*{-0.3cm}'))

    # Create figure for anatomical plots
    with doc.create(Figure(position='h!')) as plot:
        doc.append(Command('centering'))
        with doc.create(SubFigure(position='t', width=NoEscape(r'0.8\linewidth'))) as subplot1:
            # Create a 3x3 grid for displaying anatomical images
            fig, axes = plt.subplots(3, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(15, 15))

            # Adjust margins to minimize whitespace
            margins = {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
            fig.subplots_adjust(**margins)

            cmap = 'Greys_r'  # Define colormap for anatomical images
            segmentations = mask

            # Iterate through 3x3 grid positions to plot anatomical slices
            for k, (i, j) in enumerate([(i, j) for i in range(0, 3) for j in range(0, 3)]):
                axes[i, j].axis('off')  # Turn off axis for each subplot
                ind = slices[k]  # Select slice index
                anatomical_crop = anatomical[FIRST_DIM_CROPPED, SECOND_DIM_CROPPED, ind]  # Crop the anatomical image

                # Plot anatomical image with segmentation overlay
                get_overlaying_plots(axes[i, j], segmentations[:, :, ind], anatomical_crop, min_value, max_value, False, cmap, c_map_contour='autumn_r')
                get_image_sides(axes[i, j])  # Add R/L labels

            subplot1.add_plot()  # Add plot to the LaTeX document
            plt.close(fig) # Close the figure to free memory

    doc.append(NoEscape(r'\vspace{-0.7cm}'))

    # Add study description to the LaTeX document
    doc.append(NoEscape(r'{\scriptsize{' + anatomical_desc + r'}}\\'))
    if MR:
        doc.append(
        NoEscape(r'{\hspace*{0.3cm}\footnotesize Synthetic AI generated CT Cerebrum based on the MRI Dixon sequence. ' +
                 r'Only for anatomical reference.}'))
    else:
        # Add note on anatomical image usage for anatomical reference
        doc.append(
            NoEscape(r'{\hspace*{0.3cm}\footnotesize CT is for anatomical reference and is ' +
                    r'not for clinical reading. Please refer to original CT for this purpose.}'))


def produce_row(patient_values, name, normal_stat_values, pt_age):
    """
    Generates table rows for specific brain regions (Putamen, Caudate Nucleus) and hemispheres.

    Parameters:
    -----------
    patient_values : DataFrame
        Dataframe of patient values.
    name : str
        Name of the brain region.
    normal_stat_values : array
        Statistical values for normal references.
    pt_age : int
        Age of the patient.

    Yields:
    -------
    Table rows for each region and hemisphere.
    """
    # Extract patient and statistical values for right and left hemispheres
    right = patient_values[name + ' right']
    left = patient_values[name + ' left']
    sigma_left = normal_stat_values[name + ' left'].loc['sigma']
    sigma_right = normal_stat_values[name + ' right'].loc['sigma']
    slope_left = normal_stat_values[name + ' left'].loc['slope']
    slope_right = normal_stat_values[name + ' right'].loc['slope']
    intercept_left = normal_stat_values[name + ' left'].loc['intercept']
    intercept_right = normal_stat_values[name + ' right'].loc['intercept']

    # Calculate mean (mu) and lower limit of normal (LLN) values
    mu_right = slope_right*pt_age+intercept_right
    mu_left = slope_left*pt_age+intercept_left
    lln_left = mu_left - 2 * sigma_left
    lln_right = mu_right - 2 * sigma_right

    # Iterate for both right and left hemispheres and create plots
    for name, hemisphere, quantity, lln, mu, sigma in [(name, "Right", right, lln_right, mu_right, sigma_right),
                                                       ("", "Left", left, lln_left, mu_left, sigma_left)]:

        plot_path = normal_reference_SD_plot(mu, sigma, quantity, hemisphere)

        if name == 'Posterior Putamen / Caudate Nucleus':
            name = 'Posterior\n Putamen/Caudatus'
        elif name == 'Putamen / Caudate Nucleus':
            name = 'Putamen/Caudatus'
        elif name == 'Caudate Nucleus':
            name == 'Caudatus'

        yield [name, hemisphere, round(quantity, 2), round(lln, 2),
               MultiColumn(4, align="l", data=NoEscape(
                   r"\begin{minipage}{6cm}\includegraphics[width=9cm]{%s}\end{minipage}" % plot_path)), ""]


def produce_asymmetry_row(patient_values, name,  normal_stat_values):
    """
    Generates table rows for asymmetry values.

    Parameters:
    -----------
    patient_values : DataFrame
        Dataframe of patient values.
    name : str
        Name of the brain region.
    normal_stat_values : array
        Statistical values for normal references.

    Yields:
    -------
    Asymmetry table row.
    """

    # Extract asymmetry value from patient values
    asymmetry = patient_values[name + ' asymmetry']
    sigma = normal_stat_values[name + ' asymmetry'].loc['std']
    mu = normal_stat_values[name + ' asymmetry'].loc['mean']
    if name == 'Posterior Putamen / Caudate Nucleus':
        name = 'Posterior\n Putamen/Caudatus'
    elif name == 'Putamen / Caudate Nucleus':
        name = 'Putamen/Caudatus'
    elif name == 'Caudate Nucleus':
        name == 'Caudatus'
    # Yield asymmetry values and generate plot
    yield name
    yield MultiColumn(3, align="c", data=round(asymmetry, 2))
    plot_path = normal_reference_SD_plot(mu, sigma, asymmetry, 'asymmetry')
    yield MultiColumn(5, align="l", data=NoEscape(
        r"\begin{minipage}{6cm}\includegraphics[width=9cm]{%s}\end{minipage}" % plot_path))


def first_values(doc, patient_values, normal_stat_values, pt_age, pet_desc, age_range):
    """
    Generates the first table of values including brain region SBR values, asymmetry, etc.

    Parameters:
    -----------
    doc : Document
        LaTeX document to append the table.
    patient_values : DataFrame
        Dataframe of patient values.
    normal_stat_values : array
        Statistical values for normal references.
    pt_age : int
        Age of the patient.
    pet_desc : str
        Description of the PET study.
    age_range : tuple
        Age range for normal references.

    Returns:
    --------
    None
    """

    age_min = np.min(age_range)  # Minimum age in reference population
    age_max = np.max(age_range)  # Maximum age in reference population
    N = len(age_range)  # Number of subjects in the reference population

    # Create the table in LaTeX document
    with doc.create(Tabular(NoEscape(r'p{4cm} c c c c c c c p{2.47cm}'))) as table:

        table.add_hline()
        table.add_row(bold("Location"), bold("Hemisphere"), bold("SBR"), bold("LLN"),
                      MultiColumn(5, align="c", data=bold("Z-score")))

        # Add rows for different brain regions
        for row in produce_row(patient_values, "Putamen", normal_stat_values, pt_age):
            table.add_row(*list(row))
        for row in produce_row(patient_values, "Caudate Nucleus", normal_stat_values, pt_age):
            table.add_row(*list(row))

        table.add_hline()

        # Add asymmetry rows
        table.add_row(bold("Location"), bold(" "), bold("Ratio"), bold(" "),
                      MultiColumn(5, align="c", data=bold("Z-score")))
        for row in produce_row(patient_values, "Putamen / Caudate Nucleus",normal_stat_values, pt_age):
            table.add_row(*list(row))

        table.add_hline()

    now = datetime.datetime.now()
    # Add explanatory footnotes to the LaTeX document
    doc.append(NoEscape(r"\begin{flushleft} {\hspace*{0.3cm}\footnotesize Specific Binding Ratio (SBR) is relative to cerebellar grey matter.\newline}"))
    doc.append(NoEscape(r" {\hspace*{0.3cm}\footnotesize LLN: lower limit of normal at -2SD.\newline}"))
    doc.append(NoEscape(r"\hspace*{0.3cm}\footnotesize Z-score: age corrected standard deviations "
                        f"from mean in healthy reference population (n = {N}, Age [{age_min} .. {age_max} years])."))
    doc.append(NoEscape(r"{\newline}"))
    doc.append(NoEscape(r" {\hspace*{0.3cm}\footnotesize PET image shown: %s.}" % (pet_desc)))
    doc.append(NoEscape(r"{\newline}"))
    doc.append(NoEscape(r" {\hspace*{0.3cm}\footnotesize Report version: 2.0. Revision number: 223. Time of computation: %s }\end{flushleft}" % (now.strftime('%d-%m-%Y %H:%M'))))


def second_values(doc,patient_values, normal_stat_values, pt_age):
    """
    Creates and populates a table in a LaTeX document that presents patient SBR values, LLN,
    and Z-scores for different brain regions (Striatum, Posterior Putamen, etc.),
    as well as asymmetry calculations.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object.
    patient_values : DataFrame
        DataFrame with patient-specific values.
    normal_stat_values : DataFrame
        DataFrame with normal reference values (mean, slope, sigma, etc.).
    pt_age : int
        The age of the patient.

    Returns:
    --------
    None
    """
    # Create the table structure with specified column widths and alignments
    with doc.create(Tabular(NoEscape(r'p{4cm} c c c c c c c p{2.47cm}'))) as table:

        table.add_hline()
        table.add_row(bold("Location"), bold("Hemisphere"), bold("SBR"), bold("LLN"),
                      MultiColumn(5, align="c", data=bold("Z-score")))

        # Add rows for different brain regions
        for row in produce_row(patient_values, "Striatum", normal_stat_values, pt_age):
            table.add_row(*list(row))
        for row in produce_row(patient_values, "Posterior Putamen", normal_stat_values, pt_age):
            table.add_row(*list(row))

        table.add_hline()
        # Add row with names of columns
        table.add_row(bold("Location"), bold(" "), bold("Ratio"), bold(" "),
                      MultiColumn(5, align="c", data=bold("Z-score")))

        # Add row with values for each column
        for row in produce_row(patient_values, "Posterior Putamen / Caudate Nucleus", normal_stat_values, pt_age):
            table.add_row(*list(row))

        table.add_hline()

        # Add row with names of columns
        table.add_row(
            bold("Location"), MultiColumn(3, align="c", data=bold("Hemisphere")),
            MultiColumn(5, align="c", data=bold("Z-score")))

        # Add row with values for each column
        table.add_row(
            "", MultiColumn(3, align="c", data=bold("asymmetry")),
            MultiColumn(5, align="c", data=""))

        # Populate rows for asymmetry calculations for different regions
        for region in ["Caudate Nucleus", "Putamen", "Striatum", "Posterior Putamen",
                       "Putamen / Caudate Nucleus", "Posterior Putamen / Caudate Nucleus"]:
            table.add_row(*list(produce_asymmetry_row(patient_values, region, normal_stat_values)))

        table.add_hline()

        doc.append(NoEscape(r"\begin{flushleft}  {\ \hspace*{0.3cm}\footnotesize \mbox{Hemisphere asymmetry $=$ (R$-$L)$/$(R$+$L).}}\end{flushleft}"))


def get_model(normal_values, name, hemisphere='both'):
    """
    Creates a linear regression model to fit SBR values against age for normal reference values.

    Parameters:
    -----------
    normal_values : DataFrame
        DataFrame of normal values.
    name : str
        Name of the brain region (e.g., Putamen, Caudate Nucleus).
    hemisphere : str
        Specify 'both' for average of right and left hemispheres, or 'right'/'left' for a specific hemisphere.

    Returns:
    --------
    model : LinearRegression
        The linear regression model fitted to the data.
    age_obs : array
        The observed ages in the data.
    sbr_obs : array
        The observed SBR values in the data.
    sbr_fit : array
        The SBR values predicted by the model for the observed ages.
    """

    # Get SBR values for both hemispheres or one hemisphere
    if hemisphere == 'both':
        sbr_right = normal_values[[name + ' right']].values
        sbr_left = normal_values[[name + ' left']].values
        sbr_obs = (sbr_right + sbr_left)/2 # mean of two hemisphere
    else:
        sbr_obs = normal_values[[name + hemisphere]].values

    # Extract age data
    age_obs = (normal_values[['age']].values).astype(int)

    # Fit a linear regression model to the age and SBR values
    model = LinearRegression()
    model.fit(age_obs, sbr_obs)
    sbr_fit = model.predict(age_obs) # Fitted SBR values

    return model, age_obs, sbr_obs, sbr_fit


def plot_normal(normal_values, patient_values, pt_age, name, legend=False):
    """
    Generates a plot comparing patient SBR values with normal reference intervals (mean ± 2 SD).

    Parameters:
    -----------
    normal_values : DataFrame
        DataFrame of normal values for the reference population.
    patient_values : DataFrame
        DataFrame of patient-specific SBR values.
    pt_age : int
        Age of the patient.
    name : str
        Name of the brain region (e.g., Putamen, Caudate Nucleus).
    legend : bool
        Boolean flag to display the legend or not.

    Returns:
    --------
    None
    """
    # Extract patient SBR values for right and left hemispheres
    right = patient_values[name + ' right']
    left = patient_values[name + ' left']

    # Create a linear regression model for normal data
    model, age_obs, sbr_obs, sbr_fit = get_model(normal_values, name, hemisphere='both')

    # Generate age range for plotting
    combine_ages =np.concatenate((age_obs.flatten(), [pt_age]))
    min_age = np.min(combine_ages) - 1 # Set minimum age for plot range
    max_age = np.max(combine_ages) + 1  # Set maximum age for plot range
    age_plot = (np.linspace(min_age, max_age, (max_age-min_age)+1)).reshape(-1, 1)

    # Predict SBR values for the age range using the model
    sbr_pred = model.predict(age_plot)
    # sbr_fit = model.predict(age_obs)

    # Calculate standard deviation and create normal reference intervals
    sigma = np.sqrt(np.sum((sbr_fit - sbr_obs)**2)/(len(sbr_obs)-2))
    lower_predicted = sbr_pred - 2 * sigma
    upper_predicted = sbr_pred + 2 * sigma

    # Create plot and set margins
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    margins = {"left": 0.14, "bottom": 0.125, "right": 1, "top": 0.8}
    fig.subplots_adjust(**margins)

    # Optionally add a legend explaining the plot markers for left and right hemisphere
    if legend:
        legend_elements = [
            Line2D([0], [0], marker='s', color='k', label='Left \& right mean', markerfacecolor='w', markersize=15),
            Line2D([0], [0], marker='<', color='w', label='Left', markerfacecolor="#d9534f", markersize=15),
            Line2D([0], [0], marker='>', color='w', label='Right', markerfacecolor="#d9534f", markersize=15)
        ]
        axes.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.2), framealpha=0.2)

    # Plot the predicted SBR, normal patients SBR and reference intervals
    axes.plot(age_obs, sbr_obs, 's', color ='k', markerfacecolor="w", zorder=0)
    axes.plot(age_plot, sbr_pred, 'k-', alpha=0.5)
    axes.plot(age_plot, lower_predicted, 'k--', alpha=0.5)
    axes.plot(age_plot, upper_predicted, 'k--', alpha=0.5)


    # Scatter the patient's right and left hemisphere SBR values
    axes.scatter(pt_age, left, marker="<", color="#d9534f", zorder=9)
    axes.scatter(pt_age, right, marker=">", color="#d9534f", zorder=10)

    # Set axis labels and formatting
    axes.set_xlabel("Age (years)", fontsize=16)
    axes.set_ylabel({"Putamen": "Putamen SBR", "Putamen / Caudate Nucleus": "Putamen / Caudate Nucleus ratio"}[name], fontsize=16)
    axes.set_xlim(min_age, max_age)
    axes.yaxis.set_major_formatter({"Putamen": FormatStrFormatter('%.1f'), "Putamen / Caudate Nucleus": FormatStrFormatter('%.2f')}[name])

    # Display SD annotations
    label_intervals_mean = model.predict([[min_age]])
    axes.annotate(r"$\textbf{-2SD}$", (min_age, (label_intervals_mean[0] - 2 * sigma) *
                                       {"Putamen": 0.92,"Putamen / Caudate Nucleus": 1.01}[name]), size=13)
    axes.annotate(r"$\textbf{+2SD}$", (min_age, (label_intervals_mean[0] + 2 * sigma) *
                                       {"Putamen": 1.02, "Putamen / Caudate Nucleus": 1.01}[name]), size=13)

    # Hide unnecessary plot spines
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    return fig

def normal_reference_SD_plot(mu, sigma, sbr, measure_type):
    """
    Produces normal reference intervals showing the distance to the mean
    of a reference distribution for a normal population, e.g., mean ± 2*SD.
    Creates a plot highlighting the patient's SBR value against these intervals.

    Parameters:
    -----------
    mu : float
        The mean value for the reference distribution.
    sigma : float
        Standard deviation of the reference distribution.
    sbr : float
        The patient's specific SBR value to be plotted.
    measure_type : str
        Type of measure, such as 'asymmetry', to decide the plot style.

    Returns:
    --------
    None
    """
    # Define the interval limits based on mean (mu) and standard deviation (sigma)
    interval_left = mu - 2 * sigma
    interval_right = mu + 2 * sigma
    extend_left = mu - 10 * sigma

    # For asymmetry, extend to 10 SD above mean; otherwise, extend only to the right interval
    if measure_type == 'asymmetry':
        extend_right  = mu + 10 * sigma
    else:
        extend_right  = interval_right

    # Create the figure and axis for plotting
    fig, axes = plt.subplots(
        1, 1, figsize=(5, 0.67))

    # Generate the data to be plotted. For 'asymmetry', we use a different range for positive values.
    if measure_type == 'asymmetry':
        to_plot = np.where(np.logical_and(
                            interval_right > np.outer(np.arange(extend_left, extend_right, 0.00001), np.ones(1)),
                            np.outer(np.arange(extend_left, extend_right, 0.00001), np.ones(1)) > interval_right),
                           1, 0)
    else:
        to_plot = np.where(np.outer(np.arange(extend_left, extend_right, 0.00001), np.ones(1)) > interval_left, 1, 0)

    # Set up colormap for the plot
    cmap = colors.ListedColormap(['white', 'white'])
    bounds = [-float("inf"), 0.9, float("inf")]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot the background using the created data
    axes.imshow(np.rot90(to_plot), extent=[extend_left, extend_right, extend_left, extend_right],
                interpolation="nearest", origin="lower", cmap=cmap, norm=norm, aspect=0.05)

    # Highlight the area from mean - 10 SD to mean - 2 SD in red
    axes.axvspan(extend_left, interval_left, fill=False, edgecolor="#d9534f", linewidth=2)

    # If it's asymmetry, highlight the region from +2 SD to +10 SD in red as well
    if measure_type == 'asymmetry':
        axes.axvspan(interval_right, extend_right, fill=False, edgecolor="#d9534f", linewidth=2)

    # Highlight the mean ± 2 SD region in green
    axes.axvspan(interval_left, interval_right, fill=False, edgecolor="#5cb85c", linewidth=2)

    # Determine the position of the patient's SBR value in the plot
    if sbr <= extend_left:
        position = extend_left-0.01*np.abs(extend_left)
    elif sbr >= extend_right:
        position = extend_right+0.01*np.abs(extend_left)
    else:
        position = sbr

    # Define tick positions and labels, depending on whether the measure type is 'asymmetry'
    tick_positions = [position] + ([extend_left, extend_right, interval_left, interval_right]
                                   if measure_type == 'asymmetry'
                                   else [extend_left, extend_right, interval_left])

    tick_labels = ([r"\textbf{%s}" % round((sbr - mu) / sigma, 1)] +
                   (["-10 SD (R$<$L)", "10 SD (L$<$R)", "-2 SD", "+2 SD"]
                    if measure_type == 'asymmetry'
                    else ["-10 SD", "+2 SD", "-2 SD"]))

    # Set the x-ticks and labels for the plot
    plt.xticks(tick_positions, tick_labels)
    for label in axes.get_xticklabels():
        label.set_fontsize(13)

    # Remove vertical black lines from tick marks
    axes.tick_params(axis=u'both', which=u'both', length=0)

    # Remove black border lines around the plot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['bottom'].set_visible(False)

    # Hide the y-axis
    axes.get_yaxis().set_visible(False)

    # Highlight the patient's SBR value with a green or red vertical line depending on its position
    if measure_type == 'asymmetry':
        plt.axvline(x=np.min([np.max([sbr, extend_left]), extend_right]), linewidth=5,
                    color="#2a623d" if interval_left < sbr < interval_right else "#bf0000")

        for label in islice(axes.get_xticklabels(), 0, 1):
            label.set_position((0, 2.25))
            label.set_color("#2a623d" if interval_left < sbr < interval_right else "#bf0000")
    else:
        plt.axvline(x=min(max(sbr, extend_left), extend_right), linewidth=5,
                    color="#2a623d" if interval_left < sbr else "#bf0000")

        for label in islice(axes.get_xticklabels(), 0, 1):
            label.set_position((0, 2.25))
            label.set_color("#2a623d" if interval_left < sbr else "#bf0000")

    plot_path = Figure()._save_plot()
    plt.close(fig)
    return plot_path


def plots_normal_values(doc, normal_values, patient_values, pt_age):
    """
    Generates a LaTeX figure with two subplots comparing the patient's Striatal Binding Ratio (SBR) values
    to normal reference intervals for the Putamen region and the Putamen/Caudate Nucleus ratio.

    This function creates visual plots and inserts them into the given LaTeX document (`doc`) using the `pylatex` library.
    It visually represents how the patient’s SBR values compare to reference values, highlighting deviations from the norm.

    Parameters:
    -----------
    doc : pylatex.Document
        The LaTeX document object to which the plots will be added.

    normal_values : pandas.DataFrame
        DataFrame containing the normal reference intervals for SBR values, stratified by age and region.

    patient_values : pandas.DataFrame
        DataFrame containing the patient’s specific SBR values for different regions of the brain.

    pt_age : int
        The age of the patient, used to select the appropriate reference interval from `normal_values`.

    Returns:
    --------
    None
        This function modifies the `doc` in place by adding a LaTeX figure with two subplots.

    Notes:
    ------
    - The first subplot compares the patient’s SBR values with the normal reference range for the Putamen.
    - The second subplot compares the Putamen/Caudate Nucleus ratio with the reference range.
    - The plots are generated using a helper function `plot_normal()`, which handles the actual plotting.
    - The `SubFigure` objects are laid out side by side with a defined width and spacing for consistent alignment.

    Example:
    --------
    >>> plots_normal_values(doc, normal_df, patient_df, pt_age=65)
    >>> doc.generate_pdf("report", clean_tex=False)
    """

    subfig_width = r'8cm'

    with doc.create(Figure(position='!htp')) as plot:
        doc.append(Command('centering'))

        with doc.create(SubFigure(width=NoEscape(subfig_width))) as subplot1:
            doc.append(Command('centering'))

            # Generate the plot for the Putamen region
            fig1 = plot_normal(normal_values, patient_values, pt_age, "Putamen", legend=True)

            subplot1.add_plot(width=subfig_width)
            plt.close(fig1)
            
        doc.append(NoEscape(r'\hspace{0.7cm}'))

        with doc.create(SubFigure(width=NoEscape(subfig_width))) as subplot2:
            doc.append(Command('centering'))

            # Generate the plot for the Putamen/Caudate Nucleus ratio
            fig2 = plot_normal(normal_values, patient_values, pt_age, "Putamen / Caudate Nucleus")

            subplot2.add_plot(width=subfig_width)
            plt.close(fig2)

def get_age_from_dataset(ref_pet_dcm):

    if hasattr(ref_pet_dcm, "PatientAge"):
        try:
            pt_age = get_age(ref_pet_dcm.PatientAge)
        except Exception:
            pt_age = get_age_from_birth(ref_pet_dcm.PatientBirthDate)
    else:
        pt_age = get_age_from_birth(ref_pet_dcm.PatientBirthDate)
    return pt_age

def get_age_from_birth_easy(patient_birth):

    year_now = datetime.date.today().year
    year = int(patient_birth[:4])
    return year_now-year

def get_age_from_birth(patient_birth):

    today = datetime.date.today()
    birth_date = datetime.datetime.strptime(patient_birth, "%Y%m%d").date()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def generate_report(self, ref_pet_dcm, anatomical_desc, normalised_pet, anatomical_path, prediction, cerebellum, patient_values, MR=False):
    """
    Generates a comprehensive clinical report for a PET/CT or PET/MR(DeepDixon protocol) scan using LaTeX-based PDF.


    The report includes sections on dopamine transporter PET imaging, statistical analysis, and a detailed comparison
    of the patient’s data with reference populations. It also plots the relevant PET and anatomical scan slices for visual inspection.

    Parameters:
    -----------
    ref_pet_dcm : pydicom.FileDataset
        Reference DICOM metadata for the PET scan, used to extract patient information and scan details.

    anatomical_desc : str
        Description of the anatomical scan, typically indicating the anatomical region or scanning protocol.

    normalised_pet : numpy.ndarray
        Normalized PET scan data array representing tracer uptake.

    anatomical_path : str
        File path to the anatomical scan in NIfTI format (.nii), used to load and process anatomical image data.

    prediction : numpy.ndarray
        Array representing the predicted regions of interest (ROI) from model-based analysis.

    cerebellum : numpy.ndarray
        Array representing the cerebellum region for comparison with predicted regions.

    patient_values : pandas.DataFrame
        DataFrame containing patient-specific clinical values and measurements.

    MR : bool, optional (default=False)
        Specifies whether the report should use MRI-based data for cerebrum analysis.
        If `True`, the report will label the section as "Synthetic CT Cerebrum"; otherwise, it will be "CT Cerebrum".

    Returns:
    --------
    str
        File path to the generated PDF report.

    Notes:
    ------
    - The function processes and flips the input PET, prediction, cerebellum, and CT scan data to ensure correct orientation.
    - It dynamically selects statistical reference values based on the institution extracted from the PET DICOM metadata.
    - Visual plots are created for PET scan slices, basal ganglia SBR values, and statistical comparisons with a reference population.
    - The report is formatted using the `pylatex` library, with LaTeX commands for customization.

    Example:
    --------
    >>> report_path = generate_report(ref_pet_dcm, "CT Head", pet_data, "ct_scan.nii", pred_data, cereb_data, patient_df, MR=False)

    """
    self.logger.info('Generating report')
    # Flip PET, prediction, and cerebellum arrays along the axis 0 (typically to adjust orientation)
    pet_flipped = np.flip(normalised_pet, axis=0)
    pet_flipped = np.nan_to_num(pet_flipped)
    prediction_flipped = np.flip(prediction, axis=0)
    cerebellum_flipped= np.flip(cerebellum, axis=0)
    anatomical = nib.load(anatomical_path).get_fdata()
    ct_flipped = np.flip(anatomical, axis=0)
    ct_flipped = np.nan_to_num(ct_flipped)

    # Create mask by summing flipped prediction and cerebellum arrays
    mask = prediction_flipped + cerebellum_flipped

    # Extract metadata from the reference PET DICOM
    pet_desc = ref_pet_dcm.SeriesDescription
    if '_' in pet_desc:
        pet_desc = pet_desc.replace('_', r'\_')
    if '_' in anatomical_desc:
        anatomical_desc = anatomical_desc.replace('_', r'\_')
    institution = ref_pet_dcm.InstitutionName
    if institution == 'Nuklearmedicin':
        institution = 'Rigshospitalet'
    elif institution in ['OUH', 'Region Syd']:
        institution = 'Bispebjerg'

    # Load normal values and statistical data
    normal_values = pd.read_csv(os.path.join(STATIC_FILES, 'normal_values-rig+aff.csv'), index_col=0)

    # Select statistical data based on the institution
    if 'Bispebjerg' in institution:
        normal_stat_values = pd.read_csv(os.path.join(STATIC_FILES, 'stats_BBH_rig+aff.csv'), index_col=0)
        normal_values = normal_values[normal_values['institution'] == 'BBH']
    else:
        normal_stat_values = pd.read_csv(os.path.join(STATIC_FILES, 'stats_RH_rig+aff.csv'), index_col=0)
        normal_values = normal_values[normal_values['institution'] != 'BBH']

    # Extract age range and patient age
    age_range = (normal_values[['age']].values).astype(int)
    pt_age = get_age_from_dataset(ref_pet_dcm)

    # Generate slices from prediction and cerebellum data
    slices = get_slices(prediction_flipped, indices = [2,3], num_slices = 9)[2:8] + get_slices(cerebellum_flipped, indices = [CEREBELLUM_INDEX], num_slices = 4)[1:]
    slices.sort(reverse=True)

    # Define the path for output LaTeX document
    header_doc = os.getcwd() + '/doc'
    doc = create_document(header_doc)
    doc.preamble.append(Command('usepackage', 'xcolor'))
    # Define the custom color
    doc.append(NoEscape(r'\definecolor{babyblue}{RGB}{183, 227, 249}'))

    # Add report header with institution name
    get_report_header(doc, institution)
    # Add a small, clean vertical space to separate the header from the blue box below.
    doc.append(NoEscape(r'\vspace{4pt}'))
    doc.append(NoEscape(r'\vspace{-1.2cm}'))
    doc.append(NoEscape(r'\noindent\colorbox{babyblue}{\begin{minipage}{\linewidth}'))
    doc.append(NoEscape(r'\vspace{-0.5cm}'))
    # Create a section with the flushleft alignment for the title
    with doc.create(Section(NoEscape(r'\begin{flushleft}{Dopamine transporter (DAT) {[\textsuperscript{18}F]}FE-PE2I PET scanning}\end{flushleft}'), numbering=False)):
        # Adjust the vertical space
        doc.append(NoEscape(r'\vspace{-0.5cm}'))

        # Add the flushleft report number with custom blue color
        doc.append(NoEscape(r"\begin{flushleft}  {{Report v. 2.0 (10.2024)}}\end{flushleft}\end{minipage}}"))
        doc.append(NoEscape(r'\newline'))
        doc.append(NoEscape(r'\newline'))
        doc.append(NoEscape(r'\newline'))
        doc.append(NoEscape(r'\newline'))
        doc.append(NoEscape(r'\newline'))
        # Add patient information table
        get_patient_table(doc, ref_pet_dcm, pt_age)

        # Generate and add plots for PET scan data
        get_first_plots(doc, pet_flipped, mask, slices)

        # Add values from first set of analyses
        first_values(doc, patient_values, normal_stat_values, pt_age, pet_desc, age_range)

    # Add new page and report header
    doc.append(NoEscape(r'\newpage'))
    get_report_header(doc, institution)
    doc.append(NoEscape(r'\vspace{5cm}'))
    # Add second page with set of values
    second_values(doc, patient_values, normal_stat_values, pt_age)

    # Add new page and report header
    doc.append(NoEscape(r'\newpage'))
    get_report_header(doc, institution)
    doc.append(NoEscape(r'\vspace{-0.5cm}'))

    # Create section for average basal ganglia SBR
    with doc.create(Section('Average basal ganglia SBR', numbering=False)):
        doc.append(NoEscape(r'\vspace{-0.2cm}'))
        plot_average(doc, pet_flipped, mask, slices)
    doc.append(NoEscape(r'\vspace{-0.2cm}'))

    # Create section for subject relative to mean of reference population
    with doc.create(Section('Subject relative to mean of reference population', numbering=False)):
        doc.append(NoEscape(r'\vspace{-0.2cm}'))
        plots_normal_values(doc, normal_values, patient_values, pt_age)

    # Add new page and report header
    doc.append(NoEscape(r'\newpage'))
    get_report_header(doc, institution)

    # Create section for PET scanning
    with doc.create(Section(NoEscape(r'{[\textsuperscript{18}F]}FE-PE2I PET scanning'), numbering=False)):
        doc.append(NoEscape(r'\vspace{-0.2cm}'))
        plot_nine_pet(doc, pet_flipped, mask, pet_desc, slices)

    # Add new page and report header
    doc.append(NoEscape(r'\newpage'))
    get_report_header(doc, institution)

    # Choose section name based on input anatomical modality
    if MR:
        page_title ='Synthetic CT Cerebrum'

    else:
        page_title = 'CT Cerebrum'

    # Create section for CT Cerebrum
    with doc.create(Section(page_title, numbering=False)):
        doc.append(NoEscape(r'\vspace{-0.2cm}'))
        doc.append(Command('centering'))
        plot_ct(doc, ct_flipped, mask, anatomical_desc, slices, MR)


    # Close all plots
    plt.close('all')

    # Generate LaTeX document and PDF
    doc.generate_tex()
    try:
        doc.generate_pdf(header_doc)
    except Exception:
        shutil.copy(header_doc + ".tex", "/tmp/doc.tex")

    return str(header_doc)+'.pdf'


def normalize(logger,brain_path, pet_path):
    """
    Normalizes the PET and anatomical brain images.
    PET is normalized by subtracting the mean value within a mask, and anatomical is thresholded and normalized.

    Function that normalizes pet with mean value from (mask) NORMALIZATION MASK - which is a big region surrounding and containing putamens and caudate nucleui
    it also normalizes anatomical by taking mean value form thresholded anatomical between 30-50 HU

    Parameters:
    -----------
    brain_path : str
        File path to the anatomical brain NIfTI image.
    pet_path : str
        File path to the PET NIfTI image.
    logger : Logger object
        Logger for logging the process information.

    Returns:
    --------
    np.array
        Array containing the normalized PET and anatomical data.

    Raises:
    -------
    FileNotFoundError
        If any of the input files do not exist.
    """

    # Check if input files exist
    if not Path(brain_path).is_file():
        logger.error(f"anatomical brain file not found: {brain_path}")
        raise FileNotFoundError(f"anatomical brain file not found: {brain_path}")
    if not Path(pet_path).is_file():
        logger.error(f"PET file not found: {pet_path}")
        raise FileNotFoundError(f"PET file not found: {pet_path}")

    # Load anatomical and PET images
    brain = nib.load(brain_path).get_fdata()
    pet = nib.load(pet_path).get_fdata()

    # Load the normalization mask
    mask_path = STATIC_FILES / 'truth_merge_mask_dil10.nii.gz'
    mask = nib.load(mask_path).get_fdata()

    # Normalize PET by subtracting the mean value within the mask
    pet_norm_mean_mask = np.mean(pet[mask != 0])
    normalized_pet = pet - pet_norm_mean_mask

    # Normalize anatomical by thresholding between 30-50 HU and subtracting the mean value
    filtered_data = brain[(brain != 0) & (brain > 30) & (brain < 50)]
    brain_norm_mean_mask = np.mean(filtered_data)
    normalized_brain = brain - brain_norm_mean_mask

    return  np.asarray([normalized_pet, normalized_brain])


def reg_aladin(ref_file, flo_file, aff_file, rig_only_flag=False, aff_direct_flag=False, res_file=None, verbosity=None):
    """
    Perform symmetric global registration using the Block Matching algorithm.

    This function aligns a floating/source image ('flo_file') to a reference/target image ('ref_file')
    by applying a Block Matching algorithm. The result is an affine transformation matrix ('aff_file')
    that describes the alignment between the two images, and the transformed floating image is saved as
    a new file ('res_file').

    Parameters:
    -----------
    ref_file : pathlike object or str
        The input reference/target image to which the floating image will be aligned.

    flo_file : pathlike object or str
        The input floating/source image that needs to be aligned with the reference image.

    aff_file : pathlike object or str
        The output affine matrix file that contains the transformation parameters used to align the floating image.

    res_file : pathlike object or str
        The output file where the affine transformed floating image will be saved.

    verbosity : {'file', 'file_split', 'file_stdout', 'file_stderr', 'stream', 'allatonce', 'none'}, optional
        The level of verbosity for logging output during the registration process. If set to 'none', no
        logging output will be shown.

    Returns:
    --------
    runtime object
        A runtime object representing the execution of the registration process. This object allows access
        to detailed logs and error messages. For example, if verbosity is set to 'file_stdout', you can access
        the standard output of the process with 'result.runtime.stdout'.
    """

    ral = RegAladin()
    ral.inputs.ref_file = ref_file
    ral.inputs.flo_file = flo_file
    if res_file:
        ral.inputs.res_file = res_file
    ral.inputs.aff_file = aff_file
    ral.inputs.rig_only_flag=rig_only_flag
    ral.inputs.aff_direct_flag = aff_direct_flag
    if verbosity is not None:
        if verbosity not in ('file', 'file_split', 'file_stdout',
                             'file_stderr', 'stream', 'allatonce', 'none'):
            raise ValueError('Verbosity of a nipype function must be one of '
                             'the specified.')
        ral.terminal_output = verbosity
    return ral.run()


def reg_resample(ref_file, flo_file, trans_file, out_file, interpol='NN',
                 pad_val=None, verbosity=None):
    """
    Resample a NIfTI file to a reference template using a given transformation matrix.

    This function performs image resampling of the floating/source image ('flo_file') to match the reference/target image
    ('ref_file'), applying the transformation matrix ('trans_file') for the resampling. The output image is saved to
    the specified output file ('out_file').

    Parameters:
    -----------
    ref_file : pathlike object or str
        The input reference/target image to which the floating image will be resampled.

    flo_file : pathlike object or str
        The input floating/source image that needs to be resampled to match the reference image.

    trans_file : pathlike object or str
        The input transformation matrix file that defines the transformation to apply to the floating image.

    out_file : pathlike object or str
        The output filename where the transformed image will be saved.

    interpol : {'NN', 'LIN', 'CUB', 'SINC'}, optional, default='NN'
        The interpolation method to use for resampling. Options include:
        - 'NN': Nearest-neighbor interpolation
        - 'LIN': Linear interpolation
        - 'CUB': Cubic interpolation
        - 'SINC': Sinc interpolation

    pad_val : float, optional, default=None
        The padding value to apply when resampling. If None, no padding will be applied.

    verbosity : {'file', 'file_split', 'file_stdout', 'file_stderr', 'stream', 'allatonce', 'none'}, optional
        The level of verbosity for logging output during the resampling process. If set to 'none', no logging output will be shown.

    Returns:
    --------
    runtime object
        A runtime object representing the execution of the resampling process. This object allows access to
        detailed logs and error messages. For example, if verbosity is set to 'file_stdout', you can access
        the standard output of the process with 'result.runtime.stdout'.
    """

    rsl = RegResample()
    rsl.inputs.ref_file = ref_file
    rsl.inputs.flo_file = flo_file
    rsl.inputs.trans_file = trans_file
    rsl.inputs.inter_val = interpol
    if pad_val is not None:
        rsl.inputs.pad_val = pad_val
    rsl.inputs.out_file = out_file

    if verbosity is not None:
        if verbosity not in ('file', 'file_split', 'file_stdout',
                             'file_stderr', 'stream', 'allatonce', 'none'):
            raise ValueError('Verbosity of a nipype function must be one of '
                             'the specified.')
        rsl.terminal_output = verbosity

    return rsl.run()


def load_model(logger, model_file):
    """
    Load a pre-trained Keras model with custom objects, including any custom layers or functions.

    This function loads a Keras model from a file, while ensuring that any custom layers (like 'InstanceNormalization')
    or other custom objects are correctly loaded into the Keras model.

    Parameters:
    -----------
    model_file : str
        Path to the pre-trained model file (.h5 format) that contains the Keras model. The model file should include
        all necessary custom layers or objects that are part of the model architecture.

    Returns:
    --------
    keras.Model
        The loaded Keras model with custom objects. This model can then be used for inference or further training.

    Raises:
    -------
    ValueError
        If there is an issue with loading the model related to the 'InstanceNormalization' layer and the
        'keras-contrib' package is not installed, a ValueError will be raised indicating the missing dependency.
    """
    logger.info('Loading pre-trained model')

    # Define custom objects for loading the model
    custom_objects = {
        'dice_coefficient_loss': dice_coefficient_loss,
        'dice_coefficient': dice_coefficient,
        'dice_coef': tversky_coef,
        'dice_coef_loss': tversky_loss,
        'tversky_loss': tversky_loss,
        'generalized_dice_loss': generalized_dice_loss,
        'weighted_dice_coefficient': weighted_dice_coefficient,
        'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
        'get_label_dice_coefficient_function': get_label_dice_coefficient_function
    }

    # Attempt to include 'InstanceNormalization' in custom objects if keras-contrib is installed
    try:
        custom_objects['InstanceNormalization'] = keras_contrib.layers.InstanceNormalization
    except ImportError:
        pass

    # Try to load the model with the custom objects
    try:
        return keras.models.load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        # Handle the specific case where 'InstanceNormalization' is not available
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + '\n\nPlease install keras-contrib to use InstanceNormalization:\n'
                                          '"pip install git+https://www.github.com/keras-team/keras-contrib.git"')
        else:
            raise error


def patch_wise_prediction(model, data):
    """
    Perform patch-wise prediction of caudate nuclei and putamen using a U-Net model.

    This function divides the input 3D data into smaller patches, runs the prediction on each patch
    using the provided U-Net model, and then reconstructs the predicted values back into the full image.

    Parameters:
    -----------
    model : keras.Model
        The U-Net model used for prediction. The model should be capable of processing 3D patches
        and outputting predicted values for the caudate nuclei and putamen.

    data : np.ndarray
        A 3D numpy array containing the PET and anatomical data. The data should be in a format that
        can be divided into smaller patches for processing by the model.

    Returns:
    --------
    np.ndarray
        A 3D numpy array containing the prediction for the caudate nuclei and putamen,
        reconstructed from the individual patches predicted by the model.
    """
    # Get the shape of the input patches, based on the model's input shape (e.g., 80x48x48)
    patch_shape = np.asarray([int(dim) for dim in model.input.shape[-3:]])  # Model input shape, e.g., (80, 48, 48)

    # List to store predictions for each patch
    predictions = list()

    # Get the indices for the patches to be extracted from the 3D data
    indices = compute_patch_indices()

    # Loop through each patch index to process and predict patch-wise
    for i in range(len(indices)):
        # Extract the patch from the 3D data using the specified patch shape and index
        patch = get_patch_from_3d_data(data, patch_shape=patch_shape, patch_index=indices[i])[np.newaxis]

        # Predict the patch using the model. This returns the predicted values for this patch.
        prediction = model.predict(patch, verbose=0)

        # Append each predicted patch result to the predictions list
        for predicted_patch in prediction:
            predictions.append(predicted_patch)

        # Calculate the output shape based on the model's output and the input data shape
        output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])

    # Reconstruct the full output from the list of patches
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)


def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Extract a patch from a 3D numpy array containing anatomical and PET data.

    This function crops a subregion (patch) from the input 3D array based on the provided shape and corner index.

    Parameters:
    -----------
    data : np.ndarray
        A 3D numpy array containing anatomical and PET data from which the patch will be extracted.

    patch_shape : tuple
        A tuple specifying the dimensions (depth, height, width) of the patch to be extracted.

    patch_index : tuple
        A tuple specifying the starting corner index (z, y, x) of the patch in the input data.

    Returns:
    --------
    np.ndarray
        A numpy array containing the extracted patch of anatomical and PET data with the specified shape.
    """
    return data[...,
                patch_index[0]: patch_index[0] + patch_shape[0],
                patch_index[1]: patch_index[1] + patch_shape[1],
                patch_index[2]: patch_index[2] + patch_shape[2]]


def reconstruct_from_patches(patches, patch_indices, data_shape):
    """
    Reconstruct an array of the original shape from a list of patches and their corresponding indices.

    Overlapping regions in the patches are averaged to produce the reconstructed data.

    Parameters:
    -----------
    patches : list
        A list of numpy arrays representing the prediction patches.

    patch_indices : list
        A list of tuples specifying the corner indices for each patch in the original data.

    data_shape : tuple
        A tuple specifying the shape of the original data from which the patches were extracted
        (e.g., (depth, height, width)).

    Returns:
    --------
    np.ndarray
        A numpy array reconstructed from the input patches, matching the specified original data shape.
    """

    data =np.zeros(data_shape)
    count = np.zeros(data_shape, dtype=int)

    for patch, index in zip(patches, patch_indices):
        patch_index = np.zeros(data_shape, dtype=bool)
        patch_data = np.zeros(data_shape)

        # Set the appropriate part of the patch_index and patch_data
        patch_index[..., index[0]:index[0] + patch.shape[-3],
                    index[1]:index[1] + patch.shape[-2],
                    index[2]:index[2] + patch.shape[-1]] = True
        patch_data[patch_index] = patch.flatten()

        # Update data and count arrays
        data += patch_data
        count[patch_index] += 1

    epsilon = 1e-10
    result = np.where(count == 0, 0, data / np.maximum(count, epsilon))
    return result


def prediction_to_image(prediction, threshold=0.5, labels=None):
    """
    Convert model prediction to a labeled image based on a specified threshold.

    This function assigns labels to the prediction array based on a threshold value, optionally using a provided list of labels.

    Parameters:
    -----------
    prediction : np.ndarray
        A numpy array containing the model's prediction values.

    threshold : float, optional
        The threshold value for binarizing the prediction. Predictions greater than or equal to this value are labeled.
        Defaults to 0.5.

    labels : list, optional
        A list of labels corresponding to each class in the prediction. If not provided, default labels will be used.
        Defaults to None.

    Returns:
    --------
    np.ndarray
        A numpy array representing the labeled image, where each pixel/voxel is assigned to a class based on the threshold.
    """

    if prediction.shape[1] == 1:  # Check if the prediction is for binary classification (single channel)
        data = prediction[0, 0]  # Extract the binary prediction array from the first batch
        label_map_data = np.zeros(prediction[0, 0].shape, np.int8)  # Initialize an empty array for labeled data
        label = labels[0] if labels else 1  # Assign a default label (1) if no labels are provided
        label_map_data[data > threshold] = label  # Label regions where prediction exceeds the threshold
        data = label_map_data  # Update 'data' with the labeled image
    elif prediction.shape[1] > 1:  # Check if the prediction is for multi-class classification
        # Use a helper function to get the labeled image for multi-class predictions
        label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
        data = label_map_data[0]  # Extract the labeled data from the first batch
    else:
        # Raise an error if the prediction array shape is invalid or doesn't match expected formats
        raise RuntimeError('Invalid prediction array shape: {0}'.format(prediction.shape))
    # return data
    # Filter largest clusters for each label
    largest_clusters_label_2 = filter_largest_clusters(data, target_label=2.0, num_clusters=2)
    largest_clusters_label_3 = filter_largest_clusters(data, target_label=3.0, num_clusters=2)
    final_prediction = largest_clusters_label_2 + largest_clusters_label_3
    return final_prediction  # Return the labeled image


def filter_largest_clusters(mask, target_label=2.0, num_clusters=2):
    """
    Filters and retains the largest connected clusters in a given mask for a specified label.

    This function identifies connected components in a binary mask derived from the input `mask`
    and retains only the largest connected clusters corresponding to a specified label.
    It returns a mask containing these largest clusters while preserving the original label value.

    Parameters:
    -----------
    mask : ndarray
        Input 2D or 3D array representing the labeled mask. Elements with `target_label` will be processed.

    target_label : float, optional (default=2.0)
        The label value to filter within the mask. Only regions with this label will be processed.

    num_clusters : int, optional (default=2)
        The number of largest clusters to retain. If the number of clusters found is less than or equal to this value,
        all clusters are retained.

    Returns:
    --------
    largest_clusters_mask : ndarray
        A binary mask (with the same shape as the input mask) that contains only the largest clusters
        with the specified label value. All other regions are set to zero.

    Example:
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> mask = np.array([[0, 2, 2, 0], [0, 2, 0, 0], [3, 3, 2, 2]])
    >>> filtered_mask = filter_largest_clusters(mask, target_label=2.0, num_clusters=1)
    >>> print(filtered_mask)
    [[0. 2. 2. 0.]
     [0. 2. 0. 0.]
     [0. 0. 0. 0.]]
    """
    # Step 1: Isolate the region with the target label
    binary_mask = (mask == target_label).astype(float)

    # Step 2: Label connected components
    labeled_array, num_features = scipy.ndimage.label(binary_mask)

    # Step 3: Count sizes of each component
    sizes = np.bincount(labeled_array.ravel().astype(int))[1:]  # Exclude background count at index 0

    if len(sizes) > num_clusters:
        # Step 4: Find the indices of the largest clusters
        largest_cluster_indices = np.argsort(sizes)[-num_clusters:] + 1  # +1 because labels start at 1

        # Step 5: Create a mask for the largest clusters with the original float label
        largest_clusters_mask = np.isin(labeled_array, largest_cluster_indices).astype(float) * target_label
    else:
        # If there are fewer clusters than required, keep them all with the original label
        largest_clusters_mask = (labeled_array > 0).astype(float) * target_label

    return largest_clusters_mask


def compute_patch_indices():
    """
    Compute the corner indices for extracting patches from 3D data.

    This function generates a set of indices representing the starting coordinates of patches
    to be extracted from the data. These indices correspond to the top-left-front corner of each patch.

    Parameters:
    -----------
    None

    Returns:
    --------
    np.ndarray
        A numpy array where each row contains the (x, y, z) coordinates of a patch's corner.
        The output has the format:
        array([[ 80,  94, 101],
               [ 80,  94, 112],
               [ 80, 114, 101],
               [ 80, 114, 112],
               [ 94,  94, 101],
               [ 94,  94, 112],
               [ 94, 114, 101],
               [ 94, 114, 112]])
    """

    # Define the starting coordinates for the grid
    start = np.array([80, 94, 101])

    # Define the stopping coordinates for the grid
    stop = np.array([95, 115, 113])

    # Define the step size for the grid along each dimension
    step = np.array([14, 20, 11])

    # Generate a 3D grid using np.mgrid
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0],
                               start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int16)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    """
    Convert prediction scores into labeled arrays based on the provided threshold.

    This function assigns labels to each spatial location based on the predicted class with the highest score.
    Locations where the maximum score is below the threshold are labeled as background (0).
    Optionally, custom labels can be assigned to the classes.

    Parameters:
    -----------
    prediction : np.ndarray
        A numpy array of shape (n_samples, n_classes, ...) containing prediction scores.
        - 'n_samples' is the number of samples.
        - 'n_classes' is the number of predicted classes.
        - The remaining dimensions correspond to the spatial dimensions of the predictions.

    threshold : float, optional
        Minimum prediction score required to assign a label. Predictions below this value are set to 0 (background).
        Default is 0.5.

    labels : list, optional
        A list of custom labels corresponding to class indices. If provided, these labels will replace class indices
        in the output. Default is None, which keeps the class indices as labels.

    Returns:
    --------
    list
        A list of labeled arrays, one for each sample. Each array has the same spatial dimensions as the input data,
        with values representing the assigned class or custom label.
    """
    # Number of samples in the prediction array
    n_samples = prediction.shape[0]

    # Initialize an empty list to store labeled arrays
    label_arrays = []

    # Iterate over each sample in the prediction array
    for sample_number in range(n_samples):
        # Assign labels based on the highest prediction score for each spatial location
        label_data = np.argmax(prediction[sample_number], axis=0) + 1

        # Apply the threshold
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0

        # If a list of labels is provided, map class indices to actual labels
        if labels:
            # Iterate over unique non-zero values in label_data
            for value in np.unique(label_data)[1:]:
                # Replace the class index (value) with the corresponding label
                label_data[label_data == value] = labels[value - 1]

        # Append the labeled array for this sample to the output list
        label_arrays.append(label_data.astype(np.uint8))

    return label_arrays


def dice_coefficient_loss(y_true, y_pred):
    """
    Calculate the Dice coefficient loss.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth binary tensor representing the target mask.
    y_pred : tf.Tensor
        Predicted binary tensor from the model.

    Returns:
    --------
    tf.Tensor
        The negative Dice coefficient value to be minimized during training.
    """
    return -dice_coefficient(y_true, y_pred)


def dice_coefficient(y_true, y_pred, smooth=1.):
    """
    Compute the Dice similarity coefficient.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth binary tensor representing the target mask.
    y_pred : tf.Tensor
        Predicted binary tensor from the model.
    smooth : float, optional
        A small constant added to the numerator and denominator to prevent division by zero. Default is 1.0.

    Returns:
    --------
    tf.Tensor
        The Dice coefficient, a scalar value between 0 and 1, indicating the similarity between 'y_true' and 'y_pred'.
    """
    # Flatten the tensors to compute the overlap
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    # Calculate the intersection between the ground truth and predictions
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

    # Compute the Dice coefficient with smoothing
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def tversky_loss(y_true, y_pred):
    """
    Compute the Tversky loss for imbalanced datasets, a generalization of the Dice coefficient.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth tensor (binary mask), where 1 represents the presence of the object of interest.

    y_pred : tf.Tensor
        Predicted tensor (probability map), with values ranging from 0 to 1 representing the model's confidence.

    Returns:
    --------
    tf.Tensor
        A scalar tensor representing the Tversky loss, which quantifies the dissimilarity between 'y_true' and 'y_pred'.
    """
    # Constants for false positives and false negatives weighting
    alpha = 0.3
    beta = 0.7

    # Create tensor of ones with the same shape as y_true for calculating complement
    ones = tf.keras.backend.ones(tf.keras.backend.shape(y_true))

    p0 = y_pred       # Probability that voxels are of the predicted class
    p1 = ones - y_pred  # Probability that voxels are not of the predicted class
    g0 = y_true       # Ground truth
    g1 = ones - y_true # Inverse of ground truth

    # Compute the numerator and denominator for the Tversky index
    numerator = tf.keras.backend.sum(p0 * g0, (0, 1, 2, 3))
    denominator = numerator + alpha * tf.keras.backend.sum(p0 * g1, (0, 1, 2, 3)) + beta * tf.keras.backend.sum(p1 * g0, (0, 1, 2, 3))

    # Tversky loss (1 - Tversky index)
    T = tf.keras.backend.sum(numerator / denominator)

    # Number of classes (for dynamic scaling)
    Ncl = tf.keras.backend.cast(tf.keras.backend.shape(y_true)[-1], 'float32')

    return Ncl - T


def tversky_coef(y_true, y_pred):
    """
    Compute the Tversky coefficient, which is the negative of the Tversky loss.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth tensor (binary mask), where 1 represents the presence of the object of interest.

    y_pred : tf.Tensor
        Predicted tensor (probability map), with values ranging from 0 to 1 representing the model's confidence.

    Returns:
    --------
    tf.Tensor
        A scalar tensor representing the Tversky coefficient, which measures the similarity between 'y_true' and 'y_pred'.
    """
    return -tversky_loss(y_true, y_pred)


def generalized_dice_loss(y_true, y_pred):
    """
    Compute the generalized Dice loss, which accounts for class imbalance by weighting
    each label's contribution inversely proportional to its volume.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth tensor with shape (batch_size, height, width, depth, n_classes).
        The tensor contains the true labels for each class in the segmentation task.

    y_pred : tf.Tensor
        Predicted tensor with shape (batch_size, height, width, depth, n_classes).
        The tensor contains the predicted probabilities or logits for each class.

    Returns:
    --------
    tf.Tensor
        A scalar tensor representing the generalized Dice loss. Lower values indicate better performance.
    """
    Ncl = y_pred.shape[-1] # Number of classes
    w = np.zeros((Ncl,))  # Initialize an array to store the weight for each class

    # Loop through each class to calculate the weight based on the number of true positives for each class
    for l in range(0, Ncl):
        # Sum the number of true positives for class `l` in y_true
        w[l] = np.sum(np.asarray(y_true[:, :, :, :, l] == 1, np.int8))

    # Prevent division by zero by adding a small constant (0.00001) and taking the inverse of the class volumes (squared)
    w = 1 / (w**2 + 0.00001)

    # Compute the numerator of the generalized Dice coefficient:
    numerator = y_true * y_pred
    numerator = w * tf.keras.backend.sum(numerator, (0, 1, 2, 3))
    numerator = tf.keras.backend.sum(numerator)

    # Compute the denominator of the generalized Dice coefficient:
    denominator = y_true + y_pred
    denominator = w * tf.keras.backend.sum(denominator, (0, 1, 2, 3))
    denominator = tf.keras.backend.sum(denominator)

    # Calculate the generalized Dice coefficient
    gen_dice_coef = numerator / denominator

    # Return the generalized Dice loss
    return 1 - 2 * gen_dice_coef


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=1e-5):
    """
    Compute the weighted Dice coefficient for evaluating the similarity
    between the ground truth and predicted masks, with the option for a smoothing constant.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth tensor, typically with shape (batch_size, height, width, depth, n_classes)
        or similar for multi-class segmentation.

    y_pred : tf.Tensor
        Predicted tensor with the same shape as 'y_true', typically containing probability values
        for each class at each voxel location.

    axis : tuple of int, optional
        Axes along which to compute the Dice coefficient. Defaults to (-3, -2, -1) assuming 'channels first'
        data format. Modify for specific dimensionality of the data.

    smooth : float, optional
        Smoothing constant to avoid division by zero. Defaults to 1e-5.

    Returns:
    --------
    tf.Tensor
        A scalar tensor representing the weighted Dice coefficient. Higher values indicate better
        similarity between the predicted and true masks.
    """
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=axis) + smooth / 2

    # Compute the sum of y_true and y_pred, and add smooth
    summation = tf.keras.backend.sum(y_true, axis=axis) + tf.keras.backend.sum(y_pred, axis=axis) + smooth

    # Compute the Dice coefficient
    dice = 2. * intersection / summation

    # Compute the mean Dice coefficient
    return tf.keras.backend.mean(dice)


def get_label_dice_coefficient_function(label_index):
    """
    Create a function to compute the Dice coefficient for a specific label.

    Parameters:
    -----------
    label_index : int
        The index of the label (class) for which the Dice coefficient function is generated.

    Returns:
    --------
    function
        A function that computes the Dice coefficient for the specified label when passed
        the ground truth and predicted tensors.
    """
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))

    return f


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    """
    Compute the Dice coefficient for a specific label.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth tensor with shape (batch_size, n_classes, ...). Each slice along
        the second axis represents a separate class.

    y_pred : tf.Tensor
        Predicted tensor with the same shape as 'y_true', containing predicted
        probabilities for each class.

    label_index : int
        The index of the class (label) for which the Dice coefficient will be computed.

    Returns:
    --------
    tf.Tensor
        A scalar tensor representing the Dice coefficient for the specified label (class).
    """
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def weighted_dice_coefficient_loss(y_true, y_pred):
    """
    Compute the weighted Dice coefficient loss for imbalanced segmentation tasks.

    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth tensor, typically a binary or multi-class mask with shape
        (batch_size, height, width, depth, n_classes).

    y_pred : tf.Tensor
        Predicted tensor with the same shape as 'y_true', containing the predicted
        probabilities or logits for each class.

    Returns:
    --------
    tf.Tensor
        A scalar tensor representing the negative weighted Dice coefficient,
        which is minimized during training.
    """
    return -weighted_dice_coefficient(y_true, y_pred)
