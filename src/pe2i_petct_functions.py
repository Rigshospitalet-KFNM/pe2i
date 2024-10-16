import os
from datetime import datetime
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd
import nibabel as nib
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
from tensorflow import keras
import tensorflow as tf
import keras_contrib
import dotenv
dotenv.load_dotenv()
#tf.config.list_physical_devices('GPU')
# tf.config.list_physical_devices('CPU')

from nilearn.image import smooth_img
from nipype.interfaces.niftyseg import LabelFusion
from nipype.interfaces.niftyreg import RegAladin, RegResample
from pylatex import Figure, Command, NoEscape, Tabular, Document, Package,Section, SubFigure, MultiColumn, MiniPage, FlushLeft
from pylatex.utils import bold, NoEscape
from pylatex.base_classes import Environment

from rhnode import RHJob #pip install git+https://github.com/CAAI/rh-node.git
from HD_CTBET.run import run_hd_ctbet

STATIC_FILES = Path(os.environ.get("STATIC_PATH"))
FWHM = 2.35482 # converting sigma of 1 to FWHM 2.35482*1 mm
CEREBELLUM_INDEX = 4

# Colormap to be used when displaying images.
_PETRainbowCMAP = matplotlib.colors.LinearSegmentedColormap(
    'PET-Rainbow',
    {
        u'blue': [(0.0, 0.1, 0.1),
                  (0.1, 0.6667, 0.6667),
                  (0.15, 0.9667, 0.9667),
                  (0.2, 0.8667, 0.8667),
                  (0.25, 0.8667, 0.8667),
                  (0.3, 0.8667, 0.8667),
                  (0.35, 0.8333, 0.8333),
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
                   (0.1, 0.0, 0.0),
                   (0.15, 0.0, 0.0),
                   (0.20, 0.4667, 0.4667),
                   (0.25, 0.8, 0.8),
                   (0.30, 0.8667, 0.8667),
                   (0.35, 0.8667, 0.8667),
                   (0.45, 0.86, 0.86),
                   (0.5, 0.8633, 0.8633),
                   (0.55, 0.8667, 0.8667),
                   (0.58, 0.8667, 0.9667),
                   (0.63, 0.9667, 1.0),
                   (0.68, 1.0, 0.9333),
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
    
FIRST_DIM = (22, 234)
SECOND_DIM = (4, 216)  
FIRST_DIM_CROPPED = slice(*FIRST_DIM)
SECOND_DIM_CROPPED = slice(*SECOND_DIM)


class ColorBox(Environment):
    def __init__(self, color):
        super().__init__(arguments=NoEscape(r'\colorbox{' + color + r'}'))
        self.content_separator = ''


def swap_dims(self, modality, name):
    """
    Adjust the orientation of a given NIfTI image (modality) from radiological to neurological if needed, 
    and save the modified image to the specified output directory.

    Parameters:
    -----------
    modality : nib.Nifti1Image
        The NIfTI image to be processed.
    name : str
        The name of the file to be saved, which will also be used to determine if the image is a PET scan.

    Returns:
    --------
    modality_nii : str
        The file path to the reoriented and saved NIfTI image.
    """
    # Validate input type
    if not isinstance(modality, nib.Nifti1Image):
        raise ValueError("Input modality must be a Nifti1Image object.")
    
    # Construct the full output path for the new NIfTI image
    modality_nii = self.processing_directory / (name + '_swap.nii.gz')

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
    nib.save(img, modality_nii)

    # Check if the file is saved successfully
    if os.path.exists(modality_nii):
        self.logger.info(f'NIfTI image saved successfully at: {modality_nii}')
    else:
        self.logger.error(f'Failed to save NIfTI image at: {modality_nii}')
        raise IOError(f"Failed to save NIfTI image at {modality_nii}")
    
    return modality_nii


def run_skullstripping(self, input_modality_nii):
    """
    Perform skull stripping on a CT scan using the hd_ctbet method.

    Parameters:
    -----------
    input_modality_nii : str
        The file path to the NIfTI image to be processed.

    Returns:
    --------
    output_filename : str
         The file path to the resulting skull-stripped NIfTI image.
    """
    # Log the beginning of the skull stripping process
    self.logger.info('Running skullstripping')

    # Define the output filename for the skull-stripped image
    output_filename = self.processing_directory /'CT_swap_BET.nii.gz'

    # Call the hd_ctbet function to perform skull stripping
    run_hd_ctbet(str(input_modality_nii), str(output_filename), mode='fast', device='cpu', do_tta =False)

    return output_filename


def process_ct(self, brain_nii):
    """
    Preprocess a CT scan by applying thresholding and smoothing operations before segmentation.

    Parameters:
    -----------
    brain_nii : str
        The file path to the NIfTI image of the brain CT scan.

    Returns:
    --------
    brain_sm_th_nii : str
        The file path to the preprocessed and saved NIfTI image.
    """
    # Generate the output file path for the preprocessed image
    brain_sm_th_nii = self.processing_directory / 'brain_preprocessed.nii.gz'
    self.logger.info('Applying thresholding and smoothing')

    # Load the NIfTI image
    brain_nib = nib.load(brain_nii)

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
    nib.save(brain_sm_th, brain_sm_th_nii)
    self.logger.info(f'saved {brain_sm_th_nii }')

    # Return the path to the preprocessed image
    return brain_sm_th_nii


def cerebellum_mask(self, input_file):
    """
    Generate a cerebellum mask using the LabelFusion tool with the STEPS algorithm.

    Parameters:
    -----------
    input_file : str
        The file path to the input NIfTI image that needs segmentation.

    Returns:
    --------
    str
        The file path to the generated cerebellum mask NIfTI image.

    Exceptions:
    -----------
    Raises an exception if the LabelFusion process fails.
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
    lf.inputs.out_file = self.processing_directory / 'cerebellum.nii.gz'
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


def resampling(self, pet_nii, ct_nii, brain_nii):
    """
    Resample and register PET and CT scans to a brain template, ensuring that all steps 
    are performed only if the corresponding output files do not already exist.

    This function performs the following operations:
    1. Registers the CT brain image to an average brain template.
    2. Resamples the CT brain and CT images to match the brain template.
    3. Registers and resamples the PET image to the CT image and brain template.

    Parameters:
    -----------
    pet_nii : nib.Nifti1Image
        The NIfTI image representing the PET scan.
    ct_nii : nib.Nifti1Image
        The NIfTI image representing the CT scan.
    brain_nii : nib.Nifti1Image
        The NIfTI image representing the brain scan.

    Returns:
    --------
    petrsltemplate_nii : str
        File path to the PET image resampled to the brain template.
    ctrsl_nii : str
        File path to the CT image resampled to the brain template.
    brainrsl_nii : str
        File path to the brain image resampled to the brain template.
    """

    # Define file paths for templates and output files
    template_nii = STATIC_FILES / 'avg_template_swap.nii.gz'
    brainreg_nii = self.processing_directory / 'brain_reg_avg.nii.gz'
    brainrsl_nii = self.processing_directory / 'brain_rsl_avg.nii.gz'
    trans_ct = self.processing_directory / 'brain_to_avg.txt'
    ctrsl_nii = self.processing_directory / 'ct_rsl_avg.nii.gz'
    petrsl_nii = self.processing_directory / 'pet_reg_ct.nii.gz'
    petreg_nii = self.processing_directory / 'pet_rsl_ct.nii.gz'
    petrsltemplate_nii = self.processing_directory / 'pet_rsl_avg.nii.gz'
    trans_pet = self.processing_directory / 'pet_to_ct-new.txt'

    # Step 1: Register CT brain to the average template if the transformation doesn't exist
    self.logger.info(f'Registering CT brain to template')
    reg_aladin(ref_file=template_nii, 
                flo_file=brain_nii,
                aff_file=trans_ct,
                in_aff_file=None,
                res_file=brainreg_nii,
                verbosity='none')
    
    # Verify if registration was successful
    if not Path(trans_ct).is_file():
        self.logger.error(f"Failed to save CT brain registration at {trans_ct}")
        raise IOError(f"CT brain registration not saved: {trans_ct}")

    # Step 2: Resample CT brain to the template if not already resampled  
    self.logger.info(f'Resampling CT brain to template')
    reg_resample(ref_file=template_nii, 
                    flo_file=brain_nii,
                    trans_file=trans_ct,
                    out_file=brainrsl_nii,
                    interpol='LIN',
                    pad_val=-1024,
                    verbosity='none')
    
    # Check if brain resampling was successful
    if not Path(brainrsl_nii).is_file():
        self.logger.error(f"Failed to save resampled CT brain at {brainrsl_nii}")
        raise IOError(f"Resampled CT brain not saved: {brainrsl_nii}")
        
    # Step 3: Resample CT to the template if not already resampled
    self.logger.info('Resampling CT to template')
    reg_resample(ref_file=template_nii,
                    flo_file=ct_nii,
                    trans_file=trans_ct,
                    out_file=ctrsl_nii,
                    interpol='LIN',
                    pad_val=-1024,
                    verbosity='none')
    
    # Check if CT resampling was successful
    if not Path(ctrsl_nii).is_file():
        self.logger.error(f"Failed to save resampled CT at {ctrsl_nii}")
        raise IOError(f"Resampled CT not saved: {ctrsl_nii}")

    # Step 4: Register PET to CT if the transformation doesn't exist
    self.logger.info('Registering PET to CT')
    reg_aladin(ref_file=ct_nii, 
                flo_file=pet_nii,
                aff_file=trans_pet,
                in_aff_file=None,
                res_file=petreg_nii,
                verbosity='none')
    
    # Verify if PET registration was successful
    if not Path(trans_pet).is_file():
        self.logger.error(f"Failed to save PET registration at {trans_pet}")
        raise IOError(f"PET registration not saved: {trans_pet}")

    # Step 5: Resample PET to CT if not already resampled
    self.logger.info('Resampling PET to CT')
    reg_resample(ref_file=ct_nii,
                    flo_file=pet_nii,
                    trans_file=trans_pet,
                    out_file=petrsl_nii,
                    interpol='LIN',
                    pad_val=0,
                    verbosity='none')

    # Check if PET resampling to CT was successful
    if not Path(petrsl_nii).is_file():
        self.logger.error(f"Failed to save resampled PET at {petrsl_nii}")
        raise IOError(f"Resampled PET not saved: {petrsl_nii}")

    # Step 6: Resample PET to the brain template if not already resampled
    self.logger.info('Resampling PET to template')
    reg_resample(ref_file=template_nii,
                    flo_file=petrsl_nii,
                    trans_file=trans_ct,
                    out_file=petrsltemplate_nii,
                    interpol='LIN',
                    pad_val=0,
                    verbosity='none')
    
    # Check if PET resampling to template was successful
    if not Path(petrsltemplate_nii).is_file():
        self.logger.error(f"Failed to save resampled PET at {petrsltemplate_nii}")
        raise IOError(f"Resampled PET not saved: {petrsltemplate_nii}")

    return petrsltemplate_nii, ctrsl_nii, brainrsl_nii


def get_predition(logger, ct_brain_nii, pet_nii):
    """
    Obtains the prediction for brain segmentation using a trained model.

    Parameters:
    -----------
    logger : Logger object
        Logger for logging the process information.
    ct_brain_nii : str
        File path to the CT brain NIfTI image.
    pet_nii : str
        File path to the PET NIfTI image.

    Returns:
    --------
    prediction_image : numpy array
        The segmentation prediction image of basal ganglia.
    """

    logger.info('Getting predition.')
    
    # Normalize CT and PET images
    input_files = normalize(logger, ct_brain_nii, pet_nii) 
    
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


def get_statistics(logger, pet_nii, cerebellum_nii, prediction):
    """
    Calculates various statistics on PET and prediction data, including median normalization, SBR, 
    asymmetry calculations, and putamen/caudate ratios.

    Parameters:
    -----------
    logger : Logger object
        The logger used for logging information.
    pet_nii : str
        File path to the PET NIfTI image.
    cerebellum_nii : str
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
        Dictionary containing various statistics such as SBR, asymmetries, and ratios.
    """
    # Step 1: Load and normalize PET data by cerebellum cortex median
    logger.info('Calculating cerebellum cortex median') 
    pet_data = nib.load(pet_nii).get_fdata()
    pet_data = np.nan_to_num(pet_data, nan=0.0)
    cerebellum_mask = nib.load(cerebellum_nii).get_fdata()

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

    # Step 3: Get posterior putamen masks
    posterior_putamen_mask_left = get_posterior_putamen(prediction, 'left')
    posterior_putamen_mask_right = get_posterior_putamen(prediction, 'right')  

    # Step 4: Calculate SBR (Specific Binding Ratio) for each region
    putamen_left = get_sbr(pet_left[prediction_left == 2], cerebellum_median)
    putamen_right = get_sbr(pet_right[prediction_right == 2], cerebellum_median)
    caudate_left = get_sbr(pet_left[prediction_left == 3], cerebellum_median)
    caudate_right = get_sbr(pet_right[prediction_right == 3], cerebellum_median)
    striatum_left = get_sbr(pet_left[prediction_left != 0 ], cerebellum_median) 
    striatum_right = get_sbr(pet_right[prediction_right != 0 ], cerebellum_median) 
    posterior_putamen_left = get_sbr(pet_data[posterior_putamen_mask_left == 2], cerebellum_median)
    posterior_putamen_right = get_sbr(pet_data[posterior_putamen_mask_right == 2], cerebellum_median)

    # Step 5: Calculate ratios of putamen to caudate nucleus
    ratio_left = putamen_left / caudate_left
    ratio_right = putamen_right / caudate_right
    ratio_posterior_left = posterior_putamen_left / caudate_left
    ratio_posterior_right = posterior_putamen_right / caudate_right

    # Step 6: Calculate asymmetry indices
    caudate_asymmetry = get_asymmetry(caudate_right, caudate_left)
    putamen_asymmetry = get_asymmetry(putamen_right, putamen_left)
    posterior_asymmetry = get_asymmetry(posterior_putamen_right, posterior_putamen_left)
    striatum_asymmetry = get_asymmetry(striatum_right, striatum_left)
    ratio_asymmetry = get_asymmetry(ratio_right, ratio_left)
    ratio_posterior_asymmetry = get_asymmetry(ratio_posterior_right, ratio_posterior_left)
    
    # Step 7: Compile statistics into a dictionary
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

    return pet_normalized, cerebellum_mask, data


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
    """
    if direction == 'left':
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


def get_posterior_putamen(prediction_data, direction):
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
    putamen_posterior = np.zeros_like(prediction_data)
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


def get_logo(institution):
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
    surname = name_parts[0]
    first_name = name_parts[1]

    # Return the name in "FirstName Surname" format
    return first_name + ' ' + surname


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

    # Add the logo image to the header
    doc.append(NoEscape(r'\begin{minipage}{0.6\textwidth}'))  # Adjust width as needed
    doc.append(NoEscape(r'\includegraphics[height=1cm]{%s}' % icon_path))
    doc.append(NoEscape(r'\end{minipage}'))

    # Add the footnote about the institution
    doc.append(NoEscape(r'\begin{minipage}{0.5\textwidth}'))  # Adjust width as needed
    doc.append(NoEscape(r'{\footnotesize \begin{tabular}{r}' +
                           get_footnote(institution) +
                           r'\end{tabular}}'))
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
    return doc


def get_patient_table(doc, ref):
    """
    Adds a table with patient information to the LaTeX document.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object to which the table will be added.
    ref : Dataset
        The DICOM dataset from which patient information will be extracted.
    """
    
    # Create the table data including headers and patient information
    table_data = [
        ['Patient name', 'CPR', 'Age', 'Sex', 'Scan date', 'Weight [kg]', 'Dose [MBq]'],
        [get_name(ref.PatientName), ref.PatientID, get_age(ref.PatientAge), ref.PatientSex, get_date(ref.StudyDate), int(ref.PatientWeight), int(ref.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose/1e6)],
    ]

    # Center the table content
    doc.append(NoEscape(r'\centering'))

    # Create the table with column alignment
    with doc.create(Tabular('lccccccc')) as table:
        table.append(NoEscape(r'\hline'))

        # Add header row with bold text
        table.append(NoEscape(' & '.join([r'\textbf{' + element + '}' for element in table_data[0]]) + r' \\ '))

         # Add data rows
        for row in table_data[1:]:
            table.append(NoEscape(' & '.join(map(str, row)) + r' \\'))

        table.append(NoEscape(r'\hline'))


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
    cv.drawContours(mask, contours, -1, 1, 2)

    # Return the logical 'and' between the inverse of the original slice and the mask
    return np.logical_and(np.logical_not(slice), mask)


def get_overlaying_plots(axes, segmentations, image, min_value, max_value, c_map=_PETRainbowCMAP, c_map_contour='plasma_r'):
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
        Colormap to use for the PET image. Default is `_PETRainbowCMAP`.

    Returns:
    --------
    None
    '''
    # Extract the contours of the segmentations
    segmentations_cont = contours_mask_slice(segmentations)[FIRST_DIM_CROPPED, SECOND_DIM_CROPPED]
       
    # Mask the PET image where segmentations are not present
    im_ma = np.ma.array(image, mask=np.logical_not(segmentations_cont))
        
    # Display the PET image
    kwargs = {'interpolation': 'none', 'vmin': min_value, 'vmax': max_value}
    axes.imshow(np.rot90(image), cmap=c_map, **kwargs)

    # Overlay the masked segmentation contours with an 'autumn' colormap
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
    slice_numbers = sorted(np.unique([x[2] for x in np.argwhere(masks)]))

    # Select and return the slices based on the interval
    return  slice_numbers[0::int(np.ceil((len(slice_numbers) * 1.0) / num_slices))]


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
    min_value = -0.4
    max_value = 0.9 * np.max(pet*(mask.astype(bool)))
    
    # Create a new figure in the LaTeX document
    with doc.create(Figure(position='h!')) as plot:
        doc.append(Command('centering'))
        # Create a subfigure for the plots
        with doc.create(SubFigure(position='t', width=NoEscape(r'0.9\linewidth'))) as subplot1:
            fig, axes = plt.subplots(
                1, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(15, 5)
            )
            margins = {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
            fig.subplots_adjust(**margins)

            segmentations = mask
            for i, k in enumerate(range(3, 6)):
                axes[i].axis('off')
                ind = slices[k]
                pet_crop = pet[FIRST_DIM_CROPPED,SECOND_DIM_CROPPED, ind]
                get_overlaying_plots(axes[i], segmentations[:, :, ind], pet_crop, min_value, max_value)
                get_image_sides(axes[i])

            subplot1.add_plot()

        doc.append(NoEscape(r'\par \vfill'))
        add_colormap_plot(doc, plt, vmin=0, vmax=max_value, step=1 if max_value < 5 else 2)

    doc.append(NoEscape(r'\vspace*{-0.3cm}'))


def plot_collapse_pet(img_pet, seg, axial_slices, vmin, vmax):
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

    axes.axis('off')
    axes.imshow(np.rot90(img_pet_collapse[xslices, yslices]), vmin=vmin, vmax=vmax, aspect='equal', cmap=_PETRainbowCMAP)
    axes.text(2, 5, 'R', color='#f9f9f9', fontsize=45)
    axes.text(xrang - 5, 5, 'L', color='#f9f9f9', fontsize=45)


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

    with doc.create(SubFigure(width=NoEscape(subfig_width))) as subplot:
        doc.append(Command('centering'))
        doc.append(NoEscape(r'{\small\textbf{' + title + r'}}\\'))
        doc.append(NoEscape(r'\vspace{0.2cm}'))
        
        plot_collapse_pet(
            norm_pet,
            mask,
            slice(slices[6]-1, slices[0]+1),
            vmin=min_value,
            vmax=max_value
        )
        
        subplot.add_plot(width=NoEscape(subfig_width))

        doc.append(NoEscape(r'\par \vfill'))
        if title == 'Absolute scale':
            add_max_tick = True
        else:
            add_max_tick = False
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
    avgmax = 0.9 * np.max(pet[:, :, (slices[6]-1):(slices[0]+1)].mean(axis=2))
    with doc.create(Figure(position='!htp')):
        doc.append(Command('centering'))
        subfig_width = r'8cm'
        colormap_width = r'5cm'
        add_average_plot(doc, pet, mask, 'Relative scale', subfig_width, colormap_width, min_value=-1, max_value=avgmax, slices=slices)
        add_average_plot(doc, pet, mask, 'Absolute scale', subfig_width, colormap_width, min_value=-1, max_value=4.0, slices=slices)


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
    min_value = -0.4
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
                get_overlaying_plots(axes[i, j], segmentations[:, :, ind], pet_crop, min_value, max_value, _PETRainbowCMAP)
                get_image_sides(axes[i, j])  # Add R/L labels

            subplot1.add_plot()  # Add plot to the LaTeX document

        doc.append(NoEscape(r'\par \vfill'))
        
        if '_' in pet_desc:
            pet_desc = pet_desc.replace('_', r'\_')
        # Add study description to the LaTeX document
        doc.append(NoEscape(r'{\scriptsize{' + pet_desc + r'}}\\'))

        # Add colormap to the LaTeX document
        add_colormap_plot(doc, plt, vmin=0, vmax=max_value, step=1 if max_value < 5 else 2)

    
def plot_ct(doc, ct, mask, ct_desc, slices):
    """
    Plots a 3x3 grid of CT images with overlays of segmentations.

    Parameters:
    -----------
    doc : Document
        LaTeX document to append the plot.
    ct : array
        CT image as a numpy array.
    mask : array
        Segmentation mask for the CT image.
    ct_desc : str
        Description of the CT study.
    slices : list
        List of slice indices to use for plotting.

    Returns:
    --------
    None
    """

    # Define intensity range for CT display
    min_value = 0
    max_value = 100


    doc.append(NoEscape(r'\vspace*{-0.3cm}'))
    
    # Create figure for CT plots
    with doc.create(Figure(position='h!')) as plot:
        doc.append(Command('centering'))
        with doc.create(SubFigure(position='t', width=NoEscape(r'0.8\linewidth'))) as subplot1:  
            # Create a 3x3 grid for displaying CT images
            fig, axes = plt.subplots(3, 3, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(15, 15))
            
            # Adjust margins to minimize whitespace
            margins = {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
            fig.subplots_adjust(**margins)

            cmap = 'Greys_r'  # Define colormap for CT images
            segmentations = mask

            # Iterate through 3x3 grid positions to plot CT slices
            for k, (i, j) in enumerate([(i, j) for i in range(0, 3) for j in range(0, 3)]):
                axes[i, j].axis('off')  # Turn off axis for each subplot
                ind = slices[k]  # Select slice index
                ct_crop = ct[FIRST_DIM_CROPPED, SECOND_DIM_CROPPED, ind]  # Crop the CT image
                
                # Plot CT image with segmentation overlay
                get_overlaying_plots(axes[i, j], segmentations[:, :, ind], ct_crop, min_value, max_value, cmap, c_map_contour='autumn_r')
                get_image_sides(axes[i, j])  # Add R/L labels

            subplot1.add_plot()  # Add plot to the LaTeX document
    
    doc.append(NoEscape(r'\vspace{-0.7cm}'))

    if '_' in ct_desc:
        ct_desc = ct_desc.replace('_', r'\_')
    # Add study description to the LaTeX document
    doc.append(NoEscape(r'{\scriptsize{' + ct_desc + r'}}\\'))
    
    # Add note on CT image usage for anatomical reference
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
    
    now = datetime.now()
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
        
        table.add_row(bold("Location"), bold(" "), bold("Ratio"), bold(" "),
                      MultiColumn(5, align="c", data=bold("Z-score")))
        for row in produce_row(patient_values, "Posterior Putamen / Caudate Nucleus", normal_stat_values, pt_age):
            table.add_row(*list(row))
        
        table.add_hline()

        table.add_row(
            bold("Location"), MultiColumn(3, align="c", data=bold("Hemisphere")),
            MultiColumn(5, align="c", data=bold("Z-score")))
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

    return Figure()._save_plot()


def plots_normal_values(doc, normal_values, patient_values, pt_age):
    """
    Generates a LaTeX figure containing two subplots that compare the patient's SBR values 
    against the normal reference intervals for the Putamen and the Putamen/Caudate Nucleus ratio.

    Parameters:
    -----------
    doc : Document
        The LaTeX document object.
    normal_values : DataFrame
        DataFrame containing normal reference values for comparison.
    patient_values : DataFrame
        DataFrame containing the patient's SBR values.
    pt_age : int
        The age of the patient.

    Returns:
    --------
    None
    """

    subfig_width = r'8cm'
    
    with doc.create(Figure(position='!htp')) as plot:
        doc.append(Command('centering')) 

        with doc.create(SubFigure(width=NoEscape(subfig_width))) as subplot1:
            doc.append(Command('centering'))

            # Generate the plot for the Putamen region
            plot_normal(normal_values, patient_values, pt_age, "Putamen", legend=True)

            subplot1.add_plot(width=subfig_width)

        doc.append(NoEscape(r'\hspace{0.7cm}'))
        
        with doc.create(SubFigure(width=NoEscape(subfig_width))) as subplot2:
            doc.append(Command('centering'))

            # Generate the plot for the Putamen/Caudate Nucleus ratio
            plot_normal(normal_values, patient_values, pt_age, "Putamen / Caudate Nucleus")

            subplot2.add_plot(width=subfig_width)


def generate_report(self, ref_pet_dcm, ct_desc, normalised_pet, ct_nii, prediction, cerebellum, patient_values):
    """
    Generates a comprehensive report including PET and CT scan analysis, patient data, and reference comparisons.

    Parameters:
    -----------
    ref_pet_dcm : DICOM
        Reference DICOM metadata for PET scan.
    ct_desc : str
        Description of the CT scan.
    normalised_pet : array
        normalised PET scan data array.
    ct_nii : str
        CT scan path.
    prediction : array
        Predicted scan data array.
    cerebellum : array
        Cerebellum scan data array.
    patient_values : DataFrame
        DataFrame containing patient-specific values.

    Returns:
    --------
    doc : Document
        The generated LaTeX document object.
    ds : DICOM
        The DICOM header information.
    """
    self.logger.info('Generating report')
    # Flip PET, prediction, and cerebellum arrays along the axis 0 (typically to adjust orientation)
    pet_flipped = np.flip(normalised_pet, axis=0) 
    pet_flipped = np.nan_to_num(pet_flipped)
    prediction_flipped = np.flip(prediction, axis=0)
    cerebellum_flipped= np.flip(cerebellum, axis=0)
    ct = nib.load(ct_nii).get_fdata()
    ct_flipped = np.flip(ct, axis=0) 
    ct_flipped = np.nan_to_num(ct_flipped)

    # Create mask by summing flipped prediction and cerebellum arrays
    mask = prediction_flipped + cerebellum_flipped

    # Extract metadata from the reference PET DICOM
    pet_desc = ref_pet_dcm.SeriesDescription
    institution = ref_pet_dcm.InstitutionName 
    if institution == 'Nuklearmedicin':
        institution = 'Rigshospitalet'

    # Load normal values and statistical data
    normal_values = pd.read_csv(os.path.join(STATIC_FILES, 'normal_values1.csv'), index_col=0)

    # Select statistical data based on the institution
    if 'Bispebjerg' in institution:
        normal_stat_values = pd.read_csv(os.path.join(STATIC_FILES, 'stats_bbh1.csv'), index_col=0)
        normal_values = normal_values[normal_values['institution'] == 'BBH']
    else:
        normal_stat_values = pd.read_csv(os.path.join(STATIC_FILES, 'stats_rh_auh1.csv'), index_col=0)   
        normal_values = normal_values[normal_values['institution'] != 'BBH']
    
    # Extract age range and patient age
    age_range = (normal_values[['age']].values).astype(int)
    pt_age = get_age(ref_pet_dcm.PatientAge)
    
    # Generate slices from prediction and cerebellum data
    slices = get_slices(prediction_flipped, indices = [2,3], num_slices = 9)[2:8] + get_slices(cerebellum_flipped, indices = [CEREBELLUM_INDEX], num_slices = 4)[1:]
    slices.sort(reverse=True)
    
    # Define the path for output LaTeX document
    header_doc = self.processing_directory / 'doc'
    doc = create_document(header_doc)
    doc.preamble.append(Command('usepackage', 'xcolor')) 
    # Define the custom color
    doc.append(NoEscape(r'\definecolor{babyblue}{RGB}{183, 227, 249}')) 
    
    # Add report header with institution name
    get_report_header(doc, institution) 
    
    doc.append(NoEscape(r'\vspace{-1.2cm}'))
    doc.append(NoEscape(r'\colorbox{babyblue}{\begin{minipage}{\linewidth}'))
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
        get_patient_table(doc, ref_pet_dcm)
        
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
    
    # Create section for average basal ganglia SBR
    with doc.create(Section('Average basal ganglia SBR', numbering=False)):
        doc.append(NoEscape(r'\vspace{-0.2cm}'))
        plot_average(doc, pet_flipped, mask, slices)

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
    
    # Create section for CT Cerebrum
    with doc.create(Section('CT Cerebrum', numbering=False)):
        doc.append(NoEscape(r'\vspace{-0.2cm}'))
        doc.append(Command('centering'))
        plot_ct(doc, ct_flipped, mask, ct_desc, slices)

    # Close all plots
    plt.close('all')

    # Generate LaTeX document and PDF
    doc.generate_tex()
    doc.generate_pdf(header_doc)

    return str(header_doc)+'.pdf'


def normalize(logger,ct_brain_nii, pet_nii): 
    """
    Normalizes the PET and CT brain images. 
    PET is normalized by subtracting the mean value within a mask, and CT is thresholded and normalized.
   
    Function that normalizes pet with mean value from (mask) NORMALIZATION MASK - which is a big region surrounding and containing putamens and caudate nucleui
    it also normalizes ct by taking mean value form thresholded ct between 30-50 HU
    
    Parameters:
    -----------
    ct_brain_nii : str
        File path to the CT brain NIfTI image.
    pet_nii : str
        File path to the PET NIfTI image.
    logger : Logger object
        Logger for logging the process information.
        
    Returns:
    --------
    normalized_data : np.array
        Array containing the normalized PET and CT data.
    
    Raises:
    -------
    FileNotFoundError
        If any of the input files do not exist.
    """

    # Check if input files exist
    if not Path(ct_brain_nii).is_file():
        logger.error(f"CT brain file not found: {ct_brain_nii}")
        raise FileNotFoundError(f"CT brain file not found: {ct_brain_nii}")
    if not Path(pet_nii).is_file():
        logger.error(f"PET file not found: {pet_nii}")
        raise FileNotFoundError(f"PET file not found: {pet_nii}")
    
    # Load CT and PET images
    ct_brain = nib.load(ct_brain_nii).get_fdata()
    pet = nib.load(pet_nii).get_fdata()

    # Load the normalization mask
    mask_nii = STATIC_FILES / 'truth_merge_mask_dil10.nii.gz'
    mask = nib.load(mask_nii).get_fdata()

    # Normalize PET by subtracting the mean value within the mask
    pet_norm_mean_mask = np.mean(pet[mask != 0])
    normalized_pet = pet - pet_norm_mean_mask

    # Normalize CT by thresholding between 30-50 HU and subtracting the mean value
    filtered_data = ct_brain[(ct_brain != 0) & (ct_brain > 30) & (ct_brain < 50)]
    ct_norm_mean_mask = np.mean(filtered_data)
    normalized_ct = ct_brain - ct_norm_mean_mask

    return  np.asarray([normalized_pet, normalized_ct])


def reg_aladin(ref_file, flo_file, aff_file, rig_only_flag=False, in_aff_file=None, aff_direct_flag=True, res_file=None, verbosity=None):
    '''
    Block Matching algorithm for symmetric global registration
    Args:
        ref_file (a pathlike object or str): The input reference/target image
        flo_file (a pathlike object or str): The input floating/source image
        aff_file (a pathlike object or str): The output affine matrix file
        res_file (a pathlike object or str): The affine transformed floating image
        verbosity (None or str): One of file, file_split, file_stdout,
                                 file_stderr, stream, allatonce, none
    Returns:
        Runtime object (except for verbosity='none').
        Access errors by e.g.:
            result = reg_aladin(...,verbosity='file_stdout')
            result.runtime.stdout
    '''

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
    '''Resample nifty file to a reference template given a transformation matrix

    Args:
        ref_file (a pathlike object or str): The input reference/target image
        flo_file (a pathlike object or str): The input floating/source image
        trans_file (a pathlike object or str): The input transformation matrix file
        out_file (a pathlike object or str): The output filename of the transformed image
        interpol ('NN' or 'LIN' or 'CUB' or 'SINC'): Type of interpolation. Defaults to 'NN'.
        pad_val (float, optional): Padding value to pad. Defaults to None.
        verbosity (None or str): One of file, file_split, file_stdout,
                                 file_stderr, stream, allatonce, none

    Returns:
        Runtime object (except for verbosity='none').
        Access errors by e.g.:
            result = reg_resample(...,verbosity='file_stdout')
            result.runtime.stdout
    '''

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
    '''
    Load a pre-trained Keras model with custom objects.

    Args:
        model_file (str): Path to the pre-trained model file.

    Returns:
        keras.Model: Loaded Keras model with the custom objects.

    Raises:
        ValueError: If there's an error related to 'InstanceNormalization' and keras-contrib is not installed.
    '''
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
    '''
    Function for prediction of caudate nuclei and putamen using 8 patches.

    Args:
        model (keras.Model): U-Net model used for prediction.
        data (np.ndarray): PET and CT data in one numpy array.

    Returns:
        np.ndarray: Prediction of caudate nuclei and putamen reconstructed from the patches.
    '''
    patch_shape = np.asarray([int(dim) for dim in model.input.shape[-3:]]) # 80, 48, 48
    predictions = list()
    indices = compute_patch_indices()
    
    for i in range(len(indices)):
        patch = get_patch_from_3d_data(data, patch_shape=patch_shape, patch_index=indices[i])[np.newaxis]
        prediction = model.predict(patch, verbose=0)
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
        output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])

    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)


def get_patch_from_3d_data(data, patch_shape, patch_index):
    '''
    Extract a patch from 3D data (both CT and PET) numpy array.

    Args:
        data (np.ndarray): CT and PET numpy array from which to get the patch.
        patch_shape (tuple): Shape/size of the patch.
        patch_index (tuple): Corner index of the patch.

    Returns:
        np.ndarray: Cropped CT and PET data with the specified patch shape.
    ''' 
    return data[..., 
                patch_index[0]: patch_index[0] + patch_shape[0], 
                patch_index[1]: patch_index[1] + patch_shape[1],
                patch_index[2]: patch_index[2] + patch_shape[2]]


def reconstruct_from_patches(patches, patch_indices, data_shape):
    '''
    Reconstruct an array of the original shape from the list of patches and corresponding patch indices. Overlapping
    patches are averaged.

    Args:
        patches (list): List of prediction patches as numpy arrays.
        patch_indices (list): List of indices corresponding to the patches.
        data_shape (tuple): Shape of the array from which the patches were extracted.

    Returns:
        np.ndarray: Data reconstructed from the patches.
    '''
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
    '''
    Convert model prediction to a labeled image based on a threshold.

    Args:
        prediction (np.ndarray): Model prediction.
        threshold (float): Threshold for labeling. Defaults to 0.5.
        labels (list, optional): List of labels for each class. Defaults to None.

    Returns:
        np.ndarray: Labeled image.
    '''
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
        label = labels[0] if labels else 1
        label_map_data[data > threshold] = label
        data = label_map_data
    elif prediction.shape[1] > 1:
        label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
        data = label_map_data[0]
    else:
        raise RuntimeError('Invalid prediction array shape: {0}'.format(prediction.shape))
    
    return data


def compute_patch_indices():
    '''
    Compute indices for extracting patches from the data.

    Returns:
        np.ndarray: Array of patch indices.

    output look like this: x,y,z
    array([[ 80,  94, 101],
       [ 80,  94, 112],
       [ 80, 114, 101],
       [ 80, 114, 112],
       [ 94,  94, 101],
       [ 94,  94, 112],
       [ 94, 114, 101],
       [ 94, 114, 112]])
    '''
    start = np.array([80, 94, 101])
    stop = np.array([95, 115, 113])
    step = np.array([14, 20, 11])
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], 
                               start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int16)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    '''
    Convert prediction scores to labeled arrays based on a threshold.

    Args:
        prediction (np.ndarray): Array of prediction scores with shape (n_samples, n_classes, ...).
        threshold (float): Minimum score to consider for a label. Defaults to 0.5.
        labels (list, optional): List of labels corresponding to class indices. Defaults to None.

    Returns:
        list: List of labeled arrays for each sample.
    '''
    n_samples = prediction.shape[0] 
    label_arrays = []

    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data)[1:]: 
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(label_data.astype(np.uint8))

    return label_arrays


def dice_coefficient_loss(y_true, y_pred):
    '''
    Compute the Dice coefficient loss.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: The negative Dice coefficient (as a loss function).
    '''
    return -dice_coefficient(y_true, y_pred)


def dice_coefficient(y_true, y_pred, smooth=1.):
    '''
    Compute the Dice coefficient for evaluating the similarity 
    between the ground truth and predicted masks.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.
        smooth (float): Smoothing constant to avoid division by zero. Defaults to 1.

    Returns:
        tf.Tensor: The Dice coefficient.
    '''
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def tversky_loss(y_true, y_pred):
    '''
    Compute the Tversky loss for imbalanced datasets, which is a generalization of the Dice coefficient.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: The Tversky loss.
    '''
    alpha = 0.3
    beta = 0.7

    ones = tf.keras.backend.ones(tf.keras.backend.shape(y_true))
    p0 = y_pred       # Probability that voxels are of the predicted class
    p1 = ones - y_pred  # Probability that voxels are not of the predicted class
    g0 = y_true       # Ground truth
    g1 = ones - y_true # Inverse of ground truth

    numerator = tf.keras.backend.sum(p0 * g0, (0, 1, 2, 3))
    denominator = numerator + alpha * tf.keras.backend.sum(p0 * g1, (0, 1, 2, 3)) + beta * tf.keras.backend.sum(p1 * g0, (0, 1, 2, 3))

    T = tf.keras.backend.sum(numerator / denominator)
    # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = tf.keras.backend.cast(tf.keras.backend.shape(y_true)[-1], 'float32')

    return Ncl - T


def tversky_coef(y_true, y_pred):
    '''
    Compute the Tversky coefficient, which is the negative Tversky loss.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: The negative Tversky loss (as a coefficient).
    '''
    return -tversky_loss(y_true, y_pred)


def generalized_dice_loss(y_true, y_pred):
    '''
    Compute the generalized Dice loss, which accounts for class imbalance by 
    weighting each label's contribution inversely proportional to its volume.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: The generalized Dice loss.
    '''

    Ncl = y_pred.shape[-1]
    w = np.zeros((Ncl,))
    for l in range(0, Ncl):
        w[l] = np.sum(np.asarray(y_true[:, :, :, :, l] == 1, np.int8))
    w = 1 / (w**2 + 0.00001)

    # Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = w * tf.keras.backend.sum(numerator, (0, 1, 2, 3))
    numerator = tf.keras.backend.sum(numerator)

    denominator = y_true + y_pred
    denominator = w * tf.keras.backend.sum(denominator, (0, 1, 2, 3))
    denominator = tf.keras.backend.sum(denominator)

    gen_dice_coef = numerator / denominator

    return 1 - 2 * gen_dice_coef


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=1e-5):
    '''
    Compute the weighted Dice coefficient for evaluating the similarity 
    between the ground truth and predicted masks.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.
        axis (tuple of int): Axes along which to compute the Dice coefficient.
                             Defaults to (-3, -2, -1) assuming 'channels first' data format.
        smooth (float): Smoothing constant to avoid division by zero. Defaults to 0.00001.

    Returns:
        tf.Tensor: The mean Dice coefficient.
    '''
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=axis) + smooth / 2
    
    # Compute the sum of y_true and y_pred, and add smooth
    summation = tf.keras.backend.sum(y_true, axis=axis) + tf.keras.backend.sum(y_pred, axis=axis) + smooth
    
    # Compute the Dice coefficient
    dice = 2. * intersection / summation
    
    # Compute the mean Dice coefficient
    return tf.keras.backend.mean(dice)


def get_label_dice_coefficient_function(label_index):
    '''
    Create a function to compute the Dice coefficient for a specific label.

    Args:
        label_index (int): Index of the label for which the Dice coefficient is computed.

    Returns:
        function: A function that computes the Dice coefficient for the specified label.
    '''
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))

    return f


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    '''
    Compute the Dice coefficient for a specific label.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.
        label_index (int): Index of the label for which the Dice coefficient is computed.

    Returns:
        tf.Tensor: The Dice coefficient for the specified label.
    '''
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def weighted_dice_coefficient_loss(y_true, y_pred):
    '''
    Compute the weighted Dice coefficient loss.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: The negative weighted Dice coefficient (as a loss function).
    '''
    return -weighted_dice_coefficient(y_true, y_pred)
