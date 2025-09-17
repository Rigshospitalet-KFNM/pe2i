'''
add to .env file:
OUTPUT_PATH="/data/"
STATIC_PATH="/report_data/"
'''
import re
import os
import json
import logging
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any
import shutil
import dotenv
dotenv.load_dotenv()
from datetime import datetime
from pydicom.uid import PositronEmissionTomographyImageStorage, CTImageStorage, MRImageStorage
import pe2i_petct_functions as node_functions
from pe2i_environment import environment as env
import dicomnode
import dicomnode.server
from dicomnode.dicom.dimse import Address
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import DicomOutput, MultiOutput
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.grinders import NiftiGrinder, ManyGrinder, IdentityGrinder
from dicomnode.dicom.blueprints import Blueprint, StaticElement, CopyElement, FunctionalElement, get_today, get_time
from dicomnode.dicom.blueprints.secondary_image_report_blueprint import SECONDARY_IMAGE_REPORT_BLUEPRINT
from dicomnode.dicom.blueprints.error_blueprint_english import ERROR_BLUEPRINT
from dicomnode.dicom.dicom_factory import DicomFactory
from dicomnode.lib.validators import RegexValidator, NegatedValidator, CaselessRegexValidator
import pydicom.config
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
# Your pydicom configuration
pydicom.config.convert_wrong_length_to_UN = True



OUTPUT_PATH = env.OUTPUT_PATH
factory  = DicomFactory()
error_blueprint = ERROR_BLUEPRINT

PET_ARCHIVE = Address('10.49.144.6', 104, 'GOYA') # These should be  in .env

DICOM_ROUTER = Address('10.143.10.61', 104, 'VIPDICOM')

BISPEBJERG_SCANNER_1 = Address('172.23.48.81', 104, 'BFHKFNM7101')
BISPEBJERG_SCANNER_2 = Address('172.23.48.82', 104, 'BFHKFNM7102')
BISPEBJERG_SCANNER_3 = Address('172.23.48.83', 104, 'BFHKFNMMI1')
BISPEBJERG_PET_ARCHIVE = Address('172.23.48.110', 11112, 'BBHKFNMOSIRIX')
BISPEBJERG_PROD_ARCHIVE = Address('172.23.48.76', 11112, 'BBHKFAGW1')

class MyCTInput(AbstractInput):
    """
    Handles input data for CT images.
    """

    enforce_single_series = True

    def validate(self) -> bool:
        maxInstanceNumber = -1

        # Iterate through datasets to find the maximum instance number
        for dataset in self:
            maxInstanceNumber = max(maxInstanceNumber, dataset.InstanceNumber)

        # Check if the number of images matches the maximum instance number
        return self.images == maxInstanceNumber

    # Image grinder object for processing NIfTI images
    image_grinder = ManyGrinder(NiftiGrinder(), IdentityGrinder())

    # Required DICOM tags and their expected values
    required_values: Dict[int, Any] = {
        0x00080016 : CTImageStorage,
        0x00080060 : "CT",  # DICOM Modality Tag
        0x0008_103E : NegatedValidator(CaselessRegexValidator("topogram")),
    }

class MyPETInput(AbstractInput):
    """
    Handles input data for PET images.
    """

    enforce_single_series = True

    def validate(self) -> bool:
        maxInstanceNumber = -1
        # Iterate through datasets to find the maximum instance number
        for dataset in self:
            maxInstanceNumber = max(maxInstanceNumber, dataset.InstanceNumber)
        # Check if the number of images matches the maximum instance number
        return self.images == maxInstanceNumber


    # Image grinder object for processing NIfTI images
    image_grinder = NiftiGrinder()

    # Required DICOM tags and their expected values
    required_values: Dict[int, Any] = {
        0x00080016 : PositronEmissionTomographyImageStorage,
        0x00080060 : "PT"  # DICOM Modality Tag
    }


class MyMRInput(AbstractInput):
    """
    Handles input data for PET images.
    """

    enforce_single_series = True

    def validate(self) -> bool:
        maxInstanceNumber = -1
        # Iterate through datasets to find the maximum instance number
        for dataset in self:
            maxInstanceNumber = max(maxInstanceNumber, dataset.InstanceNumber)
        # Check if the number of images matches the maximum instance number (+1 DD starts with 0)
        return self.images == maxInstanceNumber + 1


    # Image grinder object for processing NIfTI images
    image_grinder = ManyGrinder(NiftiGrinder(), IdentityGrinder())

    # Required DICOM tags and their expected values
    required_values: Dict[int, Any] = {
        0x00080016 : MRImageStorage,
        0x00080060 : "MR"  # DICOM Modality Tag
    }


class Pe2iPetCtNode(AbstractQueuedPipeline):
    """
    Main pipeline node for processing PET and CT data, and generating reports.
    """
    # Factory for creating DICOM objects
    dicom_factory = factory

    # Path for logging output
    #log_path: str = "/home/zuza/pe2i/pe2ipetctnode.log"

    # AE Title for DICOM nodes (Application Entity Title)
    ae_title: str = "PE2IPETCTNODE"

    # Directory for processing output
    processing_directory = env.OUTPUT_PATH
    data_directory = env.STORAGE_PATH
    # Network settings
    port: int = 1131
    ip: str = '0.0.0.0'

    # Logger settings: disable pynetdicom logger and set log level
    disable_pynetdicom_logger = True
    log_level: int = logging.DEBUG
    log_output = env.LOG_PATH

    error_on_rejected_dataset = False

    # Blueprint for handling unhandled errors
    unhandled_error_blueprint = error_blueprint

    known_endpoints = {
        BISPEBJERG_SCANNER_1.ae_title : BISPEBJERG_SCANNER_1,
        BISPEBJERG_SCANNER_2.ae_title : BISPEBJERG_SCANNER_2,
        BISPEBJERG_SCANNER_3.ae_title : BISPEBJERG_SCANNER_3,
        BISPEBJERG_PET_ARCHIVE.ae_title : BISPEBJERG_PET_ARCHIVE,
        BISPEBJERG_PROD_ARCHIVE.ae_title : BISPEBJERG_PROD_ARCHIVE
    }

    # Input types for the pipeline
    input = {
        'anatomical': MyCTInput | MyMRInput,
        'PET': MyPETInput
    }

    # Endpoint for output
    endpoint = Address('10.49.144.35', 104, "VIA2") ## TODO move into .env

    def process(self, input_data: InputContainer):
        """
        Processes the input data, converts DICOM to NIfTI, performs necessary operations,
        and generates a report.

        Parameters:
        -----------
        input_data: InputContainer
            Container holding the input data for PET and CT.

        Returns:
        --------
        DicomOutput:
            The generated report in DICOM format.
        """

        # Extract PET and CT data from input
        ref_pet_dicom = input_data.datasets['PET'][0]
        ref_pet_dicoms = input_data.datasets['PET']
        ref_anatomical_dicom = input_data.datasets['anatomical'][0]
        pet = input_data['PET'] # NIfTI PET data
        anatomical_nifti, inputContainer = input_data['anatomical'] # NIfTI CT or DD data and input class
        nib.save(pet, env.get_patient_work_directory(ref_pet_dicom.PatientID) / "pet.nii.gz")
        # Reference DICOM datasets

        # Get CT series description (metadata)
        anatomical_desc = ref_anatomical_dicom.SeriesDescription
        # this is added for validation
        pt_id = ref_anatomical_dicom.StudyInstanceUID


        with env.VALIDATION_PATH.open('a') as file:
            file.write('\n' + pt_id)

        # Perform preprocessing steps on PET and CT/DD data:
        # Swap dimensions for PET and CT/DD (function defined in node_functions)
        pet_swap_path = node_functions.swap_dims(self, pet, 'PET')
        if isinstance(inputContainer, MyMRInput):
            dd_swap_path = node_functions.swap_dims(self, anatomical_nifti, 'DD')
            anatomical_swap_path = node_functions.convert_LAC_to_HU(self, dd_swap_path) # converting Linear Attenuation Coefficient units to Hounsefield Units
            MR_flag = True # MR flag for report settings
        else:
            anatomical_swap_path = node_functions.swap_dims(self, anatomical_nifti, 'CT')
            MR_flag = False# MR flag for report settings

        # Skull stripping on CT data
        anatomical_bet_path = node_functions.run_skullstripping(self, anatomical_swap_path)

        # Resampling PET, CT, and skull-stripped CT images to the same resolution
        pet_resampled_path, anatomical_resampled_path, anatomical_bet_resampled_path, trans_pet, trans_anatomical = node_functions.resampling(
            self, pet_swap_path, anatomical_swap_path, anatomical_bet_path
        )

        # Further processing of CT data (e.g., preprocessing)
        anatomical_bet_preproc_path = node_functions.process_anatomical(self, anatomical_bet_resampled_path)

        # Masking cerebellum cortex in CT data
        cerebellum_path = node_functions.cerebellum_mask(self, anatomical_bet_preproc_path)

        # Get prediction data and statistics based on PET and CT data
        prediction_data = node_functions.get_predition(self.logger, anatomical_bet_preproc_path, pet_resampled_path)
        pet_normalized_data, cerebellum_mask_data, patient_values, cerebellum_median = node_functions.get_statistics(
            self.logger, pet_resampled_path, cerebellum_path, prediction_data
            )

        # Generate the report
        report = node_functions.generate_report(
            self, ref_pet_dicom, anatomical_desc, pet_normalized_data, anatomical_resampled_path,
            prediction_data, cerebellum_mask_data, patient_values, MR_flag
        )

        # pet_normalized_pet_space = node_functions.reverse_pet_resampling(self, pet_normalized_data, pet_resampled_path, pet_swap_path, anatomical_bet_path, trans_anatomical, trans_pet)
        pet_sbr_data = pet.get_fdata() / cerebellum_median - 1
        self.logger.info(np.max(pet_sbr_data))
        pet_normalized_org = nib.Nifti1Image(pet_sbr_data, pet.affine)
        time_now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        modality_name = f"PET PE2I SBR {time_now}"
        self.logger.info(len(ref_pet_dicoms))
        if not isinstance(ref_pet_dicoms, list):
            ref_pet_dicoms = [ref_pet_dicoms]
        pet_dcm = node_functions.get_pet_dicom(self, pet_normalized_org, ref_pet_dicoms, modality_name)
        # pet_dcm = node_functions.nifti_to_pet_dicom(self, pet_normalized_swapped, ref_pet_dicoms, modality_name)
        # file_path = '/home/zuza/validation/' + str(pt_id) +'.json'
        # with open(file_path, 'w') as json_file:
        #     json.dump(patient_values, json_file, indent=4)

        # Extract keys from the patient_values dictionary to add them to the DICOM report
        keys = list(patient_values.keys())

        # Define the report name and create a blueprint for encoding the DICOM report
        report_name = f"PE2I Report V2.0 {time_now}"
        blueprint= Blueprint(SECONDARY_IMAGE_REPORT_BLUEPRINT)

        # Populate the blueprint with relevant DICOM tags and values
        blueprint[0x0008_0021] = CopyElement(0x0008_0021) # Series Date
        blueprint[0x0008_0023] = FunctionalElement(0x00080023, 'DA', get_today) # Content Date
        blueprint[0x0008_0031] = CopyElement(0x0008_0031) # Series Time
        blueprint[0x0008_0033] = FunctionalElement(0x00080033, 'TM', get_time) # Content Time
        blueprint[0x0008_103E] = StaticElement(0x0008_103E, 'LO', report_name) # Series Description
        blueprint[0x0010_0010] = CopyElement(0x0010_0010) # Patient's Name
        blueprint[0x0020_0011] = StaticElement(0x0020_0011, 'IS', str(random.randint(5000,100000))) # Series Number

        # Add calculated patient values to the blueprint for DICOM output
        for i in range(len(keys)):
            key = keys[i]
            blueprint[0x3003_0101 + i] = StaticElement(
                0x3003_0101 + i,
                'FL',
                np.round(patient_values[key], 2),
                name=f"{key} [SBR]" if i < 14 else f"{key}"
        )

        # Encode the report as a PDF
        encoded_report = self.dicom_factory.encode_pdf(report, [ref_pet_dicom], blueprint)

        # Ensure that all encoded report instances have matching series time
        if encoded_report[0].SeriesTime != encoded_report[1].SeriesTime:
            for i in range(1, len(encoded_report)):
                encoded_report[i].SeriesTime = encoded_report[0].SeriesTime

        # Return the file output containing the generated report
        if input_data.responding_address.ae_title in [BISPEBJERG_SCANNER_1.ae_title,\
                                                      BISPEBJERG_SCANNER_2.ae_title,\
                                                      BISPEBJERG_SCANNER_3.ae_title,\
                                                      BISPEBJERG_PET_ARCHIVE.ae_title,\
                                                      BISPEBJERG_PROD_ARCHIVE.ae_title]:
            # return DicomOutput([(BISPEBJERG_PROD_ARCHIVE, encoded_report)], self.ae_title)
            return DicomOutput([(BISPEBJERG_PROD_ARCHIVE, encoded_report),
                                (BISPEBJERG_PROD_ARCHIVE, pet_dcm)],
                                self.ae_title)

        # return DicomOutput([
        #          (self.endpoint, encoded_report),
        #          (PET_ARCHIVE, encoded_report),
                #  (DICOM_ROUTER, encoded_report)], self.ae_title)
        return DicomOutput([
                 (self.endpoint, encoded_report),
                 (PET_ARCHIVE, encoded_report),
                 (DICOM_ROUTER, encoded_report),
                 (self.endpoint, pet_dcm),
                 (PET_ARCHIVE, pet_dcm),
                 (DICOM_ROUTER, pet_dcm)
                 ], self.ae_title)

# Entry point for running the node
if __name__ == "__main__":
   node = Pe2iPetCtNode()
   node.open()
