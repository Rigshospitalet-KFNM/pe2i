'''
add to .env file:
OUTPUT_PATH="/data/"
STATIC_PATH="/report_data/"
'''
import os
import logging
import random
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, Any
import dotenv
dotenv.load_dotenv()

import pe2i_petct_functions as node_functions
import dicomnode
import dicomnode.server
from dicomnode.dicom.dimse import Address
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import DicomOutput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.grinders import NiftiGrinder
from dicomnode.dicom.blueprints import Blueprint, StaticElement, CopyElement, FunctionalElement, get_today, get_time
from dicomnode.dicom.blueprints.secondary_image_report_blueprint import SECONDARY_IMAGE_REPORT_BLUEPRINT
from dicomnode.dicom.blueprints.error_blueprint_english import ERROR_BLUEPRINT
from dicomnode.dicom.dicom_factory import DicomFactory
import pydicom.config
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
# Your pydicom configuration
pydicom.config.convert_wrong_length_to_UN = True

OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH"))
factory  = DicomFactory()
error_blueprint = ERROR_BLUEPRINT


class MyCTInput(AbstractInput):
    """
    Handles input data for CT images.
    """
    def validate(self) -> bool:
        maxInstanceNumber = 0
        # Iterate through datasets to find the maximum instance number
        for dataset in self:
            maxInstanceNumber = max(maxInstanceNumber, dataset.InstanceNumber)
        # Check if the number of images matches the maximum instance number
        return self.images == maxInstanceNumber
    
    # Image grinder object for processing NIfTI images
    image_grinder = NiftiGrinder()

    # Required DICOM tags and their expected values
    required_values: Dict[int, Any] = {
        0x00080060 : "CT"  # DICOM Modality Tag
    }

class MyPETInput(AbstractInput):
    """
    Handles input data for PET images.
    """
    def validate(self) -> bool:
        maxInstanceNumber = 0
        # Iterate through datasets to find the maximum instance number
        for dataset in self:
            maxInstanceNumber = max(maxInstanceNumber, dataset.InstanceNumber)
        # Check if the number of images matches the maximum instance number
        return self.images == maxInstanceNumber
    

    # Image grinder object for processing NIfTI images
    image_grinder = NiftiGrinder()

    # Required DICOM tags and their expected values
    required_values: Dict[int, Any] = {
        0x00080060 : "PT"  # DICOM Modality Tag
    }


class Pe2iPetCtNode(AbstractPipeline):
    """
    Main pipeline node for processing PET and CT data, and generating reports.
    """
    dicom_factory = factory
    log_path: str = "/var/log/pe2ipetctnode.log"  # Path for logging
    ae_title: str = "PE2IPETCTNODE" # AE Title for DICOM nodes
    
    # Network settings
    port: int = 1131 ## TODO change
    ip: str = '0.0.0.0' ## TODO change

    # Logger settings
    disable_pynetdicom_logger = True
    log_level: int = logging.DEBUG
    log_output = "log.log"
    unhandled_error_blueprint = error_blueprint


    # Input types for the pipeline
    input = {
        'CT': MyCTInput,
        'PET': MyPETInput
    }

    # Endpoint for output
    endpoint = Address('172.16.186.210', 104, "ENDPOINT_AE") ## TODO change

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
        DicomOutput 
            The generated report in DICOM format.
        """
        # Extract PET and CT data from input
        pet = input_data['PET'] # NIfTI PET data
        ct = input_data['CT'] # NIfTI CT data

        # Reference DICOM datasets
        ref_pet_dicom = input_data.datasets['PET'][0]
        ref_ct_dicom = input_data.datasets['CT'][0]
        ct_desc = ref_ct_dicom.SeriesDescription

        # Perform various processing steps on PET and CT data
        pet_swap_nii = node_functions.swap_dims(self.logger, pet, 'PET')
        ct_swap_nii = node_functions.swap_dims(self.logger, ct, 'CT') 
        # ct_bet_nii = node_functions.run_skullstrip(ct_swap_nii)
        ct_bet_nii = node_functions.run_skullstripping(self.logger, ct_swap_nii)
        print(ct_bet_nii)
        pet_resampled_nii, ct_resampled_nii, ct_bet_resampled_nii = node_functions.resampling(
            self.logger, pet_swap_nii, ct_swap_nii, ct_bet_nii
        ) 
        ct_bet_preproc_nii = node_functions.process_ct(self.logger, ct_bet_resampled_nii)
        cerebellum_nii = node_functions.cerebellum_mask(self.logger, ct_bet_preproc_nii)

        prediction_data = node_functions.get_predition(self.logger, ct_bet_preproc_nii, pet_resampled_nii)
        pet_normalized_data, cerebellum_mask_data, patient_values = node_functions.get_statistics(
            self.logger, pet_resampled_nii, cerebellum_nii, prediction_data
            ) # doesnt need to be file 
        print(patient_values)
        # Generate the report
        report = node_functions.generate_report(
            ref_pet_dicom, ct_desc, pet_normalized_data, ct_resampled_nii, 
            prediction_data, cerebellum_mask_data, patient_values
        )

        keys = list(patient_values.keys())

        blueprint= Blueprint(SECONDARY_IMAGE_REPORT_BLUEPRINT)
        blueprint[0x0008_103E] = StaticElement(0x0008_103E, 'LO', 'PE2I report') # Series Description
        blueprint[0x0010_0010] = CopyElement(0x0010_0010) # Patient's Name 
        blueprint[0x0020_0011] = StaticElement(0x0020_0011, 'IS', str(random.randint(5000,100000))) # Series Number
        blueprint[0x0008_0021] = FunctionalElement(0x00080021, 'DA', get_today) #Series Date
        blueprint[0x0008_0031] = FunctionalElement(0x00080031, 'TM', get_time) #Series Time
        for i in range(len(keys)):  
            key = keys[i]
            blueprint[0x3003_0101 + i] = StaticElement(
                0x3003_0101 + i,
                'FL',
                np.round(patient_values[key], 2),
                name=f"{key} [SBR]" if i < 14 else f"{key}"
    )
        #blueprint[0x3003_1000] = StaticElement(0x3003_1000, 'LO', patient_values)
        # Encode the report as a PDF
        encoded_report = self.dicom_factory.encode_pdf(report, [ref_pet_dicom], blueprint)
        print(encoded_report[0])
        # Clean up and prepare output directory
        # shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
        # if not OUTPUT_PATH.is_file():
        #     os.mkdir(OUTPUT_PATH)

        # Return the file output containing the generated report
        return dicomnode.server.output.FileOutput([(Path(OUTPUT_PATH), encoded_report)])

        return DicomOutput([(self.endpoint, [encoded_report]),], self.ae_title)
       
# Entry point for running the node
if __name__ == "__main__":
   node = Pe2iPetCtNode()
   node.open()
