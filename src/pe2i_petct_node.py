'''
add to .env file:
OUTPUT_PATH="/data/"
STATIC_PATH="/report_data/"
'''
import os
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
import dotenv
dotenv.load_dotenv()
from datetime import datetime

import pe2i_petct_functions as node_functions
import dicomnode
import dicomnode.server
from dicomnode.dicom.dimse import Address
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import DicomOutput
from dicomnode.server.nodes import AbstractQueuedPipeline
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

PET_ARCHIVE = Address('10.49.144.6', 104, 'GOYA') # These should be  in .env

class MyCTInput(AbstractInput):
    """
    Handles input data for CT images.
    """
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
        0x00080060 : "CT"  # DICOM Modality Tag
    }

class MyPETInput(AbstractInput):
    """
    Handles input data for PET images.
    """
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
        0x00080060 : "PT"  # DICOM Modality Tag
    }


class Pe2iPetCtNode(AbstractQueuedPipeline):
    """
    Main pipeline node for processing PET and CT data, and generating reports.
    """
    # Factory for creating DICOM objects
    dicom_factory = factory
    
    # Path for logging output
    log_path: str = "/var/log/pe2ipetctnode.log"
    
    # AE Title for DICOM nodes (Application Entity Title)
    ae_title: str = "PE2IPETCTNODE" 
    
    # Directory for processing output
    processing_directory = OUTPUT_PATH

    # Network settings
    port: int = 1131
    ip: str = '0.0.0.0'

    # Logger settings: disable pynetdicom logger and set log level
    disable_pynetdicom_logger = True
    log_level: int = logging.DEBUG
    log_output = "log.log"
    
    # Blueprint for handling unhandled errors
    unhandled_error_blueprint = error_blueprint

    # Input types for the pipeline
    input = {
        'CT': MyCTInput,
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
        DicomOutput 
            The generated report in DICOM format.
        """

        # Extract PET and CT data from input
        pet = input_data['PET'] # NIfTI PET data
        ct = input_data['CT'] # NIfTI CT data

        # Reference DICOM datasets
        ref_pet_dicom = input_data.datasets['PET'][0]
        ref_ct_dicom = input_data.datasets['CT'][0]
        
        # Get CT series description (metadata)
        ct_desc = ref_ct_dicom.SeriesDescription
        # this is added for validation 
        pt_id = ref_ct_dicom.PatientID
        with open("/home/zuza/validation/pt_processed.txt", "a") as file:
            file.write('\n' + pt_id)
            
        # Perform preprocessing steps on PET and CT data:
        # Swap dimensions for PET and CT (function defined in node_functions)
        pet_swap_nii = node_functions.swap_dims(self, pet, 'PET')
        ct_swap_nii = node_functions.swap_dims(self, ct, 'CT') 
        
        # Skull stripping on CT data
        ct_bet_nii = node_functions.run_skullstripping(self, ct_swap_nii)
        
        # Resampling PET, CT, and skull-stripped CT images to the same resolution
        pet_resampled_nii, ct_resampled_nii, ct_bet_resampled_nii = node_functions.resampling(
            self, pet_swap_nii, ct_swap_nii, ct_bet_nii
        ) 
        
        # Further processing of CT data (e.g., preprocessing)
        ct_bet_preproc_nii = node_functions.process_ct(self, ct_bet_resampled_nii)
        
        # Masking cerebellum cortex in CT data
        cerebellum_nii = node_functions.cerebellum_mask(self, ct_bet_preproc_nii)
        
        # Get prediction data and statistics based on PET and CT data
        prediction_data = node_functions.get_predition(self.logger, ct_bet_preproc_nii, pet_resampled_nii)
        pet_normalized_data, cerebellum_mask_data, patient_values = node_functions.get_statistics(
            self.logger, pet_resampled_nii, cerebellum_nii, prediction_data
            )

        # Generate the report
        report = node_functions.generate_report(
            self, ref_pet_dicom, ct_desc, pet_normalized_data, ct_resampled_nii, 
            prediction_data, cerebellum_mask_data, patient_values
        )
        
        file_path = '/home/zuza/validation/' + str(pt_id) +'.json'
        with open(file_path, 'w') as json_file:
            json.dump(patient_values, json_file, indent=4)

        # Extract keys from the patient_values dictionary to add them to the DICOM report
        keys = list(patient_values.keys())

        # Define the report name and create a blueprint for encoding the DICOM report
        report_name = f"PE2I Report V2.0 {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}"
        blueprint= Blueprint(SECONDARY_IMAGE_REPORT_BLUEPRINT)
        
        # Populate the blueprint with relevant DICOM tags and values
        blueprint[0x0008_103E] = StaticElement(0x0008_103E, 'LO', report_name) # Series Description
        blueprint[0x0010_0010] = CopyElement(0x0010_0010) # Patient's Name 
        blueprint[0x0020_0011] = StaticElement(0x0020_0011, 'IS', str(random.randint(5000,100000))) # Series Number
        blueprint[0x0008_0021] = FunctionalElement(0x00080021, 'DA', get_today) #Series Date
        blueprint[0x0008_0031] = FunctionalElement(0x00080031, 'TM', get_time) #Series Time
        
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
        return DicomOutput([(self.endpoint, encoded_report),(PET_ARCHIVE, encoded_report)], self.ae_title)
       
# Entry point for running the node
if __name__ == "__main__":
   node = Pe2iPetCtNode()
   node.open()
