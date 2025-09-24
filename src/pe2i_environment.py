import os
from pathlib import Path

import dotenv
dotenv.load_dotenv()

class Environment:
  """This is the programs hook for getting directories. It's important to use
  this class for paths, rather than hardcoding them because otherwise the
  program isn't portable!
  """

  def __init__(self) -> None:
    self._cwd = Path(os.getcwd())
    static_path_string = os.environ.get("STATIC_PATH","")
    output_path_string = os.environ.get("OUTPUT_PATH","")
    log_path_string = os.environ.get("LOG_PATH", "")
    validation_path_string = os.environ.get("VALIDATION_FILE", "")
    storage_path_string = os.environ.get("STORAGE_PATH", "")


    error_message = ""

    if static_path_string == "":
      error_message += "STATIC_PATH Environment variable not set, add it to .env\n"
    else:
      self._static_path = Path(static_path_string)
      if not self._static_path.exists() and not self._static_path.is_dir():
        error_message += "STATIC_PATH doesn't point to directory!"

    if output_path_string == "":
      error_message += "OUTPUT_PATH Environment variable not set, add it to .env\n"
    else:
      self._output_path = Path(output_path_string)
      if not self._output_path.exists() and not self._output_path.is_dir():
        error_message += "OUTPUT_PATH doesn't point to directory!"

    if log_path_string == "":
      log_path_string = Path(os.getcwd()) / "pipeline.log"

    self._log_path = Path(log_path_string)

    if storage_path_string == "":
      error_message += "STORAGE_PATH Environment variable not set, add it to .env\n"
    else:
      self._storage_path = Path(storage_path_string)
      if not self._storage_path.exists() and not self._storage_path.is_dir():
        error_message += "STORAGE_PATH doesn't point to directory!"

    if validation_path_string == "":
      self._validation_path = Path(self._cwd) / "validation.txt"
    else:
      self._validation_path = Path(validation_path_string)

    if error_message != "":
      raise EnvironmentError(error_message)



  @property
  def STATIC_PATH(self):
    """
    Returns:
        Path: This is path for global static files
    """
    return self._static_path

  @property
  def OUTPUT_PATH(self):
    """
    Returns:
        Path: This is path for temporary files created in the processing of a
              patient
    """
    return self._output_path

  @property
  def LOG_PATH(self):
    """Path to the log file"""
    return self._log_path

  @property
  def STORAGE_PATH(self):
    return self._storage_path

  @property
  def VALIDATION_PATH(self):
    return self._validation_path

  @property
  def CWD(self):
    """Dicomnode changes it's cwd, so if you need the original use this
    instead of os.getcwd
    """
    return self._cwd


  def get_patient_work_directory(self, patientID : str):
    return self.OUTPUT_PATH / patientID







environment = Environment()
