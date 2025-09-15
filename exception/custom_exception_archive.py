import sys
import traceback
from logger.custom_logger import CustomLogger

logger = CustomLogger().get_logger(__file__) # Initialize the custom logger

class DocumentPortalException(Exception):
    """Base class for all exceptions in the Document Portal application."""
    def __init__(self, error_exception: str, error_details: sys):
        _,_,exc_tb = error_details.exc_info() # Get the traceback details
        self.file_name = exc_tb.tb_frame.f_code.co_filename # Get the file name where the exception occurred`
        self.line_number = exc_tb.tb_lineno # Get the line number where the exception occurred
        self.error_message = str(error_exception) # Convert the exception to a string for logging
        self.traceback_str = ''.join(traceback.format_exception(*error_details.exc_info())) # Format the traceback for better readability

    def __str__(self):
        return f"Error occurred in file {self.file_name} at line {self.line_number}: {self.error_message}\nTraceback:\n{self.traceback_str}"    


# if __name__ == "__main__":

#     try:
#         a = 1/0
#     except Exception as e:
#         app_exception = DocumentPortalException(e, sys) # Create an instance of DocumentPortalException with the caught exception
#         logger.error(app_exception) #  Log the exception using the custom logger

