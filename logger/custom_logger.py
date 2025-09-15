from datetime import datetime
import logging
import os
import structlog

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #Create a log file with the current date and time
    
class CustomLogger():
    
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'logs') # Ensure the logs directory exists
        os.makedirs(self.path, exist_ok=True) # Create the logs directory if it doesn't exist

        self.log_file_path = os.path.join(self.path, LOG_FILE) # Full path for the log file


    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name) # Get the base name of the file for the logger
        
        file_formatter = logging.Formatter(
            '%(message)s')

        console_formatter = logging.Formatter(
            '%(message)s')

        
        file_handler = logging.FileHandler(self.log_file_path) # Create a file handler for logging to the file
        file_handler.setFormatter(file_formatter) # Set the formatter for the file handler
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler() # Create a console handler for logging to the console
        console_handler.setFormatter(console_formatter) # Set the formatter for the console handler
        console_handler.setLevel(logging.INFO)

        logging.basicConfig(
            level=logging.INFO, # Set the logging level to INFO
            handlers=[file_handler, console_handler], # Add both file and console handlers
            format='%(message)s' # Set the log message format
        )

        structlog.configure(
            processors = [
                structlog.processors.TimeStamper(fmt='iso', utc = True, key = "timestamp"), # Add a timestamp to the log messages
                structlog.processors.add_log_level, # Add the log level to the log messages
                structlog.processors.EventRenamer(to = "event"), # Rename the event key to 'event'
                structlog.processors.JSONRenderer() # Render the log messages as JSON
            ],
            logger_factory=structlog.stdlib.LoggerFactory(), # Use the standard library logger factory
            cache_logger_on_first_use=True, # Cache the logger on first use
        )

        return structlog.get_logger(logger_name) # Return a structlog logger with the specified name

# if __name__ == "__main__":
#     logger = CustomLogger()
#     logger_instance = logger.get_logger(__file__) 
#     logger_instance.info("User uploaded a file", user_id=123, filename="report.pdf")
#     logger_instance.error("Failed to process PDF", error="File not found", user_id=123)
