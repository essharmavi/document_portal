# exception/custom_exception.py

import sys
import traceback
from typing import Optional, cast

class DocumentPortalException(Exception):
    """
    A custom exception class for consistent error reporting in the document portal.
    Provides:
    - Normalized error message
    - File name + line number where the error occurred
    - Full traceback string (if available)
    """

    def __init__(self, error_message, error_details: Optional[object] = None):
        """
        :param error_message: Either a string or an Exception instance describing the error
        :param error_details: Can be:
                              - None (default, uses sys.exc_info())
                              - sys (to explicitly capture exc_info)
                              - Exception instance (to extract traceback directly)
        """

        # --- Normalize the error message to always be a string ---
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # --- Initialize traceback variables ---
        exc_type = exc_value = exc_tb = None

        # Case-1: No details provided → use the *current exception context*
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Case-2: User passed `sys` explicitly → call sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):  # e.g. sys module
                exc_info_obj = cast(sys, error_details)  # safely treat as sys
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()

            # Case-3: User passed an actual exception object
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = (
                    type(error_details),   # exception type (e.g., ZeroDivisionError)
                    error_details,         # the exception instance
                    error_details.__traceback__  # traceback attached to the exception
                )

            # Fallback: again use sys.exc_info()
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        # --- Walk down traceback chain to get the *last frame* (most relevant) ---
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # Extract file + line number from the last traceback frame
        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # --- Build full traceback string (like Python default, but stored as a property) ---
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        # Initialize Exception base class with a stringified version
        super().__init__(self.__str__())

    def __str__(self):
        """
        Human-readable message for logs / console
        Example:
        Error in [/path/to/file.py] at line [42] | Message: Division by zero
        Traceback:
        Traceback (most recent call last):
          ...
        """
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        """
        Debug-friendly repr() — concise info (useful in REPL / logging structured data)
        """
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"


# ---------------------------
# Example usage / self-test
# ---------------------------
if __name__ == "__main__":
    # Demo-1: generic exception -> wrapped into DocumentPortalException
    try:
        a = 1 / 0
    except Exception as e:
        raise DocumentPortalException("Division failed", e) from e

    # Demo-2: still supports sys (old pattern)
    # try:
    #     a = int("abc")
    # except Exception as e:
    #     raise DocumentPortalException(e, sys)
