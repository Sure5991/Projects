import sys
from src.mlproject.logger import logging


def error_message_details(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_msg = f'error occured in script name [{filename}] at line no - {exc_tb.tb_lineno} : {str(error)}'
    return error_msg

# Creating class for 
class customexception(Exception):
    def __init__(self,error_msg,error_details:sys):
        super().__init__(error_msg)
        self.error_msg = error_message_details(error_msg,error_details)
        self.log_error(self.error_msg)
    
    def log_error(self,message):
        logging.error(message)

    def __str__(self):
        return self.error_msg