import logging
import os
from datetime import datetime

#creating path for logfiles
log_file = f"{datetime.now().strftime('%d_%m_%Y')}.log"
log_path = os.path.join(os.getcwd(),'logs')
os.makedirs(log_path,exist_ok= True)
log_file_path = os.path.join(log_path,log_file)

# logging function for log details
logging.basicConfig(
    filename= log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s -%(message)s",
    level= logging.INFO
)