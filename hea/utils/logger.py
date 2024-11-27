import logging
from hea.comm import comm

rank = comm.Get_rank()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("running_hea.log", mode='w') if rank == 0 else logging.NullHandler(),
        logging.StreamHandler() if rank == 0 else logging.NullHandler()
    ]
)

"""
@brief Logger instance used for logging messages.

This logger is configured to handle both file and console outputs, and logs messages
at the DEBUG level and above. The log format includes the timestamp, logger name, 
log level, and message content.

Example usage:
@code
logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.error("This is an error message.")
@endcode
"""
# Set the time format to show only up to seconds
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

logger = logging.getLogger('PyHEA')