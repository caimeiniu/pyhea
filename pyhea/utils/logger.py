import logging
from pyhea.comm import comm

rank = comm.Get_rank()

class WidthLimitedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, width=47):
        super().__init__(fmt, datefmt)
        self.width = width

    def format(self, record):
        original_message = super().format(record)
        if len(original_message) > self.width and "#nocutoff" not in original_message:
            lines = [
                original_message[i:i + self.width]
                for i in range(0, len(original_message), self.width)
            ]
            return '\n'.join(lines)
        else:
            return original_message.replace("#nocutoff", "")

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
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
formatter = WidthLimitedFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S', width=87)
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

logger = logging.getLogger('PyHEA')