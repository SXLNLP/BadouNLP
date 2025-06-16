from loguru import logger
import os
import sys

LOG_FILE_NAME = "app.log"
ROTATION_TIME = "02:00"


# TODO: 适配多进程
# TODO: exception处理
class Logger:
    def __init__(self, name="inori", log_dir="static", log_file=LOG_FILE_NAME, debug=False):
        current_folder = os.path.join(os.getcwd(), log_dir, str(name))
        if not os.path.exists(log_dir):
            os.makedirs(current_folder)
        log_file_path = os.path.join(current_folder, log_file)
        # print(f'log_file: {log_file_path}')
        # Remove default loguru response_handler
        logger.remove()

        # Add console response_handler with a specific log level
        level = "DEBUG" if debug else "INFO"
        logger.add(sys.stdout, level=level)
        # logger.configure(extra={
        #     "thread": threading.current_thread().name
        # })
        # logger.add(sys.stdout, format="on <light-blue><u>{extra[thread]}</u></light-blue>", level=level)
        # Add file response_handler with a specific log level and timed rotation
        logger.add(log_file_path, rotation="2 MB", level="DEBUG")
        # logger.add('client.log', rotation="00:00", level="DEBUG")
        self.logger = logger
        # self.logger = logger.bind(thread=threading.current_thread().name)


if __name__ == "__main__":
    log = Logger(debug=True).logger

    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
