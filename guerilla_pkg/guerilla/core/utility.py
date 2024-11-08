import logging

class logger:
    def __init__(self, app_name: str = __name__, log_file_name: str = 'app.log'):
        logging.basicConfig(level=logging.INFO)
        self.logger_ = logging.getLogger(app_name)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file_name)
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        self.logger_.addHandler(c_handler)
        self.logger_.addHandler(f_handler)
    
    def warning(self, msg: str):
        self.logger_.warning(msg)
    
    def error(self, msg: str):
        self.logger_.error(msg)
        
    def info(self, msg: str):
        self.logger_.info(msg)
    
    