"""
logFileHandler.py
Minimal logger that writes to console and a shared log file.
Accepts multiple arguments like print().
"""
import datetime

class Logger:
    def __init__(self, log_file_path, script_name):
        self.script_name = script_name
        self.log_file = open(log_file_path, 'a', encoding='utf-8')

    def log(self, *args, sep=' ', end='\n'):
        """
        Works exactly like print(): accepts multiple arguments,
        writes to console and appends to log file with timestamp and script name.
        """
        message = sep.join(str(a) for a in args)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] [{self.script_name}] {message}"

        # Print to console as-is
        print(*args, sep=sep, end=end)

        # Write to log file
        self.log_file.write(full_message + end)
        self.log_file.flush()

    def close(self):
        self.log_file.close()