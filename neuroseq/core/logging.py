from neuroseq.common.config import *


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


# Add additional log levels
addLoggingLevel('ESSENS', 25)
addLoggingLevel('DETAIL', 15)
addLoggingLevel('TRACE', 5)

# Create logger
log = logging.getLogger('shtm')
log.setLevel(logging.DEBUG)

# Check if log-file folder exists
if not os.path.exists(os.path.dirname(Log.FILE)):
    os.makedirs(os.path.dirname(Log.FILE), exist_ok=True)

# Create handler for file
fh = logging.FileHandler(Log.FILE, mode='w', encoding='utf-8')
fh.setLevel(Log.LEVEL_FILE)
fh.setFormatter(logging.Formatter(Log.FORMAT_FILE, datefmt=Log.DATEFMT))
log.addHandler(fh)

# Create handler for stream (stdout)
ch = logging.StreamHandler()
ch.setLevel(Log.LEVEL_SCREEN)

try:
    import colorlog
except ModuleNotFoundError as e:
    ch.setFormatter(logging.Formatter(Log.FORMAT_SCREEN_NO_COLOR, datefmt=Log.DATEFMT))
    log.addHandler(ch)
    log.warning("Could not load module 'colorlog'. Continuing logging without colored formatter.")
else:
    cf = colorlog.ColoredFormatter(Log.FORMAT_SCREEN_COLOR, datefmt=Log.DATEFMT)
    cf.log_colors['ESSENS'] = 'green'
    cf.log_colors['DETAIL'] = 'cyan'
    cf.log_colors['TRACE'] = 'grey'
    ch.setFormatter(cf)
    log.addHandler(ch)
