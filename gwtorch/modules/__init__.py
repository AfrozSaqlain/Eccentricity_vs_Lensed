import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import logging

try:
    import bilby
    bilby_logger = logging.getLogger("bilby")
    bilby_logger.setLevel(logging.WARNING)
except ImportError:
    bilby = None
