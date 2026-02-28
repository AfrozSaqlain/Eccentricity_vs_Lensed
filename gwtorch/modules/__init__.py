import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import logging

try:
    import bilby
except ImportError:
    bilby = None
else:
    bilby_logger = logging.getLogger("bilby")
    bilby_logger.setLevel(logging.WARNING)
