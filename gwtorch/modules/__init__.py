import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import bilby

import logging
bilby_logger = logging.getLogger("bilby")
bilby_logger.setLevel(logging.WARNING)