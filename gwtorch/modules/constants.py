from pycbc.detector.ground import Detector

# ---------------------- Constants ---------------------- #

DELTA_T = 1.0 / 4096
F_LOWER = 5.0
DETECTOR_NAME = 'H1'
detector = Detector(DETECTOR_NAME)
REFERENCE_DISTANCE = 300