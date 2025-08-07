from pycbc.detector.ground import Detector

# ---------------------- Constants ---------------------- #

DELTA_T = 1.0 / 4096
F_LOWER = 5.0
DETECTOR_NAME = 'E1'
default_detector = Detector(DETECTOR_NAME)
REFERENCE_DISTANCE = 300


DETECTOR_LIST = ['E1', 'E2', 'E3', 'C1']
detector_list = []
for i in DETECTOR_LIST:
    detector_list.append(Detector(i))
