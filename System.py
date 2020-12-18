from Tracking import Tracking
from Frame import Frame


class System:
    """SLAM object containing the KeyFrames, """ """, strVocFile, strSettingsFile):"""
    def __init__(self, settings_sys):
        self.Tracker = Tracking(settings={'size': [640, 480]})
        # self.strVocFile = strVocFile
        # self.strSettingsFile = strSettingsFile
        self.frameCounter = 0

    def track(self, imRGB, imD, timestamp):
        frame_current = Frame(imRGB, imD, timestamp)
        frame_current.set_id(self.frameCounter)
        self.frameCounter = self.frameCounter + 1
