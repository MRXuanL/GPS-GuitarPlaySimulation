from dm_control import mjcf
import guitarplay.modelpy.guitar.guitar_constants as cons
class guitar(object):
    def __init__(self):
        filepath=cons._GUITAR_DIR
        self.model=mjcf.from_path(filepath)


        
