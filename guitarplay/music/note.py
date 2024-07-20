class Note:
    def __init__(self,name,time,duration) -> None:
        """音符

        Args:
            name (string): 音符在吉他弦上的位置
            time (float): 音符开始的时间
            duration (float): 音符持续的时间
        """
        
        self.name=name
        self.time=time
        self.duration=duration
        self.finger=-1