import numpy as np
import pyaudio
import math
from guitarplay.suite.tasks.base import _PHYSICS_TIMESTEP


import time
FS = 44100
LEN=int(_PHYSICS_TIMESTEP*FS)
class Musicplayer:
    def __init__(self) -> None:
        self.end=pyaudio.paComplete

    def playaudio(self,waveform: np.ndarray, sampling_rate: int = FS, chunk: int = 2500)-> None:
        self.end=pyaudio.paContinue
        """Play a waveform using PyAudio."""
        if waveform.dtype != np.float32:
            raise ValueError("waveform must be an np.float32 array.")

        # An iterator that yields chunks of audio data.
        def chunkifier():
            for i in range(0, len(waveform), chunk):
                yield waveform[i : i + chunk]

        audio_generator = chunkifier()

        def callback(in_data, frame_count, time_info, status):
            del in_data, frame_count, time_info, status
            return (next(audio_generator), self.end)

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sampling_rate,
            output=True,
            frames_per_buffer=chunk,
            stream_callback=callback,
        )

    
        self.stream.start_stream()
 
    def stop(self):
        self.end=pyaudio.paComplete
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def playNote(self,note,t=1):
        if self.end==pyaudio.paContinue:
            self.stop()
        self.playaudio(NOTEWAVE[note])

def gwave(t,f,r=0.5):
    N=math.floor(FS*t+0.5) #总长度
    p=FS/f
    l=math.ceil(p)
    nump=math.floor(N/p)-1
    part=np.random.rand(l)-0.5
    part=part-np.mean(part)
    y=np.zeros(N)
    for i in range(0,nump):
        
        b=(np.concatenate((np.array([part[-1]]),part[:-1])))*(1-r)
        a=part*r
        part=(a+b)
        pos=math.floor(i*p)
        y[pos:pos+l]=part
        
    y=y*np.linspace(1,0,len(y))
    return y.astype(np.float32)
    
    
    

def note_to_frequency(note):
    A4="A4"
    offset=[9,11,0,2,4,5,7]
    bias=0
    if(note[0]=='#' or note[0]=='b'):
        octave=int(note[2])-4
        stdbias=offset[ord(note[1])-ord('A')]-offset[0]
        if(note[0]=='#'):
            bias=1
        elif(note[0]=='b'):
            bias=-1
    else:
        stdbias=offset[ord(note[0])-ord('A')]-offset[0]
        octave=int(note[1])-4
    return 440 * (2 ** ((octave*12+bias+stdbias) / 12))

notes=['E2','F2','#F2','G2','#G2','A2','#A2','B2',
       'C3','#C3','D3','#D3','E3','F3','#F3','G3','#G3','A3','#A3','B3',
       'C4','#C4','D4','#D4','E4','F4','#F4','G4','#G4','A4','#A4','B4',
       'C5','#C5','D5','#D5','E5','F5','#F5','G5','#G5','A5','#A5','B5',
       'C6','#C6','D6','#D6','E6','F6','#F6','G6','#G6','A6','#A6','B6',]

NOTEWAVE={
    note : gwave(1,freq) for note,freq in zip(notes,[note_to_frequency(note) for note in notes])
}





def example():
    player=[Musicplayer() for i in range(6)]

    #和声
    player[0].playNote('C3')
    player[1].playNote('E3')
    player[2].playNote('G3')

    time.sleep(1)

    #测试打断效果
    player[0].playNote('C3')
    time.sleep(0.5)
    # player[0].stop()
    player[0].playNote('C4')

if __name__ == "__main__":
    example()
