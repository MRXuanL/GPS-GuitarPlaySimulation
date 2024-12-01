"""A wrapper for rendering videos with sound."""
import shutil
import subprocess
import wave
import numpy as np
from pathlib import Path
from guitarplay.suite.tasks.base import _FRAME_RATE
import dm_env
from dm_env_wrappers import DmControlVideoWrapper
from guitarplay.music.audio import NOTEWAVE,FS
def addwave(totaltick,notes):
    waveform=np.zeros((int(totaltick*FS*1.0/_FRAME_RATE)+1),dtype=np.float32)
    for tick,note in notes:
        wavef=NOTEWAVE[note]
        starttick=int(tick*FS*1.0/_FRAME_RATE)
        for i,value in enumerate(wavef):
            if(i+starttick<len(waveform)):
                waveform[i+starttick]+=value
    return waveform




class GuitarSoundVideoWrapper(DmControlVideoWrapper):
    """Video rendering with sound from the piano keys."""

    def __init__(
        self,
        environment: dm_env.Environment,
        **kwargs,
    ) -> None:
        super().__init__(environment, **kwargs)
        self._height=1080
        self._width=1920
        


    def _write_frames(self) -> None:
        super()._write_frames()


        # Exit if there are no MIDI events or if all events are sustain events.
        # Sustain only events cause white noise in the audio (which has shattered my
        # eardrums on more than one occasion).


        # Synthesize waveform.
        notes= self.environment.task.notes
        totaltick= self.environment.task.totaltick
        waveform=addwave(totaltick,notes)
        pcm_waveform = np.int16(waveform * 32767)
        # Save waveform as mp3.
        waveform_name = self._record_dir / f"{self._counter:05d}.mp3"
        wf = wave.open(str(waveform_name), "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FS * self._playback_speed)
        wf.writeframes(pcm_waveform)  # type: ignore
        wf.close()

        # Make a copy of the MP4 so that FFMPEG can overwrite it.
        filename = self._record_dir / f"{self._counter:05d}.mp4"
        temp_filename = self._record_dir / "temp.mp4"
        shutil.copyfile(filename, temp_filename)
        filename.unlink()

        # Add the sound to the MP4 using FFMPEG, suppressing the output.
        # Reference: https://stackoverflow.com/a/11783474
        # ret = subprocess.run(
        #     [
        #         "ffmpeg",
        #         "-nostdin",
        #         "-y",
        #         "-i",
        #         str(temp_filename),
        #         "-i",
        #         str(waveform_name),
        #         "-map",
        #         "0",
        #         "-map",
        #         "1:a",
        #         "-c:v",
        #         "copy",
        #         "-shortest",
        #         str(filename),
        #     ],
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.STDOUT,
        #     check=True,
        # )
        # if ret.returncode != 0:
        #     print(f"FFMPEG failed to add sound to video {temp_filename}.")

        # # Remove temporary files.
        # temp_filename.unlink()
        # waveform_name.unlink()

    def _render_frame(self, observation) -> np.ndarray:
        del observation  # Unused.
        physics = self.environment.physics
        if self._camera_id is not None:
            return physics.render(
                camera_id=self._camera_id,
                height=self._height,
                width=self._width,
            )
        # If no camera_id is specified, render all cameras in a grid.
        height = self._height
        width = self._width
        frame = np.zeros(( height, width, 3), dtype=np.uint8)
        num_cameras = physics.model.ncam
        for camera_id in range(num_cameras):
            if(camera_id==0):
                subframe = physics.render(
                    camera_id=camera_id, height=height, width=width
                )
                frame=subframe
            elif(camera_id==1):
                subframe=physics.render(
                    camera_id=camera_id, height=int(height/4), width=int(width/4)
                )
                frame[height-int(height/4):,:int(width/4)]=subframe
            elif(camera_id==2):
                subframe=physics.render(
                    camera_id=camera_id, height=int(height/4), width=int(width/4)
                )
                frame[height-int(height/4):,width-int(width/4):]=subframe
        return frame

    def __del__(self) -> None:
        # self._synth.stop()
        pass
