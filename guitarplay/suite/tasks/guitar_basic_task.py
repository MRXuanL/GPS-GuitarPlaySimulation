from typing import Sequence
from guitarplay.modelpy.hands import shadow_hand
from guitarplay.music.goal import Goal
from guitarplay.music.tablature import Tablature
from guitarplay.suite.tasks.base import _ATTACHMENT_YAW, _CONTROL_TIMESTEP, _PHYSICS_TIMESTEP, GuitarOnlyTask,GuitarTask
import numpy as np
class GuitarBasicTask(GuitarTask):
    def __init__(self, 
                 table:Tablature=None,
                 gravity_compensation: bool = False, 
                 change_color_on_activation: bool = True, 
                 primitive_fingertip_collisions: bool = False, 
                 reduced_action_space: bool = False, 
                 attachment_yaw: float = _ATTACHMENT_YAW, 
                 forearm_dofs: Sequence[str] = shadow_hand._DEFAULT_FOREARM_DOFS, 
                 physics_timestep: float = _PHYSICS_TIMESTEP, 
                 control_timestep: float = _CONTROL_TIMESTEP,
                 key_bound:float=0.01) -> None:
        super().__init__(gravity_compensation, 
                         change_color_on_activation, 
                         primitive_fingertip_collisions, 
                         reduced_action_space, attachment_yaw, 
                         forearm_dofs,
                         physics_timestep, 
                         key_bound,
                         control_timestep)
        self._tick_id=0
        self._goal=Goal(0,control_timestep,table)
        self._goalstate=self._goal.goalstate

        
    def after_substep(self, physics, random_state):
        pass
        
    
    def initialize_episode(self, physics, random_state):
        self._tick_id=0
        
        pass
        
    def after_step(self, physics, random_state):
        pass
    
    def should_terminate_episode(self, physics):
        if(self._tick_id>=self._goal.totaltick-1):
            return True
        return False
    
        
