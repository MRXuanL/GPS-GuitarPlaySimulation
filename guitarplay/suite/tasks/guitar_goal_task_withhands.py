from typing import Sequence
from guitarplay.modelpy.hands.shadow_hand import _DEFAULT_FOREARM_DOFS
from guitarplay.music.tablature import Tablature
from guitarplay.suite.tasks.base import _ATTACHMENT_YAW, _CONTROL_TIMESTEP, _PHYSICS_TIMESTEP
from guitarplay.suite.tasks.guitar_task_withhands import _FINGER_CLOSE_ENOUGH_TO_KEY, _KEY_CLOSE_ENOUGH_TO_PRESSED, GuitarTaskWithHands
from guitarplay.modelpy.hands import shadow_hand
class GuitarGoalWithHands(GuitarTaskWithHands):
    def __init__(self, table:Tablature=None,
                 gravity_compensation: bool = False, 
                 change_color_on_activation: bool = True, 
                 primitive_fingertip_collisions: bool = False, 
                 reduced_action_space: bool = False, 
                 attachment_yaw: float = _ATTACHMENT_YAW, 
                 forearm_dofs: Sequence[str] = shadow_hand._DEFAULT_FOREARM_DOFS, 
                 physics_timestep: float = _PHYSICS_TIMESTEP, 
                 control_timestep: float = _CONTROL_TIMESTEP,
                 init_buffer_time: float=0,
                 n_step_lookahead: int=1,
                 wrong_press_termination:bool=False,
                 disable_finger_reward:bool=False,
                 disable_key_reward:bool=False,
                 disable_energy_reward:bool=False,
                 finger_bound:float=_FINGER_CLOSE_ENOUGH_TO_KEY,
                 finger_margin:float=_FINGER_CLOSE_ENOUGH_TO_KEY*10,
                 energy_coef:float=0.005,
                 rightKey_weight:float=0.5,
                 key_bound:float=_KEY_CLOSE_ENOUGH_TO_PRESSED,
                 key_margin:float=_KEY_CLOSE_ENOUGH_TO_PRESSED,
                 sigmoid:str="gaussian",
                 operator:str="+",
                 test:bool=False) -> None:
        super().__init__(table, 
                         gravity_compensation, 
                         change_color_on_activation, 
                         primitive_fingertip_collisions, 
                         reduced_action_space, attachment_yaw, 
                         forearm_dofs, physics_timestep, 
                         control_timestep, init_buffer_time, n_step_lookahead, 
                         wrong_press_termination, disable_finger_reward, 
                         disable_key_reward, disable_energy_reward, finger_bound, 
                         finger_margin, energy_coef, rightKey_weight, key_bound, 
                         key_margin, sigmoid, operator, test)
        


        

    


    
    
    