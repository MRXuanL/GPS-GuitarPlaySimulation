import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from guitarplay.suite.tasks.test_task import traintask
from guitarplay.suite.tasks.guitar_task_withhands import GuitarTaskWithHands
from guitarplay.suite.tasks.guitarvideoenv import GuitarSoundVideoWrapper
from guitarplay.suite.tasks.base import _CONTROL_TIMESTEP
from guitarplay.suite.tasks.guitar_task_withhands import _FINGER_CLOSE_ENOUGH_TO_KEY,_KEY_CLOSE_ENOUGH_TO_PRESSED
from dm_control import viewer
from dm_control import composer
from guitarplay.music.tablature import Tablature as tb

checkaction=[0,1,2,-3,-2,-1]
sequece=['goal','last_state',
         'state',
        #  'lh_shadow_hand/position',
         'lh_shadow_hand/joints_pos',
        #  'lh_shadow_hand/joints_vel',
        #  'lh_shadow_hand/joints_torque',
        #  'lh_shadow_hand/fingertip_positions'
        ]
MAX_VEL=60
task=traintask().task
joint_range=np.array([
    [-1.00000e-04,  1.00000e-04],
    [ 0.00000e+00 , 7.89000e-01],
    [-3.49000e-01 , 3.49000e-01],
    [ 0.00000e+00 , 1.57000e+00],
    [ 0.00000e+00 , 1.57080e+00],
    [ 0.00000e+00 , 1.57080e+00],
    [-3.49000e-01 , 3.49000e-01],
    [ 0.00000e+00 , 1.57000e+00],
    [ 0.00000e+00 , 1.57080e+00],
    [ 0.00000e+00 , 1.57080e+00],
    [-3.49000e-01 , 3.49000e-01],
    [ 0.00000e+00 , 1.57000e+00],
    [ 0.00000e+00 , 1.57080e+00],
    [ 0.00000e+00 , 1.57080e+00],
    # [-0.5 ,      0.785], 
    [-3.49000e-01 , 3.49000e-01],
    [ 0.00000e+00 , 1.57000e+00],
    [ 0.00000e+00 , 1.57080e+00],
    [ 0.00000e+00 , 1.57080e+00],
    [-1.00000e-04 , 1.00000e-04],
    [-1.00000e-04 , 1.00000e-04],
    [ 0.00000e+00 , 6.98132e-01],
    [-2.30000e-01,  9.00000e-02],
    [-0.01,  0.10],
    [-1.00000e-01,  2.00000e-02],
])
joint_center=(joint_range[:,1]+joint_range[:,0])/2.0
joint_power=(joint_range[:,1]-joint_range[:,0])/2.0
testMode=2
class GuitarPlayEnv(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"], "render_fps": 4}
    def __init__(self,render_mode=None,
                change_color_on_activate: bool=False,
                disable_finger_reward:bool=False,
                disable_key_reward:bool=False,
                disable_energy_reward:bool=False,
                wrong_press_termination:bool=False,
                init_buffer_time:int=1,
                n_step_lookahead:int=10,
                finger_bound:float=_FINGER_CLOSE_ENOUGH_TO_KEY,
                finger_margin:float=_FINGER_CLOSE_ENOUGH_TO_KEY*10,
                energy_coef:float=0.005,
                key_bound:float=_KEY_CLOSE_ENOUGH_TO_PRESSED,
                key_margin:float=_KEY_CLOSE_ENOUGH_TO_PRESSED,
                sigmoid:str="gaussian",
                rightKey_weight:float=0.5,
                reduced_action_space:bool=True,
                operator:str='+',
                normalization:bool=False,
                table:int=0,
                test:bool=False,
                freedom:bool=False) -> None:
        self.DMCtask=GuitarTaskWithHands(change_color_on_activation=change_color_on_activate,
                                        disable_finger_reward=disable_finger_reward,
                                        disable_key_reward=disable_key_reward,
                                        disable_energy_reward=disable_energy_reward,
                                        wrong_press_termination=wrong_press_termination,
                                        init_buffer_time=init_buffer_time,
                                        n_step_lookahead=n_step_lookahead,
                                        finger_bound=finger_bound,
                                        finger_margin=finger_margin,
                                        energy_coef=energy_coef,
                                        key_bound=key_bound,
                                        key_margin=key_margin,
                                        sigmoid=sigmoid,
                                        rightKey_weight=rightKey_weight,
                                        reduced_action_space=reduced_action_space,
                                        operator=operator,
                                        table=task[table],
                                        test=test,
                                        freedom=freedom)
        self.DMCenv=composer.Environment(self.DMCtask, random_state=np.random.RandomState(42))
        if(test==1):
            self.DMCenv=GuitarSoundVideoWrapper(self.DMCenv,record_every=1) 
        self.DMCob_spc=self.DMCenv.observation_spec()
        self.DMCac_spc=self.DMCenv.action_spec()
        self.ac_center=(self.DMCac_spc.minimum+self.DMCac_spc.maximum)/2
        self.ac_power=(self.DMCac_spc.maximum-self.DMCac_spc.minimum)/2
        self.init_buffer_step=int(init_buffer_time/_CONTROL_TIMESTEP)
        self.f1s=[]
        self.rewards=[]
        self.max_vel=-1
        self.cnt=0
        self.check=0
        self.flag=1
        self.normalization=normalization
        ob_shape=0
        for i in range(len(sequece)):
            ob_shape+=self.DMCob_spc[sequece[i]].shape[1]
            
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(ob_shape,),dtype=np.float64)
        
        self.action_space=spaces.Box(low=-1,high=1,shape=self.DMCac_spc.shape,dtype=np.float64)
        self.render_mode=render_mode
    def _get_obs(self,timestep):
        obs=np.array([])
        for name in sequece:
            obs_part=timestep.observation[name]
            obs_part=obs_part.flatten()
            if name=='last_state':
                obs_part[:-1]=(obs_part[:-1]-self.ac_center)/self.ac_power
            if self.normalization:
                if name=='lh_shadow_hand/joints_pos':
                    # print('before {}'.format(obs_part))
                    obs_part=(obs_part-joint_center)/joint_power
                    # print('{}'.format(obs_part))
                if name=='lh_shadow_hand/joints_vel':
                    self.max_vel=max(np.max(obs_part),self.max_vel)
                    # print(self.max_vel)
                    obs_part/=MAX_VEL
            if self.normalization:
                obs_part.clip(-1,1,out=obs_part)
            obs=np.concatenate((obs,obs_part))
        return obs
    def _get_info(self,timestep):
        return{
            "f1":timestep.observation['f1'].flatten(),
            "reward":timestep.observation['reward'].flatten()
        }
        
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.f1s=[]
        self.rewards=[]
        timestep=self.DMCenv.reset()
        observation=self._get_obs(timestep)
        info=self._get_info(timestep)
        self.f1s.append(info['f1'])
        self.rewards.append(info['reward'])
        return observation,info
    
    def step(self,action):
        rescaleAction=action*self.ac_power+self.ac_center
        timestep=self.DMCenv.step(rescaleAction)
        reward=timestep.reward
        terminated=timestep.last()
        observation=self._get_obs(timestep)
        info=self._get_info(timestep)
        self.f1s.append(info['f1'])
        self.rewards.append(info['reward'])
        if terminated:
            f1s=np.array(self.f1s)[self.init_buffer_step:,:]
            rewards=np.array(self.rewards)[self.init_buffer_step:,:]
            rewards=np.sum(rewards,0)
            case=np.sum(f1s,0)
            tp=case[0]
            fp=case[1]
            fn=case[2]
            precision,recall,F1=self.compute_f1(tp,fp,fn)
            info={'f1':F1,'precision':precision,'recall':recall,'reward':rewards}
            
        return observation, reward, terminated, False, info
    
    def compute_f1(self,tp,fp,fn):
      
        if(tp+fn!=0 and tp+fp==0):
            precision=1
            recall=0
            return 1,0,0
        if(tp+fn==0 and tp+fp!=0):
            precision=0
            recall=1
            return 0,1,0
        if(tp+fn==0 and tp+fp==0):
            precision=1
            recall=1
            return 1,1,1

        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        if(tp==0):
            return 0,0,0
        F1=2*precision*recall/(precision+recall)
        return precision,recall,F1
    def render(self):
        pass
    
            
    def viewer(self,model):
        if self.render_mode=="human":
            def policy(timestep):
                if testMode==1:
                    action=np.ones(self.action_space.shape)*(-1)
                    self.cnt+=1
                    if self.cnt%40==0:
                        self.cnt=0
                        # self.flag=-self.flag
                        self.check=(self.check+1)%6
                    action[checkaction[self.check]]=self.flag
                elif testMode==0:
                    action=self.action_space.sample()
                else:
                    obs=self._get_obs(timestep)
                    action, _states = model.predict(obs, deterministic=True)
                rescaleAction=action*self.ac_power+self.ac_center
                return rescaleAction
            viewer.launch(self.DMCenv,policy=policy)
    
    def close(self):
        pass
            
