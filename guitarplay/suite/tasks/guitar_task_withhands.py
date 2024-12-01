from typing import Sequence
from guitarplay.modelpy.hands import shadow_hand
from guitarplay.music.goal import Goal
from guitarplay.music.tablature import Tablature
from guitarplay.suite.tasks.base import _ATTACHMENT_YAW, _CONTROL_TIMESTEP, _PHYSICS_TIMESTEP,_ANIMATION_NUM
from guitarplay.suite.tasks.guitar_basic_task import GuitarBasicTask
from guitarplay.modelpy.guitar import guitar_constants as cons
from dm_control.utils.rewards import tolerance
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from mujoco_utils import spec_utils,mjcf_utils
from dm_control import mjcf
from dm_env import specs
from typing import List, Sequence, Tuple
from dm_control.composer.observation import observable
from guitarplay.music.audio import Musicplayer
import numpy as np
from guitarplay.modelpy.hands.shadow_hand import INITIAL_NAME,INITIAL_POS
INITIAL_RIGHT_NAME=['rh_WRJ2','rh_WRJ1','rh_THJ5','rh_THJ4','rh_THJ3',
              'rh_THJ2','rh_THJ1','rh_FFJ4','rh_FFJ3','rh_FFJ0',
              'rh_MFJ4','rh_MFJ3','rh_MFJ0','rh_RFJ4','rh_RFJ3',
              'rh_RFJ0','rh_LFJ5','rh_LFJ4','rh_LFJ3','rh_LFJ0',]
INITIAL_RIGHT_POS=[-2.5e-05,-0.297,0.597,0.659,0.0503,0.133,0,-0.0803,0.297,1.62,-0.255,0.27,1.7,0.342,0.581,1.59,0.546,-0.311,0.563,1.59]
STANDNAME=['E4','B3','G3','D3','A2','E2']
STANDPOS=[4,11,7,2,9,4]
STANDPITCH=[4,3,3,3,2,2]
NAMEDIR=['C','#C','D','#D','E','F','#F','G','#G','A','#A','B']
LENGTH=[0.018077354260089686, 0.017064049649098113, 0.01610754462392445, 
        0.015204655351282946, 0.014352376463879194, 0.013547871056711078,
        0.012788461244115167, 0.012071619246126647, 0.011394958974482776, 
        0.010756228090262889, 0.010153300506727974, 0.009584169312404658, 
        0.009046940090857311, 0.008539824614912395, 0.008061134894345557, 
        0.007609277557218563, 0.007182748546163711, 0.006780128111961709, 
        0.006400076087748608, 0.006041327428121444]
_energy_penalty_coef=0.005
_KEY_CLOSE_ENOUGH_TO_PRESSED=cons.ACTIVATION_THRESHOLD
_FINGER_CLOSE_ENOUGH_TO_KEY=0.01
FINGER_JOINT=[
    ['rh_shadow_hand/rh_THJ6','rh_shadow_hand/rh_THJ4','rh_shadow_hand/rh_THJ3','rh_shadow_hand/rh_THJ2'],
    ['rh_shadow_hand/rh_FFJ4','rh_shadow_hand/rh_FFJ3','rh_shadow_hand/rh_FFJ2','rh_shadow_hand/rh_FFJ1'],
    ['rh_shadow_hand/rh_MFJ4','rh_shadow_hand/rh_MFJ3','rh_shadow_hand/rh_MFJ2','rh_shadow_hand/rh_MFJ1'],
    ['rh_shadow_hand/rh_RFJ4','rh_shadow_hand/rh_RFJ3','rh_shadow_hand/rh_RFJ2','rh_shadow_hand/rh_RFJ1'],
    ['rh_shadow_hand/rh_LFJ4','rh_shadow_hand/rh_LFJ3','rh_shadow_hand/rh_LFJ2','rh_shadow_hand/rh_LFJ1'],
]
#目标可能出现的位置集合
FINGERPOSRANGE=[
    [-0.5,0.1],
    [-0.1,0.1],
    [0.63,0.755]
]


def norm_from_range(k,range):
    center=(range[0]+range[1])/2
    power=(range[1]-range[0])/2
    return (k-center)/power
fix_joints=['rh_WRJ2','rh_WRJ1']
fix_pos=[-2.5e-05,-0.297]
rotate_anglex=20
rotate_anglex=rotate_anglex/180*np.pi
rotate_angley=-20
rotate_angley=rotate_angley/180*np.pi
rotate_anglez=-90
rotate_anglez=rotate_anglez/180*np.pi

rowx=200
rowy=400
last_pos=[]

#右手是否等待
wait=0

#goalstate or stringstate
STRINGSTATE=1
class GuitarTaskWithHands(GuitarBasicTask):
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
                 test:bool=False,
                 freedom:bool=False,
                 ) -> None:
        super().__init__(table, gravity_compensation, 
                         change_color_on_activation, 
                         primitive_fingertip_collisions, 
                         reduced_action_space, 
                         attachment_yaw, 
                         forearm_dofs, 
                         physics_timestep, 
                         key_bound,
                         control_timestep)
        self._last_trajactory=[]
        self._last_finger_qpos=[[] for i in range(5)]
        self._fingers_joint=[[] for i in range(5)]
        self._test=test
        #每根弦作为音乐播放器
        self._string_player=[Musicplayer() for i in range(6)]
        self._string_max_pos=[0 for i in range(6)]
        self._freedom=freedom
        self._notes=[]
        
        
        self._finger_bound=finger_bound
        self._finger_margin=finger_margin
        self._key_bound=key_bound
        self._key_margin=key_margin
        self._energy_coef=energy_coef
        self._sigmoid=sigmoid
        self._rightKey_weight=rightKey_weight
        self._init_buffer_time=init_buffer_time
        self._tick_id=0
        self._operator=operator
        self._goal=Goal(self._init_buffer_time,control_timestep,table)
        self._goalstate=self._goal.goalstate
        self._pluckstate=self._goal.pluckstate
        self._stringstate=self._goal.stringstate
        self._totaltick=self._goal.totaltick
        self._wrong_press_termination=wrong_press_termination
        self._n_steps_lookahead=n_step_lookahead
        self._disable_finger_reward=disable_finger_reward
        self._disable_key_reward=disable_key_reward
        self._disable_energy_reward=disable_energy_reward
        
        self._reset_quantities_at_episode_init()
        self._add_observables()
        
        print(self._goal.totaltick)
    
    @property
    def totaltick(self):
        return self._totaltick
    
    @property
    def notes(self):
        return self._notes

    
    @property
    def task_observables(self):
        return self._task_observables
    
    
    def _compute_key_press_reward(self, physics: mjcf.Physics) -> float:
        # print(physics.named.model.jnt_range)
        # print(physics.model.jnt_range[22:])
        # joints=mjcf_utils.safe_find_all(self._left_hand.mjcf_model,'joint')
        # actuators=mjcf_utils.safe_find_all(self._left_hand.mjcf_model,"actuator")
        # print('task')
        # for joint in joints:
        #     print(joint.name)
        """Reward for pressing the right keys at the right time."""
        del physics  # Unused.
        # print(self._goal_current)
        # print(self._goal_current[:,0])
        on = np.flatnonzero(self._goal_current[:,0])
        rew = 0.0

        if on.size > 0:
            actual = np.array(self.guitar.state/cons.MAX_FORCE)
            rews = tolerance(
                actual[on],
                bounds=(self._key_bound,1),
                margin=(self._key_margin),
                sigmoid="gaussian",
                value_at_margin=0.1,
            )
            rew +=0.5 * rews.mean()
            
            
        # If there are any false positives, the remaining 0.5 reward is lost.
        off = np.flatnonzero(1 - self._goal_current[:,0])
        offnew=[]
        if self._freedom:

            #on 与 off的关系 ：如果off中的某些激活了，且off在on中的某个弦的右侧 
            #维持一个数组表示每根弦的最外侧的位置：on, self._pluckstate[]
            #当前拨动的弦
            #可以在谱子分配阶段就处理好
            # stringstate=[int(cons.NUM_KEYS/6) for i in range(6)]

            # for i,pluckon in enumerate(self._pluckstate[self._tick_id-1]):
            #     if pluckon:
            #         stringstate[i]=0


            # for key in on:
            #     stringstate[key%6]=int(key/6)+1
            
            for key in off:

                if self._string_current[key%6][0]<(int(key/6)+1):
                    offnew.append(key)

            off=offnew
            
   
        rew += 0.5 * (1 - float(self.guitar.activation[off].any()))
        return rew

    def _compute_energy_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing energy."""
        rew = 0.0
        flag=1
        # for hand in [self.right_hand, self.left_hand]:
        on = np.flatnonzero(self._goal_current[:,0])
        if on.size==0:
            flag=2
        for hand in [self.left_hand]:
            power = hand.observables.actuators_power(physics).copy()
            rew -= flag*self._energy_coef * np.sum(power)
        return rew

    def _create_trajactory(self,string,physics: mjcf.Physics,ticknum,endtick):
        site=self.guitar.mjcf_model.find('site','test'+str(string))
        
        sitepos=physics.bind(site).xpos.copy()
        
        trajactory=[]
        #六弦、五弦、四弦
        if(string>=4):
            fingerpos=physics.bind(self.right_hand.fingertip_sites[0]).xpos.copy()
            tz=sitepos[2]+0.002
            ty=sitepos[1]-0.002
            #从拇指当前位置到达指定琴弦的上前方,花费1/4总tick
            ticknum1=0
            flag=tz>fingerpos[2]
            afz=sitepos[2]-0.007
            afy=sitepos[1]+0.01
            for i in range(ticknum1):
                z=fingerpos[2]+(tz-fingerpos[2])*i/ticknum1
                y=fingerpos[1]+(ty-fingerpos[1])*i/ticknum1
                x=sitepos[0]+0.00005*i*flag
                trajactory.append([x,y,z])
            #拨弦并折返到最上方，花费3/4总tick
            ticknum2=ticknum-ticknum1
            lastx=sitepos[0]
            for i in range(ticknum2):
                z=tz+(afz-tz)*(i)/ticknum2
                y=ty+(afy-ty)*(i)/ticknum2
                x=lastx+0.00005*(i)
                trajactory.append([x,y,z])
        #三弦、二弦、一弦
        else:
            oz=sitepos[2]-0.0025
            oy=sitepos[1]+0.0025*1.73205
            ox=sitepos[0]
            total=2*np.pi
            for i in range(ticknum):
                y=np.cos(-i*total/ticknum+np.pi)/rowx
                z=np.sin(-i*total/ticknum+np.pi)/rowy
                x=0
                rz=z
                ry=y
                y=np.cos(rotate_anglex)*ry-np.sin(rotate_anglex)*rz
                z=np.sin(rotate_anglex)*ry+np.cos(rotate_anglex)*rz
                rx=x
                ry=y
                y=np.cos(rotate_anglez)*rx-np.sin(rotate_anglez)*ry
                x=np.sin(rotate_anglez)*rx+np.cos(rotate_anglez)*ry
                rx=x
                rz=z
                x=np.cos(rotate_angley)*rx-np.sin(rotate_angley)*rz
                z=np.sin(rotate_angley)*rx+np.cos(rotate_angley)*rz
                x+=ox
                y+=oy
                z+=oz
                trajactory.append([x,y,z])       
        if string>=4:
            finger=0
        else:
            finger=4-string
        self._fingerPos[finger]=trajactory
        self._finger_tick_id[finger]=0
        self._fingerstate[finger]=self._posforendtickstring(endtick,int(string))
        self._last_trajactory=trajactory
        self._fingerstring[finger]=int(string)
    def _posforendtickstring(self,tick:int,string:int):
        #索引从0开始
        pos=0
        ns=string-1
        cnt=0
        while cnt*6+ns<cons.NUM_KEYS:
            if self._goalstate[tick][cnt*6+ns][0]:
                pos=cnt+1
            cnt+=1
        return pos

    def _pos_to_note(self,string,pos):
        num=int((STANDPOS[string]+pos)/12)
        name=(STANDPOS[string]+pos)%12
        newPitch=STANDPITCH[string]+num
        if(self._test==2):
            print(NAMEDIR[name]+str(newPitch))
        return NAMEDIR[name]+str(newPitch)
    
    def _string_play(self,string):
        #手指拨弦弹奏,在手指达到轨迹的第n个点时,弹奏
        #string 0-5
        note=self._pos_to_note(string,self._string_max_pos[string])
        if(self._test==2):
            print(str(string+1)+' '+str(self._string_max_pos[string]))
            self._string_player[string].playNote(note)
        self._notes.append((self._tick_id,note))
        

    def _string_stop(self,string):
        self._string_player[string].stop()

    def _update_music_play(self):
        # 左手换按住的弦停止弹奏
        maxpos=[int(0) for i in range(6)]
        for key in range(cons.NUM_KEYS):
            string=int(key%6)
            pos=int(key/6)+1
            if maxpos[string]<pos and self.guitar.activation[key]:
                maxpos[string]=pos

        for i in range(6):
            if self._string_max_pos[i]!=maxpos[i]:
                self._string_max_pos[i]=maxpos[i]

        # pass
        
        
    def _update_right_hand_pos(self,physics: mjcf.Physics):
        #手腕保持住
        for i,name in enumerate(fix_joints):
            joint=self.right_hand.mjcf_model.find('joint',name)
            if(joint!=None):
                physics.bind(joint).qpos=fix_pos[i]
                
        #空弦列表
        stringpos0=[]
        stringpluck=[0 for i in range(6)]

        #手指拨动时按照轨迹移动，否则保持原型
        for i in range(4):
            if self._fingerstate[i]!=-1:
                site_pos=self._fingerPos[i][self._finger_tick_id[i]]
                ikresult=qpos_from_site_pose(physics,'rh_shadow_hand/'+self.right_hand.fingertip_sites[i].name
                            ,site_pos,None,FINGER_JOINT[i])
                if(ikresult.success==True):
                    physics.data.qpos=ikresult.qpos
                self._finger_tick_id[i]+=1
                if wait==0:
                    if self._finger_tick_id[i]==_ANIMATION_NUM:
                        self._string_play(self._fingerstring[i]-1)
                        stringpluck[self._fingerstring[i]-1]=1
                if wait==1:
                #在10个tick内 如果有一个tick达到了目标 就弹奏，否则在最后一个tick演奏
                    if self._fingerstate[i]==self._string_max_pos[self._fingerstring[i]-1]:
                        if self._fingerstate[i]==0:
                            #如果与该空弦时间相同的弦，也按住了正确的位置，此时空弦才发出声响
                            stringpos0.append(i)
                            pass
                        else:
                            self._string_play(self._fingerstring[i]-1)
                            self._fingerstate[i]=-1
                if(self._finger_tick_id[i]>=len(self._fingerPos[i])):
                    self._finger_tick_id[i]=0
                    self._fingerstate[i]=-1
                    self._fingerPos[i][0]=site_pos
                    self._fingerstring[i]=0
                self._last_finger_qpos[i]=physics.bind(self._fingers_joint[i]).qpos.copy()
                
            else:
                physics.bind(self._fingers_joint[i]).qpos=self._last_finger_qpos[i]

        #如果与该空弦时间相同的弦，也按住了正确的位置，此时空弦才发出声响
        if wait:
            for i in stringpos0:
                if self._fingerstate[i]==self._string_max_pos[self._fingerstring[i]-1]:
                    can = 1
                    for j in range(4):
                        if self._fingerstate[j]!=0 and self._fingerstate[j]!=-1:
                            can=0
                    if can:
                        self._string_play(self._fingerstring[i]-1)
                        self._fingerstate[i]=-1
                    
        physics.bind(self._fingers_joint[4]).qpos=self._last_finger_qpos[4]
        self._stringpluck=stringpluck
    

    def _compute_fingering_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing the distance between the fingers and the keys."""
        def _reward_distance(diff,bound):
            rew=tolerance(
                    diff,
                    bounds=(0,bound),
                    margin=(self._finger_margin),
                    sigmoid=self._sigmoid,
                    value_at_margin=0.1
            )
            return rew
        
        def _reward_distance_y(diff,xbound):
            dx=abs(diff[0])
            dz=abs(diff[2])
            dy=abs(diff[1])
            zbound=0.005
            hstart=0.007
            if(dx<=xbound and dz<=zbound):
                hx=hstart*tolerance(
                    dx,
                    bounds=(xbound,xbound),
                    margin=(xbound),
                    sigmoid='reciprocal',
                    value_at_margin=0.0001,
                )
                hz=hstart*tolerance(
                    dz,
                    bounds=(zbound,zbound),
                    margin=(zbound),
                    sigmoid='reciprocal',
                    value_at_margin=0.0001,
                )
                hstart=(hx+hz)/2

            if(dy<hstart):
                rew=tolerance(
                    dy,
                    bounds=(hstart,hstart),
                    margin=(hstart),
                    sigmoid='reciprocal',
                    value_at_margin=0.0001
                )
            else:
                rew=tolerance(
                    dy,
                    bounds=(hstart,hstart),
                    margin=(self._finger_margin),
                    sigmoid=self._sigmoid,
                    value_at_margin=0.1
                )
            # print(hstart)
            return rew

            

        
        def _diff_finger_to_pos(pos,finger,hand):
            fingertip_site = hand.fingertip_sites[finger]
            fingertip_pos = physics.bind(fingertip_site).xpos.copy()
            #key_geom_pos来计算键的表面位置
                
            diff = pos - fingertip_pos
            return diff
        def _reward_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand:shadow_hand.ShadowHand
        ) -> List[float]:
            rewards=[]
            minpos=30
            minkey=120
            flag=None
            for key, mjcf_fingering in hand_keys:
                rew=0
                if minpos>int(key/6):
                    minpos=int(key/6)
                    minkey=key
                key_site= self.guitar.keys_sites[key]
                key_site_pos = physics.bind(key_site).xpos.copy()
                diff=_diff_finger_to_pos(key_site_pos,mjcf_fingering,hand)
                
                z=abs(diff[2]) #上下
                rew+=_reward_distance(z,self._finger_bound)
                x=abs(diff[0]) #左右
                rew+=_reward_distance(x,LENGTH[int(key/6)])


                y=abs(diff[1]) #前后
                rew+=_reward_distance(y,self._finger_bound)
                rewards.append(rew/3)
                # rew+=_reward_distance(np.linalg.norm(diff),self._finger_bound)
                # rewards.append(rew)
            # if not minkey==120: 
            #     key_site=self.guitar.keys_sites[minkey]
            #     key_site_pos = physics.bind(key_site).xpos.copy()
            #     key_site_pos[1]=-0.01
            #     key_site_pos[2]=0.7
            #     diff=_diff_finger_to_pos(key_site_pos,0,hand)
            #     flag=np.where(diff[1]>0,1,-1)
            #     rewards.append(flag*_reward_distance(np.linalg.norm(diff),self._finger_bound))

            
            return rewards
        
        rews = _reward_finger_to_key(self._lh_keys_current, self.left_hand)
        # print("distance:{}".format(np.mean(distances)))
        # distances += _distance_finger_to_key(self._lh_keys_current, self.right_hand)

        # Case where there are no keys to press at this timestep.
        if not rews:
            return 1.0
        #计算各个手指到对应键的平均距离
        return float(np.mean(rews))
    
    def _update_goal_state(self) -> None:
        if self._tick_id >= self._goal.totaltick:
            return

        #目标状态是一个3维数组
        self._obs_goal_state = np.zeros(
            (self._n_steps_lookahead+1, cons.NUM_KEYS ,2),
            dtype=np.float64,
        )
        self._obs_string_state=np.zeros(
            (self._n_steps_lookahead+1, 6 ,2),
            dtype=np.float64,
        )
        #开始为当前的tick数
        t_start = self._tick_id
        #结束为n步之后与末位的最小值
        t_end = min(t_start + self._n_steps_lookahead + 1,self._goal.totaltick)
        for i, t in enumerate(range(t_start, t_end)):
            #更新第t秒 激活的键盘数
            if(STRINGSTATE):
                self._obs_string_state[i]=self._stringstate[t]
            self._obs_goal_state[i]=self._goalstate[t]
        
            
    def _reset_quantities_at_episode_init(self,physics: mjcf.Physics=None) -> None:
        
        self._last_action=np.zeros(len(self.left_hand.actuators))
        self._stringpluck=[0 for i in range(6)]
        self._last_reward=0
        self._tick_id:int=0
        self._should_terminate=False
        self._discount:float=1.0
        self._lh_keys_current:List[Tuple[int, int]]=[]
        self._fingerPos=[[] for i in range(5)]
        # self._F1=[]
        if(physics!=None):
            self.init_hand_pos(physics)
            
            
        #弦的拨弦状态 -1表示没有弹奏，其他表示所应按住的弦
        self._fingerstate=np.ones(5)*(-1)
        self._fingerstring=np.zeros(5,dtype=int)
        #手指的轨迹数组

        
        #手指的单独tick
        self._finger_tick_id=[0 for i in range(5)]
        
    def init_hand_pos(self,physics:mjcf.Physics):
        for i,name in enumerate(INITIAL_NAME):
            joint=self.left_hand.mjcf_model.find('joint',name)
            if(joint!=None):
                physics.bind(joint).qpos=INITIAL_POS[i]
        if self._test:
            for i,name in enumerate(INITIAL_RIGHT_NAME):
                joint=self.right_hand.mjcf_model.find('joint',name)
                if(joint!=None):
                    physics.bind(joint).qpos=INITIAL_RIGHT_POS[i]
            for i in range(4):
                if i ==0:
                    site=self.guitar.mjcf_model.find('site','test4')
                    sitepos=physics.bind(site).xpos.copy()
                else:
                    site=self.guitar.mjcf_model.find('site','test'+str(4-i))
                    sitepos=physics.bind(site).xpos.copy()
                ikresult=qpos_from_site_pose(physics,'rh_shadow_hand/'+self.right_hand.fingertip_sites[i].name
                            ,sitepos,None,FINGER_JOINT[i])
                if(ikresult.success==True):
                    # joints=[]
                    # for jointname in FINGER_JOINT[i]:
                    #     joints.append(self.root_entity.mjcf_model.find('joint',jointname))
                    physics.data.qpos=ikresult.qpos
                self._fingerPos[i].append(sitepos)

            #每根手指对应的joint存起来
            for i in range(5):
                for name in FINGER_JOINT[i]:
                    joint=self.root_entity.mjcf_model.find('joint',name)
                    if(joint!=None):
                        self._fingers_joint[i].append(joint)
                #保存每根手指的最后的joint
                self._last_finger_qpos[i]=physics.bind(self._fingers_joint[i]).qpos.copy()
                
                
        
        
        
    #composer method
    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
                                        ) -> None:
        self._reset_quantities_at_episode_init(physics)


        
    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Applies the control to the hands."""
        # two hand train together
        # action_right, action_left = np.split(action[:], 2)
        # self.right_hand.apply_action(physics, action_right, random_state)
        # self.left_hand.apply_action(physics, action_left, random_state)
        
        #one hand train
        self._last_action=action
        self.left_hand.apply_action(physics, action, random_state)

    def after_substep(self, physics, random_state):
        if(self._tick_id>=self._goal.totaltick):
            return



    def after_step(self, physics, random_state):
        self._tick_id+=1
        self._should_terminate=self._tick_id>=self._goal.totaltick
        # self._goal_current=self._obs_goal_state[0]
        # self._string_current=self._obs_string_state[0]

        
        keys = np.flatnonzero(self._goal_current[:,0])
        fingers=self._goal_current[keys,1]
        self._lh_keys_current:List[Tuple[int, int]]=[]
        for i,key in enumerate(keys):
            self._lh_keys_current.append((int(keys[i]),int(fingers[i])))
        
        
        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:,0])
        # print(should_not_be_pressed)
        
        # print(self.guitar.activation)
        self._failure_termination = self.guitar.activation[should_not_be_pressed].any()
        # self._F1.append(self.compute_F1_score())
        if self._test:
            self._update_right_hand_pos(physics)
        if not self._should_terminate and self._test:
            for i in range(6):
                endtick=self._tick_id+3
                if endtick<self._goal.totaltick and self._pluckstate[endtick][i]:
                    self._create_trajactory(i+1,physics,_ANIMATION_NUM,endtick)

                    
                    

        if len(self._last_trajactory)!=0:
            physics.bind(self._trajactory_site).xpos=self._last_trajactory
            
        if self._test:
            self._update_music_play()
        if not self._should_terminate and self._test:
            goalstate=np.array(self._goalstate[self._tick_id])
            # print(self._tick_id)
            self.guitar._update_color_by_goal(physics,goalstate)
        
    def compute_F1_score(self):
        tp:int
        fp:int
        fn:int
        self._goal_current=self._obs_goal_state[0]
        self._string_current=self._obs_string_state[0]
        keys = np.flatnonzero(self._goal_current[:,0]) #当前目标需要激活的键
        ons = np.flatnonzero(self.guitar.activation[:]) #当前激活的键
        tp=len(set(ons).intersection(set(keys)))
        fp=len(set(ons).difference(set(keys)))
        fn=len(set(keys).difference(set(ons)))
        return np.array([tp,fp,fn])
        
        
    
    #判断任务是否为终止状态
    #有键按错了且_wrong_press_termination为True表示结束任务
    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics  # Unused.
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False
        
    def get_reward(self, physics) -> float:
        
        return self._last_reward
    
    def get_discount(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return self._discount
    
    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        # right_spec = self.right_hand.action_spec(physics)
        left_spec = self.left_hand.action_spec(physics)
        # hands_spec = spec_utils.merge_specs([right_spec, left_spec])

        # return hands_spec
        return left_spec

    
    
    def _add_observables(self) -> None:
        # Enable hand observables.
        enabled_observables = [
            # "fingertip_positions",
            "joints_pos",
            # "position",
            # "joints_vel",
            # "joints_torque",
    
        ]
        #将手的joints_pos观察变量开启
        for hand in [self.right_hand, self.left_hand]:
        # for hand in [self.left_hand]:
            for obs in enabled_observables:
                getattr(hand.observables, obs).enabled = True

        # This returns the current state of the guitar keys.
        if(not self._disable_key_reward):
            self.guitar.observables.state.enabled = True

        # 将手指的目标状态和键盘的目标状态都放到observation中
        # This returns the goal state for the current timestep and n steps ahead.
     
        
        
        def _get_goal_state(physics) -> np.ndarray:
            # self._update_goal_state()
            fingstate=np.array([[0,0,0] for i in range(5)])
            # fingrelativestate=np.array([[0,0,0] for i in range(5)])
            # fingstate=np.array([0 for i in range(5)])
            minpos=20
            minkey=0
            for key, mjcf_fingering in self._lh_keys_current:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_site= self.guitar.keys_sites[key]
                key_site_pos = physics.bind(key_site).xpos.copy()
                pos=key/6
                if(pos<minpos):
                    minpos=pos
                    minkey=key
                for i in range(3):
                    fingstate[mjcf_fingering][i]=norm_from_range(key_site_pos[i],FINGERPOSRANGE[i])
                    # fingrelativestate[mjcf_fingering][i]=(fingstate[mjcf_fingering][i]-norm_from_range(fingertip_pos[i],FINGERPOSRANGE[i]))/2
            
            # 设定的拇指的目标位置
            # key_site= self.guitar.keys_sites[minkey]
            # key_site_pos = physics.bind(key_site).xpos.copy()
            # key_site_pos[1]=-0.01
            # key_site_pos[2]=0.7
            # for i in range(3):
            #     fingstate[0][i]=norm_from_range(key_site_pos[i],FINGERPOSRANGE[i])
                    

            #手指距离目标相对位置
            # fingrelativestate=fingrelativestate.ravel()
            fingstate=fingstate.ravel()
            # goalpos=[]
            goal=[]
            if(not self._disable_key_reward):
                if(STRINGSTATE):
                    goal=self._obs_string_state[:,:,0].ravel()/(cons.NUM_KEYS/6);
                else:
                    goal=self._obs_goal_state[:,:,0].ravel()
                # goal=self._obs_goal_state.ravel()
            return np.concatenate([goal,fingstate
                                #    ,fingrelativestate
                                   ])               


        def _get_last_state(physics)-> np.ndarray:
            reward=np.array([self._last_reward])
            action=np.array(self._last_action)
            
            return np.concatenate([action,reward])
        
        def _get_pluck_state(physics)-> np.ndarray:
            stringpluck=np.array(self._stringpluck)
            return stringpluck
        def _get_f1_score(physics) -> np.ndarray:
            self._update_goal_state()
            f1=self.compute_F1_score()
            return f1
        
        def _get_reward(physics) -> np.ndarray:
            # self._update_goal_state()
            f_reward=(1-self._rightKey_weight)*self._compute_fingering_reward(physics)
            e_reward=self._compute_energy_reward(physics)
            k_reward=self._rightKey_weight*self._compute_key_press_reward(physics)
            self._last_reward=f_reward+e_reward+k_reward
            return np.array([f_reward,e_reward,k_reward])
        
        def _get_task_qpos(physics) -> np.ndarray:
            return np.array(physics.data.qpos)
            
        reward_observable=observable.Generic(_get_reward)
        reward_observable.enabled=True
        f1_state_observable=observable.Generic(_get_f1_score)
        f1_state_observable.enabled=True
        last_state_observable=observable.Generic(_get_last_state)
        last_state_observable.enabled=True
        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        string_pluck_observable=observable.Generic(_get_pluck_state)
        string_pluck_observable.enabled = True
        task_qpos=observable.Generic(_get_task_qpos)
        task_qpos.enabled=True
        self._task_observables = {
            "f1": f1_state_observable,
            "goal": goal_observable,
            "last_state": last_state_observable,
            "reward": reward_observable,
            "pluck": string_pluck_observable,
            "qpos": task_qpos
        }