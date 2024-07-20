import sys
# sys.path.append('/home/lcy/guitarplayerlast') #add your project path where contains guitarplay filefolder
# import guitarplay.envs.guitar_gym_env
# env=guitarplay.envs.guitar_gym_env.GuitarPlayEnv()
import example
import json
import gymnasium as gym
from sbx import TQC, DroQ, SAC, PPO, DQN, TD3, DDPG
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import sys
from typing import Callable
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from example.envs.guitar_gym_env import sequece
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from guitarplay.suite.tasks.guitar_task_withhands import STRINGSTATE
from guitarplay.suite.tasks.base import _FRAME_RATE
from guitarplay.suite.tasks.test_task import traintask
import time
import gc
task=traintask().task
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func




class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enviorment parameters

        self.add_argument('--change_color', default=False, action='store_true')
        self.add_argument('--disable_finger_reward', default=False, action='store_true')
        self.add_argument('--disable_key_reward', default=False, action='store_true')
        self.add_argument('--disable_energy_reward', default=False, action='store_true')
        self.add_argument('--wrong_press_termination', default=False, action='store_true')
        self.add_argument("--init_buffer_time", default=1, type=int)
        self.add_argument("--n_step_lookahead", default=10, type=int)
        self.add_argument("--table", default=1, type=int)
        self.add_argument("--finger_bound",default=0.01,type=float)
        self.add_argument("--finger_margin",default=0.1,type=float)
        self.add_argument("--energy_coef",default=0.005,type=float)
        self.add_argument("--key_bound",default=0.01,type=float)
        self.add_argument("--key_margin",default=0.01,type=float)
        self.add_argument("--disreduce_space",default=False,action='store_true')
        self.add_argument("--sigmoid",default="gaussian",choices=['reciprocal','long_tail',
                                                                  'hyperbolic','linear',
                                                                  'cosine','quadratic'])
        self.add_argument("--rightKey_weight",default=0.5,type=float)
        self.add_argument("--operator",default='+',choices=['*'])
        self.add_argument('--normalization',default=True,action='store_false')
        self.add_argument('--n_env',default=1,type=int)
        self.add_argument('--env_type',default='dummy',choices=['subproc'])
        # DroQ hyperparameters
        self.add_argument("--batch_size", default=256, type=int)
        self.add_argument("--total_step",default=int(1e6),type=int)
        self.add_argument("--dropout_rate",default=0.01,type=float)
        self.add_argument("--start_step",default=50000,type=int)
        self.add_argument("--use_noise",default=False,action='store_true')
        self.add_argument("--seed",default=42,type=int)
        self.add_argument("--buffer_size",default=int(1e6),type=int)
        self.add_argument("--learningrate",default=3e-4,type=float)
        self.add_argument("--linerlr",default=False,action='store_true')
        self.add_argument("--freedom",default=False,action='store_true')
        self.add_argument("--gamma",default=0.99,type=float)
        # Model Save Dir
        self.add_argument("--work_dir",default='./result/',type=str)
        
        
        # test
        self.add_argument("--test",default=0, type=int)
        
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        
        done=self.locals['dones'][0].item()
        if(done):
            f1=0
            precision=0
            recall=0
            k_reward=0
            e_reward=0
            f_reward=0
            cnt=0
            for info in self.locals['infos']:
                f1+=info['f1']
                precision+=info['precision']
                recall+=info['recall']
                k_reward+=info['reward'][2]
                e_reward+=info['reward'][1]
                f_reward+=info['reward'][0]
                cnt+=1
            self.logger.record('env/f1',f1/cnt)
            self.logger.record('env/precision',precision/cnt)
            self.logger.record('env/recall',recall/cnt)
            self.logger.record('env/k_reward',k_reward/cnt)
            self.logger.record('env/e_reward',e_reward/cnt)
            self.logger.record('env/f_reward',f_reward/cnt)
            
        if self.n_calls % self.check_freq == 0:
          
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 10 episodes
                mean_reward = np.mean(y[-10:])
                if self.verbose >= 1:
                  print(f"Num timesteps: {self.num_timesteps}")
                  print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                      print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path+'/best_model')

        return True
    
    
class Trainer():
    def __init__(self,args):
        self.args=args
        self.setup(args)
        np.random.seed(args.seed)
        self.log_dir =os.path.join(self.experiment_dir, 'resultandmodel')
        os.makedirs(self.log_dir, exist_ok=True)
        self.total_step=args.total_step
        # Create and wrap the environment
        env=self.create_env(args)
        self.env=env
        # model=DroQ.load('droq_guitar',env)
        # env.unwrapped.viewer(model)
        # Add some action noise for exploration
        n_actions = env.action_space.shape[-1]
        
        print(f"training task:{task[args.table].songname}")
        
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        if args.linerlr:
            lr=linear_schedule(args.learningrate)
        else:
            lr=args.learningrate
        # Because we use parameter noise, we should use a MlpPolicy with layer normalization
        if(args.use_noise):
            self.model = DroQ("MlpPolicy", env, action_noise=action_noise,gradient_steps=-1,
                         verbose=1,tensorboard_log=self.experiment_dir+'/tensorboardlog',
                         seed=args.seed,batch_size=args.batch_size,learning_rate=lr,
                         learning_starts=args.start_step,dropout_rate=args.dropout_rate,buffer_size=args.buffer_size,gamma=args.gamma)
        else:
            self.model = DroQ("MlpPolicy", env,gradient_steps=-1,
                         verbose=1,tensorboard_log=self.experiment_dir+'/tensorboardlog',
                         seed=args.seed,batch_size=args.batch_size,learning_rate=lr,
                         learning_starts=args.start_step,dropout_rate=args.dropout_rate,buffer_size=args.buffer_size,gamma=args.gamma)
        # Create the callback: check every 1000 steps
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=self.log_dir)
        # Train the agent

        # model=DroQ.load('droq_guitar',env)
        # env.unwrapped.viewer(model)
    def create_env(self,args):
        if args.test:
            env = gym.make("GuitarPlay-v0",render_mode="human",
                    table=args.table,
                    change_color_on_activate=args.change_color,
                    disable_finger_reward=args.disable_finger_reward,
                    disable_key_reward=args.disable_key_reward,
                    disable_energy_reward=args.disable_energy_reward,
                    wrong_press_termination=args.wrong_press_termination,
                    init_buffer_time=args.init_buffer_time,
                    n_step_lookahead=args.n_step_lookahead,
                    finger_bound=args.finger_bound,
                    finger_margin=args.finger_margin,
                    energy_coef=args.energy_coef,
                    key_bound=args.key_bound,
                    key_margin=args.key_margin,
                    sigmoid=args.sigmoid,
                    rightKey_weight=args.rightKey_weight,
                    operator=args.operator,
                    reduced_action_space=not args.disreduce_space,
                    normalization=args.normalization,
                    test=args.test,
                    freedom=args.freedom
                    )
            env = Monitor(env, self.log_dir,info_keywords=('f1','precision','recall','reward'))
        else:
            if args.env_type=='dummy':
                envtype=DummyVecEnv
            else:
                envtype=SubprocVecEnv
            env =make_vec_env("GuitarPlay-v0",n_envs=args.n_env,env_kwargs={ 
                    'table':args.table,
                    'change_color_on_activate':args.change_color,
                    'disable_finger_reward':args.disable_finger_reward,
                    'disable_key_reward':args.disable_key_reward,
                    'disable_energy_reward':args.disable_energy_reward,
                    'wrong_press_termination':args.wrong_press_termination,
                    'init_buffer_time':args.init_buffer_time,
                    'n_step_lookahead':args.n_step_lookahead,
                    'finger_bound':args.finger_bound,
                    'finger_margin':args.finger_margin,
                    'energy_coef':args.energy_coef,
                    'key_bound':args.key_bound,
                    'key_margin':args.key_margin,
                    'sigmoid':args.sigmoid,
                    'rightKey_weight':args.rightKey_weight,
                    'operator':args.operator,
                    'reduced_action_space':not args.disreduce_space,
                    'normalization':args.normalization,
                    'test':args.test,
                    'freedom':args.freedom
                },monitor_kwargs={
                    'info_keywords':('f1','precision','recall','reward')
                },monitor_dir=self.log_dir,
                vec_env_cls=envtype
            )

        return env
    def train(self):
        self.model.learn(total_timesteps=self.total_step,
            callback=self.callback,progress_bar=True,
            log_interval=1,tb_log_name="droq_guitar",
            )
    def test(self):
        load_path=os.path.join(self.log_dir,'best_model')
        model=DroQ.load(load_path+'/best_model',self.env)
        self.env.unwrapped.viewer(model)
    def eval(self,num):
        load_path=os.path.join(self.log_dir,'best_model')

        model=DroQ.load(load_path+'/best_model',self.env)
        f1s=[]
        rs=[]
        pres=[]
        res=[]
        for i in range(num):
            obs,info=self.env.reset()
            end=False
            r=0
            while 1:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated,_, info=self.env.step(action)
                end=terminated 
                r+=reward
                if end:
                    f1s.append(info['f1'])
                    pres.append(info['precision'])
                    res.append(info['recall'])
                    rs.append(r)
                    break
        print(f'mean f1 score:{np.mean(f1s)}')
        print(f'mean precision score:{np.mean(pres)}')
        print(f'mean recall score:{np.mean(res)}')
        print(f'mean reward :{np.mean(rs)}')
    def close(self):
        del self.model.replay_buffer
        del self.model
    def setup(self, args):
        exp_name =  \
                    str(args.gamma)+'_'+\
                    str(args.rightKey_weight)+'_'+\
                    str(args.key_bound)+'_'+\
                    str(args.key_margin)+'_'+\
                    str(args.finger_bound)+'_'+\
                    str(args.finger_margin)+'_'+\
                    str(args.energy_coef)+'_'+\
                    str(args.sigmoid)+'_'+\
                    str(args.dropout_rate)+'_'+\
                    str(args.total_step)+'_'+\
                    str(args.normalization)+'_'+\
                    str(args.batch_size)+'_'+\
                    str(args.table)+'_'+\
                    str(args.freedom)+'_'+\
                    str(args.buffer_size)+'_'+\
                    str(args.linerlr)+'_'+\
                    str(args.learningrate)+'_'+\
                    str(STRINGSTATE)+' '+\
                    str(_FRAME_RATE)
        statepart=''
        for state in sequece:
            statepart+='_'+state
        exp_name+=statepart
        self.experiment_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.save_args(args)

    def save_args(self, args):
        with open(os.path.join(self.experiment_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
            
            
args=[
        [0.5, 1, 1, 0.01, 0.1, 0.005, 'gaussian', 256],
        [0.5, 1, 1, 0.01, 0.1, 0.005, 'gaussian', 512],
        [0.5, 1, 1, 0.01, 0.1, 0.005, 'reciprocal', 256],
        [0.5, 1, 1, 0.01, 0.1, 0.005, 'reciprocal', 512],
        [0.5, 1, 1, 0.005, 0.05, 0.0005, 'gaussian', 256],
        [0.5, 1, 1, 0.005, 0.05, 0.0005, 'gaussian', 512],
        [0.5, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 256],
        [0.5, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 512],
        [0.7, 1, 1, 0.01, 0.1, 0.005, 'gaussian', 256],
        [0.7, 1, 1, 0.01, 0.1, 0.005, 'gaussian', 512],
        [0.7, 1, 1, 0.01, 0.1, 0.005, 'reciprocal', 256],
        [0.7, 1, 1, 0.01, 0.1, 0.005, 'reciprocal', 512],
        [0.7, 1, 1, 0.005, 0.05, 0.0005, 'gaussian', 256],
        [0.7, 1, 1, 0.005, 0.05, 0.0005, 'gaussian', 512],
        [0.7, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 256],
        [0.7, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 512],
        [0.9, 1, 1, 0.01, 0.1, 0.005, 'gaussian', 256],
        [0.9, 1, 1, 0.01, 0.1, 0.005, 'gaussian', 512],
        [0.9, 1, 1, 0.01, 0.1, 0.005, 'reciprocal', 256],
        [0.9, 1, 1, 0.01, 0.1, 0.005, 'reciprocal', 512],
        [0.9, 1, 1, 0.005, 0.05, 0.0005, 'gaussian', 256],
        [0.9, 1, 1, 0.005, 0.05, 0.0005, 'gaussian', 512],
        [0.9, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 256],
        [0.9, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 512],
        [0.9, 1, 1, 0.001, 0.1, 0.0005, 'reciprocal', 512,int(3e6)],
        [0.5, 1, 1, 0.001, 0.1, 0.0025, 'reciprocal', 512,int(1e7)],
        [0.9, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 512,int(2e6),'+'],
        [0.9, 1, 1, 0.005, 0.05, 0.0005, 'reciprocal', 512,int(1e6),'*'],
]


def main():
    # train(args[25])
    test(args[25])

def train(a):
    arg=ArgParserTrain().parse_args()
    arg.rightKey_weight=a[0]
    arg.key_bound=a[1]
    arg.key_margin=a[2]
    arg.finger_bound=a[3]
    arg.finger_margin=a[4]
    arg.energy_coef=a[5]
    arg.sigmoid=a[6]
    arg.batch_size=a[7]
    arg.n_env=4
    arg.table=1
    arg.gamma=0.84
    arg.env_type='subproc'
    # arg.sigmoid='gaussian'
    # arg.linerlr=True
    # arg.freedom=True
    arg.use_noise=True
    if(len(a)>8):
        arg.total_step=a[8]
    if(len(a)>9):
        arg.operator=a[9]
    train=Trainer(arg)
    train.train()

def test(a):
    arg=ArgParserTrain().parse_args()
    arg.rightKey_weight=a[0]
    arg.key_bound=a[1]
    arg.key_margin=a[2]
    arg.finger_bound=a[3]
    arg.finger_margin=a[4]
    arg.energy_coef=a[5]
    arg.sigmoid=a[6]
    arg.batch_size=a[7]
    arg.use_noise=True
    arg.table=3
    arg.buffer_size=1000000
    arg.gamma=0.84
    # arg.linerlr=True
    arg.freedom=True
    if(len(a)>8):
        arg.total_step=a[8]
    if(len(a)>9):
        arg.operator=a[9]

    arg.test=1
    train=Trainer(arg)

    train.eval(1)
    arg.test=2
    train=Trainer(arg)
    train.test()
    


if __name__=='__main__':
    main()
    
