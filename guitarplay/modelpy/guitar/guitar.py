from dm_control import composer
from dm_control.composer.observation import observable
from guitarplay.modelpy.guitar.guitar_mjcf import guitar
from dm_control import mjcf
import guitarplay.modelpy.guitar.guitar_constants as cons
import numpy as np
class Guitar(composer.Entity):

    def _build(self,change_color_on_activation:bool=True,key_bound: float=cons.ACTIVATION_THRESHOLD):
        self._guitarclass=guitar()
        self._model=self._guitarclass.model
        self.guitarBody=self._model.worldbody.body['guitar']
        self.sites=self.createSites()
        self.forcesensor=self.createForceSensors()
        self.possensor=self.createPosSensors()
        self._change_color_on_activation=change_color_on_activation
        self._key_bound=key_bound
        self._initialize_state()
        
    def createSites(self):
        restLength=cons.LENGTH
        base=cons.BASE
        lastZ=cons.KEY_RIGHTEST_POS
        lastLi=0
        sitelist=[]
        length=[]
        for i in range(int(cons.NUM_KEYS/6)):
            curLi=restLength/base
            newZ=lastZ+((curLi+lastLi)/2)
            length.append(curLi/2)
            for j in range(6):
                site=self.guitarBody.add('site',name="key "+str(j+1)+" "+str(i+1),
                                    pos=[0.02-j*0.008,0.013,newZ],
                                    size=[0.002,0.002,curLi/2],
                                    type="box",
                                    rgba=[1,i%2,0,1],
                                    )
                sitelist.append(site)
            lastLi=curLi
            lastZ=newZ
            restLength-=curLi
        print(length)
        return sitelist       
    
    def createForceSensors(self):
        sensorlist=[]
        for i in range(int(cons.NUM_KEYS/6)):
            for j in range(6):
                sensor=self.mjcf_model.sensor.add('touch',site="key "+str(j+1)+" "+str(i+1))
                sensorlist.append(sensor)
        return sensorlist
    
    def createPosSensors(self):
        sensorlist=[]
        for i in range(int(cons.NUM_KEYS/6)):
            for j in range(6):
                sensor=self.mjcf_model.sensor.add('framepos',objname="key "+str(j+1)+" "+str(i+1),
                                             objtype="site")
                sensorlist.append(sensor)
        return sensorlist
    def _build_observables(self):
        return CreateObservables(self)
    
    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._initialize_state()
        self._update_key_state(physics)
        self._update_key_color(physics)

    def after_substep(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.

    def after_step(self, physics, random_state):
        self._update_key_state(physics)
        self._update_key_color(physics)
        
    def _initialize_state(self) -> None:
        self._state = np.zeros(cons.NUM_KEYS, dtype=np.float64)
        self._activation = np.zeros(cons.NUM_KEYS, dtype=bool)
        
    def _update_key_color(self,physics:mjcf.Physics)->None:
        """Updates the color of the guitar keys."""
 
        if self._change_color_on_activation:
            physics.bind(self.keys_sites).rgba=np.where(
                self._activation[:,None],
                cons.ACTIVATE_COLOR,
                cons.INIT_COLOR
            )
        # else:
            # physics.bind(self.keys_sites).rgba=cons.INIT_COLOR
        
    def _update_color_by_goal(self,physics: mjcf.Physics,goal):
        physics.bind(self.keys_sites).rgba=cons.INIT_COLOR
        goals=np.flatnonzero(goal[:,0])
        actives=np.flatnonzero(self._activation[:])
        if(goals.size>0):
            physics.bind(self.keys_sites).rgba[goals]=cons.PRE_COLOR
        if(actives.size>0):
            physics.bind(self.keys_sites).rgba[actives]=cons.ACTIVATE_COLOR

        
    def _update_key_state(self, physics: mjcf.Physics) -> None:
        """Updates the state of the guitar keys."""
        
        keys_forces = physics.bind(self.keys_forces_sensors).sensordata
        # print('keys_forces{}'.format(keys_forces))
        
        keys_forces=keys_forces.clip(0,cons.MAX_FORCE)
        self._state[:] =keys_forces/cons.MAX_FORCE
        self._activation[:] = (
            np.abs(self._state) >= self._key_bound
        )
        # print(self._activation)
        
        
    @property
    def mjcf_model(self):
        return self._model
    
    @property
    def keys_positions_sensors(self):
        # pos_sensors=self.mjcf_model.find_all('sensor','framepos')
        return self.possensor
    
    @property
    def keys_forces_sensors(self):
        # forse_sensors=self.mjcf_model.find_all('sensor','touch')
        return self.forcesensor
    
    @property
    def keys_sites(self):
        return self.sites
    
    @property
    def activation(self):
        return self._activation
    
    @property
    def state(self):
        return self._state
  
        
        
class CreateObservables(composer.Observables):
    @composer.observable
    def keys_positions(self):
        all_xpos = self._entity.keys_positions_sensors
        return observable.MJCFFeature('site_xpos',all_xpos)
        
    @composer.observable
    def key_forces(self):
        all_force = self._entity.keys_forces_sensors
        return observable.MJCFFeature('sensordata', all_force)
    
    @composer.observable
    def state(self):
        """Returns the guitar key states."""
        def _get_state(physics: mjcf.Physics)->np.ndarray:
            del physics 
            return self._entity._state.astype(np.float64)
        
        return observable.Generic(raw_observation_callable=_get_state)
    
    @composer.observable
    def activation(self):
        """Returns the guitar key activations."""

        def _get_activation(physics: mjcf.Physics) -> np.ndarray:
            del physics  # Unused.
            return self._entity._activation.astype(np.float64)

        return observable.Generic(raw_observation_callable=_get_activation)


        
        