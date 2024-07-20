from guitarplay.modelpy.guitar import guitar_constants as cons
from guitarplay.music.tablature import Tablature
import numpy as np
class Goal():
    def __init__(self,init_buffer_time,control_timestep,table):
        self._dt=control_timestep
        self._init_buffer_time=init_buffer_time
        self._totaltick=int((table.endtime+init_buffer_time)/self._dt)
        self._goalstate,self._pluckstate,self._stringstate=self.tableToGoal(table)
        
    
    def tableToGoal(self,table:Tablature):
        goalstate=[[[0,-1] for i in range(cons.NUM_KEYS)] for j in range(self._totaltick)]
        pluckstate=[[0 for i in range(6)] for j in range(self._totaltick)]
        stringstate=[[[int(cons.NUM_KEYS/6),-1] for i in range(6)] for j in range(self._totaltick)]
        

        keys=table.keys
        plucks=table.plucks
        for i,key in enumerate(keys):
            for step in key:
                time,duration,finger=step
                time+=self._init_buffer_time
                ticknum=duration/self._dt
                starttick=time/self._dt
                endtick=starttick+ticknum
                for k in range(int(starttick),int(endtick)):
                    goalstate[k][i][0]=1
                    goalstate[k][i][1]=finger
                    stringstate[k][int(i%6)]=[int(i/6)+1,finger]
                    

        for i,stringplucks in enumerate(plucks): #每根弦拨动的时间
            for step in stringplucks: 
                time,duration,pos,finger=step
                time+=self._init_buffer_time
                ticknum=duration/self._dt
                starttick=time/self._dt
                endtick=starttick+ticknum
                for k in range(int(starttick),int(endtick)):
                    if(k==int(starttick)):
                        pluckstate[k][i]=1
                    stringstate[k][i]=[(pos+1),finger]
                
        with open('goalstate.txt','w') as f:
            for i in range(self._totaltick):
                if(i==0 or goalstate[i-1]!=goalstate[i]):
                    f.writelines(f"goalstate{i}\n")
                    for j in range(6):
                        for k in range(3):
                            f.write(f"{goalstate[i][k*6+j]} ")
                        f.write("\n")

                    f.writelines(f"pluckstate{i} {pluckstate[i]}\n")
                    f.writelines(f"stringstate{i} {stringstate[i]}\n")

        return np.array(goalstate),np.array(pluckstate),np.array(stringstate)
    
    
    @property
    def goalstate(self):
        return self._goalstate
    @property
    def pluckstate(self):
        return self._pluckstate
    @property
    def stringstate(self):
        return self._stringstate
    @property
    def totaltick(self):
        return self._totaltick