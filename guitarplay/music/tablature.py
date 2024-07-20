from guitarplay.music.note import Note
from guitarplay.modelpy.guitar import guitar_constants as cons
ChordDic={
    'C':['5 3','4 2','2 1'],
    'Dm':['3 2','2 3','1 1'],
    'Em':['5 2','4 2'],
    'F':['4 3','3 2','2 1'],
    'G':['6 3','5 2','1 3'],
    'Am':['4 2','3 2','2 1'],
    'A':['4 2','3 2','2 2'],
}

class Tablature():
    """创建六线谱"""
    def __init__(self,notes=[],songname=""):
        self.songname=songname
        self.keys=[[] for i in range(cons.NUM_KEYS)]
        self.plucks=[[]  for i in range(6)]
        self.endtime=0  
        self.notes=[]
        self.addNotes(notes)

    def updateEndtime(self):
        endtime=-1
        for note in self.notes:
            if(endtime<note.time+note.duration):
                endtime=note.time+note.duration
                
        self.endtime=endtime
        
    def notesToState(self,notes):
        self.keys=[[] for i in range(cons.NUM_KEYS)]
        self.plucks=[[]  for i in range(6)]
        notes=self.sortNotes(notes)
        notes=self.arrangeFinger(notes)
        if(notes==False): return False
        for note in notes:
            name=note.name
            time=note.time
            duration=note.duration
            string=int(name.split(' ')[0])-1
            pos=int(name.split(' ')[1])-1
            if(pos>=0):
                self.keys[pos*6+string].append((time,duration,note.finger))
            if(string>=0):
                self.plucks[string].append((time,duration,pos,note.finger))
                
                
        
    def addNotes(self,notes):
        self.notes=self.notes+notes
        self.updateEndtime()
        self.notesToState(self.notes)
        

    def addNotesWithSameStartTime(self,names,time,duration):
        notes=[]
        for name in names:
            note=Note(name,time,duration)
            notes.append(note)
        
        self.addNotes(notes)
        
    def addNotesWithSameDuration(self,names,duration):
        notes=[]
        time=0
        for name in names:
            note=Note(name,time,duration)
            time+=duration
            notes.append(note)
        
        self.addNotes(notes)
        
    def addChord(self,chordNames,times,durations):
        for i,chordName in enumerate(chordNames):
            notesname=ChordDic[chordName]
            time=times[i]
            duration=durations[i]
            self.addNotesWithSameStartTime(notesname,time,duration)
            
        
    def addChordAutoTime(self,chordNames,duration):
        times=[]
        durations=[]
        time=0
        for chordname in chordNames:
            times.append(time)
            durations.append(duration)
            time+=duration
        self.addChord(chordNames,times,durations)
    
    def addNotesByTimeAuto(self,names,durations):
        notes=[]
        time=0
        for i,name in names:
            note=Note(name,time,durations[i])
            time+=durations[i]
            notes.append(note)
        
        self.addNotes(notes)
    
    def sortNotes(self,notes):
        def sort_key(note):
            return note.time
        sorted_notes=sorted(notes,key=sort_key)
        return sorted_notes
    def addListNotes(self,noteslist,k=1):
        """
        add notes list
        Args:
            notesarray(List):each element include note's name,time,duration
            example [['5 3',0,0.25],['5 3',0,0.3]]
            k:the speed of song,k>1 means quicker,k<1 means slower 
                        
        """
        notes=[]
        for note in noteslist:
            notes.append(Note(note[0],note[1]*(1/k),note[2]*(1/k)))
        self.addNotes(notes)


    def arrangeFinger(self,notes):
        def sortkey1(choice):
            return choice[1]

        def sortkey2(choice):
            return choice[0]
        
        def sortkey(string):
            return -string
        
        #检查同一时刻位于同一品格的手指
        fingers=[0 for i in range(0,5)]
        fingersname=["-1 -1" for i in range(0,5)]
        current=0
        handpos=1
        #handpos代表把位
        while(1):
            if(current==len(notes)):
                break
            note=notes[current]
            string=note.name.split(' ')[0]
            string=int(string)
            pos=note.name.split(' ')[1]
            pos=int(pos)
            prbchoices=[]
            if(pos>0):
                time=note.time
                duration=note.duration
                choice=-1
                min=2000
                
                #若该把位够不着给定的品
                if pos-handpos>4:
                    handpos=pos
                if handpos-pos>0:
                    handpos=max(pos-3,1)
                    
                #寻找手指来按住该弦
                for i in reversed(range(1,5)):
                    #如果该手指已经确定被放开了
                    if(fingers[i]<time or abs(fingers[i]-time)<=0.001):
                        #找距离最小的手指作为目标手指
                        prbchoices.append((i,abs(i+handpos-1-pos)))
                
                #如果没有可选的手指，就返回错误
                if(len(prbchoices)==0):
                    with open('arrange.txt','a') as f:
                        f.writelines(f'{notes[current].name} have not enough finger to press\n')
                    current+=1
                    continue
                
                possame=[string]
                #找到时间相同的Notes
                for j in range(current+1,len(notes)):
                    newnote=notes[j]
                    newstring=newnote.name.split(' ')[0]
                    newstring=int(newstring)
                    newpos=newnote.name.split(' ')[1]
                    newpos=int(newpos)
                    newtime=newnote.time
                    #如果时间不同就跳出
                    if abs(newtime-time)>0.001:
                        break

                    if pos==newpos:
                        possame.append(newstring)


                prbchoices=sorted(prbchoices,key=sortkey1)
                if len(possame)==0:
                    #选择相对位置最小的手指
                    choice=prbchoices[0]

                else:

                    if len(prbchoices)<len(possame):
                        with open('arrange.txt','a') as f:
                            f.writelines(f'{notes[current].name} have not enough finger to press\n')
                        current+=1
                        continue
                    else:
                        #首先有n个possame证明我必须要n个手指去按住，因此我最优选择
                        #首先选n根距离最小的手指
                        prbchoices=prbchoices[0:len(possame)]
                        
                        #然后比较当前string在所有posssame的位置，根据该位置来安排手指
                        #将手指序列按照序号升序排好
                        prbchoices=sorted(prbchoices,key=sortkey2)
                        
                        #将弦按照降序排好
                        possame=sorted(possame,key=sortkey)

                        for id,s in enumerate(possame):
                            if s==string:
                                choice=prbchoices[id][0]


                    
                fingers[choice]=time+duration
                fingersname[choice]=notes[current].name
                notes[current].finger=choice
                with open('arrange.txt','a') as f:
                    f.writelines(f"{notes[current].name} {notes[current].finger}\n")
                
            current+=1
        
        return notes
                
            
        
        