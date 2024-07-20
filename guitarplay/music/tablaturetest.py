from absl.testing import absltest
from guitarplay.music.tablature import Tablature
from guitarplay.music.note import Note
import random

class TableTest(absltest.TestCase):
    # def testSortNotes(self):
    #     notes=[]
    #     num=100
    #     for i in range(num):
    #         pos=random.randint(1,20)
    #         string=random.randint(1,6)
    #         name=str(string)+" "+str(pos)
    #         time=random.randint(0,120)
    #         duration=random.random()
    #         note=Note(name,time,duration)
    #         notes.append(note)
    #     table=Tablature(notes)
    #     for i,note in enumerate(table.notes):
    #         print("time:{} name:{} duration:{}".format(note.time,note.name,note.duration))
    #         if(i+1<len(table.notes)):
    #             self.assertLessEqual(note.time,table.notes[i+1].time)
    #     for key in table.keys:
    #         for i,timedu in enumerate(key):
    #             print("i:{} timedu{}".format(i,timedu))
    #             if(i+1<len(key)):
    #                 self.assertLessEqual(timedu[0],key[i+1][0])
                    
    def testarrangefinger(self):
        table=Tablature()
        table.addNotesWithSameDuration(["5 3","5 3","3 0","3 0","3 2","3 2","3 0","0 0",
                                        "4 3","4 3","4 2","4 2","4 0","4 0","5 3","0 0",
                                        "3 0","3 0","4 3","4 3","4 2","4 2","4 0","0 0",
                                        "3 0","3 0","4 3","4 3","4 2","4 2","4 0","0 0",
                                        "5 3","5 3","3 0","3 0","3 2","3 2","3 0","0 0",
                                        "4 3","4 3","4 2","4 2","4 0","4 0","5 3","0 0",
                                        ],1)
        for note in table.notes:
            print("time:{} name:{} duration:{} finger:{}".format(note.time,note.name
                                                                 ,note.duration,note.finger))
            
        table=Tablature()
        table.addNotesWithSameDuration(["1 0","1 1","1 2","1 3","1 4","1 5","1 4","1 3","1 2","1 1","1 0",
                                        "2 0","2 1","2 2","2 3","2 4","2 3","2 2","2 1","2 0",
                                        "3 0","3 1","3 2","3 3","3 4","3 3","3 2","3 1","3 0",
                                        "4 0","4 1","4 2","4 3","4 4","4 3","4 2","4 1","4 0",
                                        "5 0","5 1","5 2","5 3","5 4","5 3","5 2","5 1","5 0",],1)
        for note in table.notes:
            print("time:{} name:{} duration:{} finger:{}".format(note.time,note.name
                                                                 ,note.duration,note.finger))
            
        table=Tablature()
        table.addChordAutoTime(['C','Am','F','G'],2)
        
        for note in table.notes:
            print("time:{} name:{} duration:{} finger:{}".format(note.time,note.name
                                                                 ,note.duration,note.finger))
        
        

if __name__ == "__main__":
    absltest.main()