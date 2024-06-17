# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:59:09 2023

@author: 19259
"""

#import functions
import numpy as np
import math
import matplotlib.pyplot as plt

class ReadSimcoreSprings():
    def __init__(self,path):
        self.headerDT=np.dtype([
                ('number of steps', 'i4'),
                ('Steps between posit entries', 'i4'),
                ('time step', 'f8'),
                ])
        self.file=open(path+'_crosslink_cut7.spec')
        self.link_array=[]
        self.header = np.fromfile(self.file, dtype=self.headerDT, count=1)[0]
        self.frames=math.floor(self.header[0]/self.header[1])
        #This time step is the time between each recorded frame   
        self.frame_time=self.header[1]*self.header[2]
        print("Simulation has", self.frames, "frames")
        
    def ReadSpecFile(self):
        
        #Define data types
        self.link_paramsDT=np.dtype([
                
                ('double', 'i1'),
                ('free', 'i1'),
                ('diameter', 'f8'),
                ('length', 'f8'),
                              
                ])
    
        self.anchor_paramsDT=np.dtype([
                
                ('bound', 'i1'),
                ('active', 'i1'),
                ('static', 'i1'),
                              
                ])
        
        self.posDT=np.dtype([
                ('position', 'f8',(3))                
                ])
        self.oriDT=np.dtype([
                ('orientation', 'f8',(3))                
                ])
        
    
        total_info=[]
        time_steps=[]
        for t in range(int(self.frames)):
            time_steps.append(t)
            if (t%1000==0):
                print('Reading frame=',t)
            self.links=np.fromfile(self.file, dtype='i4', count=1)[0]
            frame_info=[]
            
            for i in range(self.links):
                
                crosslink_info=[]

                #MAP information
                self.params=np.fromfile(self.file, dtype=self.link_paramsDT, count=1)[0]
                position=np.fromfile(self.file, dtype=self.posDT, count=1)[0]
                orientation=np.fromfile(self.file, dtype=self.oriDT, count=1)[0]
                ID=np.fromfile(self.file, 'i4', count=1)[0]
                
                
                #Read anchor files
                self.paramsA=np.fromfile(self.file, dtype=self.anchor_paramsDT, count=1)[0]
                positionA=np.fromfile(self.file, dtype=self.posDT, count=1)[0]
                orientationA=np.fromfile(self.file, dtype=self.oriDT, count=1)[0]
                np.fromfile(self.file, 'f8', count=1)[0]
                np.fromfile(self.file, 'i4', count=1)[0]
                
                self.paramsB=np.fromfile(self.file, dtype=self.anchor_paramsDT, count=1)[0]
                positionB=np.fromfile(self.file, dtype=self.posDT, count=1)[0]
                orientationB=np.fromfile(self.file, dtype=self.oriDT, count=1)[0]
                np.fromfile(self.file, 'f8', count=1)[0]
                np.fromfile(self.file, 'i4', count=1)[0]
                

                #If double 
                if (self.params['double']==1):
                    crosslink_info.append([position,orientation,positionA,orientationA,positionB,orientationB, self.params['double'],self.params['free'], self.paramsA['bound'],self.paramsB['bound'],self.params[3]])
                else:
                    if positionA[0][0]==0:
                        crosslink_info.append([positionB,orientation,positionA,orientationA,positionB,orientationB,self.params['double'],self.params['free'], self.paramsA['bound'],self.paramsB['bound'],self.params[3]])
                    else:
                        crosslink_info.append([positionA,orientation,positionA,orientationA,positionB,orientationB,self.params['double'],self.params['free'], self.paramsA['bound'],self.paramsB['bound'],self.params[3]])

                frame_info.append(crosslink_info)
            total_info.append(frame_info)
        self.total_info=total_info
        save_name=path+'_processed_crosslink_cut7'
        np.save(save_name,self.total_info);

path='.\\Example\\example'
sim_info=ReadSimcoreSprings(path)
sim_info.ReadSpecFile()


