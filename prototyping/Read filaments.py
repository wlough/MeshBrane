# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:58:54 2021

@author: Daniel
"""
import numpy as np
class ReadCGLASSTubesRigid():
    def __init__(self,path):
        self.headerDT=np.dtype([
                ('number of steps', 'i4'),
                ('Steps between posit entries', 'i4'),
                ('time step', 'f8'),
                ])
        
        self.paths=[]
        self.base=path
        self.paths.append(path+'_rigid_filament_rig1.spec')
        self.paths.append(path+'_rigid_filament_rig2.spec')
        self.files=[]
        self.files.append(open(self.paths[0]))
        self.files.append(open(self.paths[1]))
        self.header = np.fromfile(self.files[0], dtype=self.headerDT, count=1)[0]
        self.header = np.fromfile(self.files[1], dtype=self.headerDT, count=1)[0]
        self.frames=int(self.header['number of steps']/self.header['Steps between posit entries']/2)
        print("Simulation has", self.frames, "frames")

        #print(self.header)

    def ReadSpecFile(self):
    
    
        
        'filament_positions'
        filament_list=[]
        
        rec_steps=int(self.header['number of steps']/self.header['Steps between posit entries'])

        self.tubeDT=np.dtype([
                ('name','S4'),
                ('diameter', 'f8'),
                ('length', 'f8'),
                ('segment length', 'f8'),
                ('number of sites', 'i4'),
                
                ])
        self.siteDT=np.dtype([
                ('position', 'f8',(3))                
                ])
        
        

        self.length_array=[]        
        self.fil_number=[]
        for t in range(self.frames):
            'tubes=number of tubes'
            #self.tubes=np.fromfile(self.file, dtype='i4', count=1)[0]
            self.tubes=2
            #print(self.tubes)
            filament_list.append([])
            self.fil_number.append([])
            for i,path in enumerate(self.files):
                tube_info=np.fromfile(path, dtype=self.tubeDT, count=1)[0]
                #print(tube_info)
                self.length_array.append(tube_info['segment length'])
                self.fil_number[t].append(tube_info['number of sites'])



                pos=[0,0,0]
                for j in range(tube_info['number of sites']):
                    site_info=np.fromfile(path, dtype=self.siteDT, count=1)[0]                
                    for dim in range(3):
                        pos[dim]+=site_info['position'][dim]
                filament_list[t].append(pos/tube_info['number of sites'])
                    
                    
                #p_length=np.fromfile(self.file, dtype='f8', count=1)[0]

                #pol_state=np.fromfile(self.file, dtype='i1', count=1)[0]

                
        'filament_list[time frame][tube number][dimension]'
        save_name=self.base+'_processed_filaments'
        np.save(save_name,filament_list);
        return filament_list
       
path='.\\Example\\example'
sim_info=ReadCGLASSTubesRigid(path)
sim_info.ReadSpecFile()