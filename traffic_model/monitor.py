import csv
import os
import re
from typing import List
import importlib

from traffic_model.data_class import  data

# module = importlib.import_module('/')

class Monitor:

    def __init__(self, data):
        print('[MONITOR]: Initializing...\n', end='')
        self.time=0
        self.data = data
        self.data_all={} # Store data of each ego
        self.statistical={} # Store statistics
        # self.surr=data.surr
        # self.IDC=data.IDC


        
        for ego in self.data['ctrl_vehs_name']:
            self.data_all[ego]=[]
            self.statistical[ego]=[]

        self.save_fold = './projects/data'  # ./Projects/project_name/current_time
        self.path = './projects/data'
        self.reset_count = 0 # rest number of times
        print('Done')



    def reset(self):
        '''
        Restart after one call
        '''

        self.stop() # Save the unsaved data from memory
        # initialization
        self.reset_count += 1
        self.path=self.save_fold + '_reset_' + str(self.reset_count)
        self.creat_file(self.path)
        self.time=0
        self.data_all={}
        self.statistical={}
        for ego in self.data.ego_veh_name:
            self.data_all[ego]=[]
            self.statistical[ego]=[]
        
    def stop(self):
        '''
        Stop operation,save the unsaved data from memory
        '''
        ego_num=0
        for ego in self.data.ego_veh_name:
            ego_num+=1
            if self.data_all[ego]:
                path=self.set_path(self.path+'/ego_'+str(ego_num)+'.csv')
                f = open(path,'a+',newline='')
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.set_title())
                csv_writer.writerows(self.data_all[ego])
                f.close()
                path=self.set_path(self.path+'/statistics_ego_'+str(ego_num)+'.csv')
                f = open(path,'a+',newline='')
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.statistical_title())
                csv_writer.writerows(self.statistical[ego])
                f.close()



    def creat_file(self,file: str):
        '''
        create folder
        '''
        if not os.path.exists(file):
            os.makedirs(file)

    def set_path(self,oldPath: str):
        '''
        Check whether the CSV file has the same name.
        If it has the same name, the file name will increase automatically
        '''

        path=oldPath
        directory, file_name = os.path.split(path)
        while os.path.isfile(path):
            if re.search('(\d+)\)\.', file_name) is None:
                file_name = file_name.replace('.', '(1).')
            else:
                current_number = int(re.findall('(\d+)\)\.', file_name)[-1])
                new_number = current_number + 1
                file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
            path = os.path.join(directory + os.sep + file_name)
        return path

    def save_data(self):
        '''
        Save every 1000 pieces of data
        '''
        self.time+=1
        if self.time==1:
            self.creat_file(self.path)
        ego_num=0#每个自车为一个csv文件，命名方式：ego_1, ego_2,…
        #每1000条数据保存为一个csv文件
        for ego in self.data['ctrl_vehs_name']:
            ego_num+=1
            save_interval=500 #1000
            if self.time %save_interval == 0:
                self.data_all[ego].append(self.set_data(ego))
                # self.statistical[ego].append(self.statistical_data(ego))
                #Save data
                path=self.set_path(self.path+'/ego_'+str(ego_num)+'.csv')
                f = open(path,'a+',newline='')
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.set_title())
                csv_writer.writerows(self.data_all[ego])
                f.close()
                self.data_all[ego]=[]
                #Save statistics
                path=self.set_path(self.path+'/statistics_ego_'+str(ego_num)+'.csv')
                f = open(path,'a+',newline='')
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.statistical_title())
                # csv_writer.writerows(self.statistical[ego])
                f.close()
                self.statistical[ego]=[]
            else:
                self.data_all[ego].append(self.set_data(ego))
                # self.statistical[ego].append(self.statistical_data(ego))



    def set_title(self) -> List[str]:
        '''
        Used to set the title
        '''

        title = [
            'time',
            'x',
            'y',
            'phi',
            'u',
            'v',
            'w',
            'acc_lon',
            'acc_lat',
            'throttle',
            'brake',
            'ste_wheel',
            'acce_des',
            'theta_des',
            'static_path_num',
            'static_path',
            'path_index',
            'behav_set',
            'behav_index',
            'planned_traj',
            'eva_risk',
            'act_collicion',
            'pred_collision',
            'eva_energy',
            'eva_comfort',
            'eva_efficiency',
            'eva_compliance',
            'lane_index',
            'lateral_position'
        ]
        print("="*100,'\n')
        print(len(title))

        return title


    def set_data(self,ego_name: str) -> List[str]:
        '''
        Used to set data
        '''

        static_path=[]
        for nda in self.data.agents[ego_name].static_paths:
            static_path.append(nda.tolist())#转成list
        path_index=self.data.agents[ego_name].path_index
        lane_index=self.data.agents[ego_name].lane_index
        # path_index_selected=path_index + 1 if lane_index==2 else path_index
        sorted_paths=sorted(static_path,key=lambda x:x[1])
        # print('sorted_paths:',sorted_paths)
        print('path_index_selected:', path_index)
        y=self.data.agents[ego_name].y
        lane_index_apply=lane_index if lane_index<2 else lane_index-1

        # y_ref_center=sorted_paths[lane_index_apply][1][0]
        y_ref_center=[-8,-4.8,-1.6][lane_index]

        lateral_position=y-y_ref_center
        saved_data = [
            self.data.simu_time / 1000,
            # Vehicle State, Decision and Control,
            self.data.agents[ego_name].x,
            self.data.agents[ego_name].y,
            self.data.agents[ego_name].phi,
            self.data.agents[ego_name].u,
            self.data.agents[ego_name].v,
            self.data.agents[ego_name].w,
            self.data.agents[ego_name].acc_lon,
            self.data.agents[ego_name].acc_lat,


            self.data.agents[ego_name].throttle,
            self.data.agents[ego_name].brake,
            self.data.agents[ego_name].ste_wheel,
            self.data.agents[ego_name].acce_des[0],
            self.data.agents[ego_name].theta_des[0],

            self.data.agents[ego_name].static_path_num,#static_path_num
            sorted_paths,
            path_index,#path_index
            self.data.agents[ego_name].behav_set,#behav_set
            self.data.agents[ego_name].behav_index,#behav_index
            self.data.agents[ego_name].planned_traj,#planned_traj
            #Evaluation Results
            self.data.agents[ego_name].eva_risk,
            self.data.agents[ego_name].act_collicion,
            self.data.agents[ego_name].pred_collision,
            self.data.agents[ego_name].eva_energy,
            self.data.agents[ego_name].eva_comfort,
            self.data.agents[ego_name].eva_efficiency,
            self.data.agents[ego_name].eva_compliance,
            self.data.agents[ego_name].lane_index,
            self.data.agents[ego_name].lateral_position

        ]


        return saved_data




    def statistical_title(self) -> List[str]:
        title =['time',
                #Evaluation Results
                'eva_risk',
                'act_collicion',
                'pred_collision',
                'eva_energy',
                'eva_comfort',
                'eva_efficiency',
                'eva_compliance'

        ]
        return title

    def statistical_data(self,ego_name: str) -> List[str]:
        '''
        Used to set statistical data
        '''
        data=[self.data.simu_time / 1000,
              #Evaluation Results
              self.data.agent[ego_name].eva_risk,
              self.data.agent[ego_name].act_collicion,
              self.data.agent[ego_name].pred_collision,
              self.data.agent[ego_name].eva_energy,
              self.data.agent[ego_name].eva_comfort,
              self.data.agent[ego_name].eva_efficiency,
              self.data.agent[ego_name].eva_compliance

        ]
        return data

if __name__=='__main__':
    mon=Monitor(data)
    title=mon.set_title()
    data=mon.set_data('ego')
    print(len(data))
    # print(title)
