import pickle
import grpc,copy,random,math,time,os,csv
import numpy as np
from traffic_model.podar import PODAR
from traffic_model.utils.ccbf import ccbf_controller, CCBFOption, VehicleSpec
from risenlighten.lasvsim.process_task.api.cosim.v1 import simulation_pb2
from risenlighten.lasvsim.process_task.api.cosim.v1 import simulation_pb2_grpc
from dataclasses import dataclass

model_params= dict(
    target_v=15,           #期望速度
    bias_lat_miu=0,        #纵向位置偏差均值
    bias_lat_var=0.2,      #纵向位置偏差方差
    bias_v_miu=0,        #纵向速度偏差均值
    bias_v_var=0.5,      #纵向速度偏差方差
    driver_nou=1    #random.randint(1,10)*0.1,          #驾驶员激进系数（增大），期望速度（变大），期望间距（变小），转向期望间距（变小）
)               #6个参数

abnormal_behv_str={1:'加塞',2:'向右超车',3:'连续换道',4:'压线行驶',5:'骑线行驶',
                   6:'变速行驶',7:'急刹车',8:'快车道慢速行驶',9:'快车道停车',10:'慢车道高速行驶',
                   11:'车道内倒车',12:'蛇形驾驶',13:'龟速驾驶'}

driver_delay=min(1/model_params['driver_nou'],3),  #驾驶员反应时间


#控制车辆类
class Surr(object):
    def __init__(self,id,vehicle):
        self.id=id
        self.d=3.2
        self.abnormal_status= {}  #{'i_beha':2,'status':0},status:0，表示符合触发条件，1，分步行为第一步完成，2，表示异常行为完成
        self.lane_change_time,self.lane_change_times,self.LC_desir_time,self.v_A,self.J_max_id,self.step_tag,self.abnormal_run_time,self.i_beha=0,0,0,0,0,0,0,0
        self.Length,self.Width,self.Weight=vehicle['Length'],vehicle['Width'],vehicle['Weight']     #传vehicle信息，改车的长宽质量
        self.max_dec,self.max_acc,self.x,self.y,self.phi,self.u,self.acc_lon,self.lane_rad,self.dis_to_link_end,self.selected_path_id=vehicle['max_dec'],vehicle['max_acc'],0,0,0,0,0,0,0,0
        self.static_path,self.all_veh_info,self.beha_list=[],[],[]
        self.matched_static_path,self.matched_static_path_array,self.matched_static_path_veh,self.xy_lane,self.ego,self.i_beha_condition_info = {},{},{},{},{},{}
        self.lane_id,self.old_lane_id,self.ego_info= '','',''
        # self.link_id=vehicle['link_id']     #认为车辆link不变,后续需要修改成变化的link
        self.ccbf_controller = ccbf_controller.CCBF_Controller(VehicleSpec())
        (P1, P2, P3, P4, P5, P6) = CCBFOption.weight
        self.beha_done,self.beha_stop=False,False
        self.ccbf_controller.update_lyapunov_parameter(P1, P2, P3, P4, P5, P6)
        self.bias_lat = random.gauss(model_params['bias_lat_miu'], model_params['bias_lat_var'])  # 驾驶道路中心线横向偏差
        self.bias_v = random.gauss(model_params['bias_v_miu'], model_params['bias_v_var'])  # 驾驶道路速度偏差
        self.DT = 0.05  # todo 调试控制代码

        self.controller = ccbf_controller.CCBF_Controller(dt=self.DT, b_lp_filter=False)
        self.controller.BuildCCBF()  # initialization

    def update(self, all_veh_info:list, vehicle:object, nearest_veh_id_around:list, static_paths, i_beha=0,
               case_info=None):
        '''

        :return:
        '''
        if case_info is None:
            case_info = dict(test_veh_id='ego')
        self.all_veh_info = all_veh_info
        self.lane_change_time+=1
        self.x=vehicle.moving_info.position.point.x
        self.y=vehicle.moving_info.position.point.y
        self.phi=vehicle.moving_info.position.phi
        self.u=vehicle.moving_info.u
        self.acc_lon=vehicle.moving_info.lon_acc
        self.lane_id=vehicle.moving_info.position.lane_id
        self.link_id=vehicle.moving_info.position.link_id
        self.dis_to_link_end=vehicle.moving_info.position.dis_to_link_end
        print(f'x:{self.x:.2f},y:{self.y:.2f},u:{self.u:.2f},phi:{self.phi:.4f},cl_time:{self.lane_change_time},lane_id:{self.lane_id},dis_to_end:{self.dis_to_link_end:.2f}')
        self.static_path=static_paths
        self.static_path_match_new()
        self.lane_rad=self.cul_lane_rad()  #todo 用道路中心线计算
        if self.id!=case_info['test_veh_id']:
            self.ego_info = list(filter(lambda d: d.id == 'ego', self.all_veh_info))[0]
        vehs_relation=self.find_vehs(nearest_veh_id_around)
        if i_beha==0:
            if 'i_beha' in case_info:
                if case_info['i_beha']!=0:
                    eval_value = {}
                    if self.id == case_info['test_veh_id']:
                        self.selected_path_id=1
                    elif case_info['is_ego_leader']:
                        self.selected_path_id=int(self.lane_id[-1])
            elif len(self.beha_list)!=0:self.selected_path_id=int(self.lane_id[-1])     #执行完异常行为后继续
            else:eval_value=self.eval_function()
        else:
            self.i_beha=i_beha
            self.get_path_id(i_beha)
            eval_value={}
            self.abnormal_status=dict(i_beha=i_beha,status=self.step_tag,beha_done=self.beha_done,beha_stop=self.beha_stop)
            if self.step_tag == 2:
                self.beha_list.append(i_beha)
            print(f"行为执行时间：{self.abnormal_run_time},异常行为：{abnormal_behv_str[self.abnormal_status['i_beha']]},选择路径：{self.selected_path_id}"
                  f"状态：{self.abnormal_status['status']},完成情况：{self.abnormal_status['beha_done'] if not self.abnormal_status['beha_done'] else self.abnormal_status['beha_done']}")
        if self.selected_path_id not in list(self.matched_static_path.keys()):
            self.selected_path_id = list(self.matched_static_path.keys())[0]
        ref_path=self.matched_static_path_array[self.selected_path_id]
        self.v_A,y_bias = self.get_track_info(i_beha,case_info)
        new_ref_path = self.gen_new_static_path(ref_path, target_v=self.v_A,target_phi=0,y_bias=y_bias)
        print(f"new_ref_path_x:{new_ref_path[0][0]},new_ref_path_y:{new_ref_path[1][0]},静态路径数量：{len(self.static_path.data)}")
        control_value=self.ccbf_controler(new_ref_path) if len(new_ref_path[0])>2 else [[0],[0]]
        self.reset()
        return {'control_value':control_value,'eval_value':eval_value,'vehs_relation':vehs_relation,'abnormal_status':self.abnormal_status}



    def reset(self):
        self.matched_static_path.clear()
        self.matched_static_path_array.clear()
        self.matched_static_path_veh.clear()
        self.beha_done = False
        if self.step_tag == 2:
            self.step_tag = 0
            self.J_max_id = int(self.lane_id[-1])

    def construct_cpath(self):
        #当前所在车道号，静态路径对应车道号，补充当前车道号静态路径
        spath_idx=list(self.matched_static_path_array.keys())[0]
        if spath_idx>int(self.lane_id[-1]):
            self.matched_static_path_array[int(self.lane_id[-1])] = [np.array([p - 3.2 * math.sin(self.lane_rad) for p in item]) if key == 0 else np.array([p + 3.2 * math.cos(self.lane_rad) for p in item])
                                                            for key,item in enumerate(list(self.matched_static_path_array.values())[0])]
        elif spath_idx<int(self.lane_id[-1]):
            self.matched_static_path_array[int(self.lane_id[-1])] = [np.array([p + 3.2 * math.sin(self.lane_rad) for p in item]) if key == 0 else np.array([p - 3.2 * math.cos(self.lane_rad) for p in item])
                                                            for key, item in enumerate(list(self.matched_static_path_array.values())[0])]
        else:
            if spath_idx==3:self.matched_static_path_array[2] = [np.array([p - 3.2 * math.sin(self.lane_rad) for p in item]) if key == 0 else np.array([p + 3.2 * math.cos(self.lane_rad) for p in item])
                                                            for key,item in enumerate(list(self.matched_static_path_array.values())[0])]
            if spath_idx==1:self.matched_static_path_array[2] = [np.array([p + 3.2 * math.sin(self.lane_rad) for p in item]) if key == 0 else np.array([p - 3.2 * math.cos(self.lane_rad) for p in item])
                                                            for key, item in enumerate(list(self.matched_static_path_array.values())[0])]
    def get_path_id(self,i_beha):
        self.abnormal_run_time+=1
        condition1,condition2=False,False
        ego_dis_to_link = self.ego_info.moving_info.position.dis_to_link_end
        ego_lane_id = self.ego_info.moving_info.position.lane_id
        ego_u = self.ego_info.moving_info.u
        t_define=2 if ego_u>10 else 1

        leader_list = [veh for veh in self.all_veh_info if veh.moving_info.position.lane_id == ego_lane_id and 0>veh.moving_info.position.dis_to_link_end - ego_dis_to_link > -30]
        if leader_list:
            leader = leader_list[max(range(len(leader_list)), key=lambda idx: leader_list[idx].moving_info.position.dis_to_link_end)]
            leader_dis_to_link = leader.moving_info.position.dis_to_link_end
            leader_Length=leader.base_info.Length
        else:leader_dis_to_link,leader_Length=ego_dis_to_link-100,5
        self.i_beha_condition_info['real_ego_dis_head'] = real_ego_dis_head = ego_dis_to_link - self.dis_to_link_end - 0.5 * (self.ego_info.base_info.Length - self.Length)
        self.i_beha_condition_info['real_ego_leader_dis']=real_ego_leader_dis=ego_dis_to_link-leader_dis_to_link- 0.5 * (leader_Length + self.ego_info.base_info.Length)
        self.i_beha_condition_info['real_ego_dis']=real_ego_dis=ego_dis_to_link-self.dis_to_link_end-0.5*(self.ego_info.base_info.Length+self.Length)
        if i_beha==1:#加塞
            if self.step_tag==0 and 0<real_ego_dis_head< self.Length:
                self.selected_path_id=int(ego_lane_id[-1])
                self.step_tag = 1
            if int(self.lane_id[-1]) == int(ego_lane_id[-1]) and self.step_tag==1:self.step_tag = 2
        if i_beha==2:#右侧超车:1.判断与前车距离在阈值范围内，前车速度低于0.5倍限速即40km/h 2.选择的路径为当前路径的右侧 3.换道到右侧后，加速，等换道到左侧的条件， 4.选择当前路径的左侧
            if self.step_tag==0:
                self.selected_path_id=int(ego_lane_id[-1])+1
            if int(self.lane_id[-1]) > int(ego_lane_id[-1]):
                self.step_tag = 1
            if self.step_tag==1:
                l_base=(ego_u-self.u)*t_define if ego_u>self.u else 0
                if real_ego_leader_dis>15:condition1 = True
                if real_ego_dis>l_base:condition2=True
                if condition1 and condition2:
                    self.selected_path_id=int(ego_lane_id[-1])
                    self.step_tag = 1.5
            if int(self.lane_id[-1])==int(ego_lane_id[-1]) and self.step_tag==1.5:
                self.step_tag=2
        if i_beha == 3:#连续换道
            if self.step_tag ==0:
                if int(self.lane_id[-1]) != int(ego_lane_id[-1]):
                    self.selected_path_id=int(ego_lane_id[-1])              #case1:第一步到ego所在车道
                    self.step_tag = 1.2
                else:                                                       #case2:第一步到ego相邻车道
                    if int(ego_lane_id[-1]) !=1:
                        self.selected_path_id = int(ego_lane_id[-1]) - 1
                    elif int(ego_lane_id[-1]) !=3:
                        self.selected_path_id = int(ego_lane_id[-1]) + 1
                    else:
                        self.selected_path_id = int(ego_lane_id[-1]) + random.choice([-1,1])
                    self.step_tag = 1.3

            if self.step_tag == 1.2 and int(self.lane_id[-1])==int(ego_lane_id[-1]):        #case1:第二步到ego相邻车道
                self.selected_path_id = int(ego_lane_id[-1]) + random.choice([-1,1])
                self.step_tag = 1.5

            if self.step_tag==1.5 and int(self.lane_id[-1])!=int(ego_lane_id[-1]):
                self.step_tag = 2

            if self.step_tag == 1.3 and int(self.lane_id[-1])!=int(ego_lane_id[-1]):        #case2:第二步回到ego所在车道
                self.selected_path_id = int(ego_lane_id[-1])
                self.step_tag = 1.6

            if self.step_tag==1.6 and int(self.lane_id[-1])==int(ego_lane_id[-1]):
                self.step_tag = 2#

        if i_beha==4:#压线行驶
            if self.step_tag ==0:
                self.selected_path_id=int(ego_lane_id[-1])
                self.step_tag = 1
        if i_beha==5:#骑线行驶
            if self.step_tag == 0:
                self.selected_path_id = int(self.lane_id[-1])
                self.step_tag = 1
        if i_beha==6:#变速行驶
            if self.step_tag == 0:
                self.selected_path_id = int(ego_lane_id[-1])  # case1:第一步到ego所在车道
                self.step_tag = 1
        if i_beha==7:
            ...
        if i_beha==8:
            ...
        if i_beha==9:
            ...
        if i_beha==10:
            ...
        if i_beha==11:
            ...
        if i_beha==12:
            ...
        if i_beha==13:
            ...


        if self.step_tag==2:
            self.abnormal_run_time=0
            self.beha_done=True
            self.selected_path_id = int(self.lane_id[-1])

        if  (self.step_tag>=1 and self.abnormal_run_time>100) or \
                abs(self.selected_path_id-int(self.lane_id[-1]))>1 or \
                self.selected_path_id not in list(self.matched_static_path.keys()):
            self.selected_path_id = int(self.lane_id[-1])
            self.abnormal_run_time = 0
            self.beha_stop = True

    def get_track_info(self,i_beha,case_info):#todo 留速度，加一个速度偏置
        v_static_path = model_params['target_v'] -0.1*model_params['target_v']*(1-model_params['driver_nou'])
        v_A = v_static_path + self.bias_v
        y_bias=0
        w_road = 3.2
        if self.id!=case_info['test_veh_id']:ego_lane_id = self.ego_info.moving_info.position.lane_id

        if self.dis_to_link_end<30: v_A = max(((self.dis_to_link_end-5)  / 30) * v_A,0)
        if 'i_beha' in case_info:
            if case_info['i_beha']!=0:

                if i_beha==1:       #控制待控车速度
                    v_A=3
                if i_beha==4:#压线行驶
                    y_bias=random.choices([-1,1])*(0.5*w_road-0.5*self.Width+round(random.uniform(0,0.25),2))
                if i_beha==5:#骑线行驶
                    y_bias = np.sign(int(self.lane_id[-1])-int(ego_lane_id[-1]))* \
                        (0.5 * w_road - 0.5 * self.Width + round(random.uniform(0.25, 0.5 * self.Width), 2))

            if case_info['i_beha'] != 0:
                if case_info['is_ego_leader'] and case_info['v_leader']: v_A = case_info['v_leader']  # 控制前车速度
                if self.id == case_info['test_veh_id'] and case_info['v_ego']: v_A = case_info['v_ego']  # 控制ego车速度

        print('v_static_path={:<2.2f} bias_v={:<2.2f} v_A={:<2.2f}'.format(v_static_path, self.bias_v, v_A))
        return v_A,y_bias




    def cul_lane_rad(self):
        waypoints = list(self.matched_static_path.values())[0]
        print(f'waypoins长度：{len(waypoints)}，静态路径集：{list(self.matched_static_path.keys())}')
        cosest_idx = min(range(len(waypoints)), key=lambda closest_idx: math.sqrt(
            (waypoints[closest_idx].x - self.x) ** 2 + (waypoints[closest_idx].y - self.y) ** 2))
        waypoints = waypoints[cosest_idx:cosest_idx + 41]
        try:
            lane_rad = math.atan((waypoints[2].y - waypoints[0].y) / (waypoints[2].x - waypoints[0].x))
            lane_rad2deg = lane_rad * 180 / math.pi
            if waypoints[2].y - waypoints[0].y > 0:
                up = True
            else:
                up = False
            if waypoints[2].x - waypoints[0].x > 0:
                right = True
            else:
                right = False
            if up and not right: lane_rad = lane_rad + math.pi
            if not up and not right: lane_rad = lane_rad + math.pi
            if not up and right: lane_rad = lane_rad + math.pi * 2
            return lane_rad # unit is rad
        except:return self.lane_rad # unit is rad


    def find_vehs(self,vehs_list):
        #1.通过静态路径中（x,y）和车辆（x,y）距离最小判断车辆属于哪条静态路径
        nearest_vehs=[self.get_veh_info_from_traffic(veh_id) for veh_id in vehs_list]
        vehs_d = dict(v1l=[],v2l=[],v3l=[],v1f=[],v2f=[],v3f=[])
        self.get_xy_translane()
        def get_d(idx):
            x0=self.xy_lane[idx]['x']
            y0=self.xy_lane[idx]['y']
            b=y0-np.tan(self.lane_rad)*x0
            return abs(-np.tan(self.lane_rad)*veh['x']+veh['y']-b)/np.sqrt(np.tan(self.lane_rad)**2+1)  #点到直线距离
        for veh in nearest_vehs:
            #todo 不能使用自车的静态路径keys
            idx_=int(veh['lane_id'][-1])
            if idx_==int(self.lane_id[-1]):
                idx = min(list(self.xy_lane.keys()), key=lambda idx: get_d(idx))
            # elif get_d(int(self.lane_id[-1]))<1.8 and idx_!=int(self.lane_id[-1]):idx=int(self.lane_id[-1])
            else:idx=idx_

            vehs_l,vehs_f=vehs_d[f'v{idx}l'],vehs_d[f'v{idx}f']
            if veh['dis_to_link'] >self.dis_to_link_end:exec('vehs.append(value)',{'vehs':vehs_f,'value':veh})
            else:exec('vehs.append(value)',{'vehs':vehs_l,'value':veh})
            vehs_l, vehs_f = [], []
        for key,vehs in vehs_d.items():
            vehs_d[key]=order_vehicles(self,vehs) if vehs else []
        self.matched_static_path_veh={1:[vehs_d['v1l'],vehs_d['v1f']],2:[vehs_d['v2l'],vehs_d['v2f']],3:[vehs_d['v3l'],vehs_d['v3f']]}
        print(f'车道1:{self.matched_static_path_veh[1]}\n车道2:{self.matched_static_path_veh[2]}\n车道3:{self.matched_static_path_veh[3]}')
        return vehs_d
    def select_leader(self):
        #1.当前车道和其他车道的前车进行距离比较，2.其他车道的前车如果abs(phi)>0.2，将该前车放到候选列表，3.将候选列表中的前车与当前车道前车比较，选择离自车最近的
        veh_list=[]
        phi_base=0.08
        lat_base=1.9
        leader=self.matched_static_path_veh[int(self.lane_id[-1])][0]
        leader_candi=[item[0] for key,item in self.matched_static_path_veh.items() if key != int(self.lane_id[-1]) ]
        for veh in leader_candi:
            if isinstance(veh, list):continue
            delta_phi = np.sign(int(self.lane_id[-1])-int(veh['lane_id'][-1]))*(self.phi-veh['phi'])
            if isinstance(leader,dict):
                condition1=veh['dis_to_link'] > leader['dis_to_link']
                condition2= abs(veh['y']-self.y)<1.9 and abs(veh['phi']) > phi_base
                condition3=abs(veh['y']-self.y) < 1.5
                if (condition1 and condition2) or (condition1 and condition3):veh_list.append(veh)
            else:
                if abs(veh['y']-self.y)<1.9 and delta_phi > phi_base:veh_list.append(veh)
        if len(veh_list)>=1:
            max_item=max(veh_list,key=lambda x:x['dis_to_link'])
            print('找到前车',max_item['id'])
            return max_item
        else:return leader


    def static_path_match(self,y_list):
        static_path=copy.deepcopy(self.static_path[0].point)
        closest_idx = min(range(len(static_path)), key=lambda closest_idx: abs(static_path[closest_idx].x - self.x))
        for i,y in enumerate(y_list):
            idx=min(range(len(self.static_path)), key=lambda index: abs(self.static_path[index].point[closest_idx].y - y))
            if (self.lane_id[-1]=='1' and i==2) or (self.lane_id[-1]=='3' and i==0):continue #1号车道时，不需要评估3号车道路径，3号车道时，不需要1号车道路径
            self.matched_static_path[i + 1] = self.static_path[idx].point[closest_idx:closest_idx+min(41,len(self.static_path[idx].point)-closest_idx)]
            self.matched_static_path_array[i+1] =self.out_static_path(self.matched_static_path[i + 1])

    def static_path_match_new(self):
        #1.静态路径的point，2.循环取每条路径的path_id,将id最后1位转为Int，3.构建字典
        static_path = copy.deepcopy(self.static_path.data[0].lines[0].Points)
        closest_idx = min(range(len(static_path)), key=lambda closest_idx: abs(static_path[closest_idx].x - self.x))
        for path_tuple in [(item.lines[0].Points, item.lines[0].path_id) for item in self.static_path.data]:
            if (self.lane_id[-1] == '1' and path_tuple[1][-1]== '3') or (
                        self.lane_id[-1] == '3' and path_tuple[1][-1] == '1'): continue  # 1号车道时，不需要评估3号车道路径，3号车道时，不需要1号车道路径
            self.matched_static_path[int(path_tuple[1][-1])] = path_tuple[0][closest_idx:closest_idx + min(41, len(
                    path_tuple[0]) - closest_idx)]
            self.matched_static_path_array[int(path_tuple[1][-1])] = self.out_static_path(self.matched_static_path[int(path_tuple[1][-1])])

    def out_static_path(self,path_raw):
        path,x_list,y_list=[[],[]],[],[]
        x_list = np.array([item.x for item in path_raw])
        y_list = np.array([item.y for item in path_raw])
        path[0],path[1]=x_list,y_list
        return path
    def cul_real_dis(self,vehs_relation,lane_idx:int,lof:int):
        car_leader = vehs_relation[lane_idx][lof] if vehs_relation[lane_idx][lof] else []
        if car_leader:return abs(car_leader['dis_to_link']-self.dis_to_link_end)-0.5*(self.Length+car_leader['Length'])
        else:return 100

    def eval_function(self):
        J_value,val_safe_d,val_traffic_d={},{},{}
        safe_base=0.8
        if len(self.matched_static_path_array) == 1:self.construct_cpath()
        def is_LC_complete():
            if self.J_max_id != int(self.lane_id[-1]) and self.old_lane_id == '': self.old_lane_id = int( self.lane_id[-1])  # 有换道动机
            if int(self.lane_id[-1])!=self.old_lane_id and isinstance(self.old_lane_id,int):#实际换道后时间清零
                self.lane_change_time = 0
                self.lane_change_times += 1
                self.old_lane_id = ''
        def LC_safe_judge():
            for idx, path in self.matched_static_path_array.items():  # 风险评估
                val_safe_d[idx] = self.safe_evaluation(idx, self.matched_static_path_veh[idx])
            if val_safe_d[self.J_max_id]<safe_base and isinstance(self.old_lane_id,int):self.selected_path_id=self.old_lane_id #换道不安全，不换道

        is_LC_complete()  # 是否换道
        if self.lane_change_times==0 or self.lane_change_time>15 :#换道后，time置为0，不评估静态路径，每次time累加1，到20以后再评估静态路径
            for idx, path in self.matched_static_path_array.items():  # self.cul_real_dis可以计算与前车的实际距离
                val_safe = self.safe_evaluation(idx, self.matched_static_path_veh[idx])
                val_traffic = self.traffic_evaluation(self.matched_static_path_veh[idx])    #todo safe,<1时都归零，
                J_value[idx], val_safe_d[idx], val_traffic_d[idx] = val_safe + val_traffic, val_safe, val_traffic
                print(f'idx:{idx},val_safe:{val_safe},val_traffic:{val_traffic}')   #todo condition,加一个累积时间1s,以后考虑相同激进系数，不同类别阈值
            J_max_id = max(list(J_value.keys()), key=lambda index: J_value[index])
            self.J_max_id=J_max_id
            e_delta=(1-model_params['driver_nou'])*0.15+0.15    #激进驾驶员也需要添加阈值
            # assert len(J_value)>=2,f'评估值{J_value}'

            if (J_value[J_max_id] - J_value[int(self.lane_id[-1])]) / J_value[int(self.lane_id[-1])] > e_delta :#0.3，换道效益提高阈值
                if self.lane_change_times == 0: self.selected_path_id = J_max_id
                else:self.LC_desir_time+=1
                if self.LC_desir_time>5:
                    self.selected_path_id = J_max_id    #10
                    self.LC_desir_time = 0
            else:
                # if J_max_id !=int(self.lane_id[-1]):
                self.LC_desir_time = 0
                if self.lane_change_times == 0: self.selected_path_id = int(self.lane_id[-1])
        else:LC_safe_judge()


        print(f'评估函数值{J_value},安全评估值：{val_safe_d}选择静态路径索引{self.selected_path_id},lane_change_time:{self.lane_change_time},换道次数：{self.lane_change_times},效益提升累计时间：{self.LC_desir_time}')
        return {'J_value':J_value,'val_safe':val_safe_d,'val_traffic':val_traffic_d,'lane_change_times':self.lane_change_times}

    def get_veh_info_from_traffic(self,veh_id)->dict:
        vehs_same_link = [item for item in self.all_veh_info if item.moving_info.position.link_id == self.link_id]
        assert len(list(filter(lambda d: d.id == veh_id, vehs_same_link)))==1,'{},{},{}'.format(veh_id,len(vehs_same_link),list(filter(lambda d: d.id == veh_id, vehs_same_link)))
        veh_info = list(filter(lambda d: d.id == veh_id, vehs_same_link))[0]
        x=veh_info.moving_info.position.point.x
        y=veh_info.moving_info.position.point.y
        phi=veh_info.moving_info.position.phi
        u=veh_info.moving_info.u
        lon_acc=veh_info.moving_info.lon_acc
        lane_id=veh_info.moving_info.position.lane_id
        dis_to_link=veh_info.moving_info.position.dis_to_link_end
        Length=veh_info.base_info.Length
        Width=veh_info.base_info.Width
        junction_id=veh_info.moving_info.position.junction_id
        return dict(id=veh_id,x=x,y=y,lane_id=lane_id,dis_to_link=dis_to_link,Length=Length,Width=Width,phi= phi,u=u,lon_acc= lon_acc,junction_id= junction_id)


    def gen_new_static_path(self, path, target_v, target_phi,y_bias):#todo 加一个横向偏置
        path.append(target_phi * np.ones(41))
        path.append(target_v * np.ones(41))
        path[1]=path[1]+(self.bias_lat if y_bias==0 else y_bias)
        print(f'bias_lat:{self.bias_lat:.3f}') if y_bias==0 else print(f'异常偏置:{y_bias:.3f}')
        return path

    def transfer_path(self,new_ref_path, lane_rad):
        x_list,y_list = new_ref_path[0],new_ref_path[1]
        x_list_new,y_list_new,tansfered_path = [],[],[]
        x_list_new, y_list_new = [world_to_self_car(x, y, self.x, self.y, lane_rad)[0] for x, y in
                zip(x_list, y_list)],[world_to_self_car(x, y, self.x, self.y, lane_rad)[1] for x, y in zip(x_list, y_list)]
        tansfered_path.append(x_list_new)
        tansfered_path.append(y_list_new)
        tansfered_path.append(new_ref_path[2])
        tansfered_path.append(new_ref_path[3])
        return tansfered_path
    #todo 在matlab中测试phi的值
    def ccbf_controler(self,static_paths):
        if abs(self.lane_rad - 2 * math.pi) * 180 / math.pi < 0.2 or abs(self.lane_rad * 180 / math.pi < 0.2):self.lane_rad = 0
        new_phi = self.phi - self.lane_rad
        print(f"new_phi:{new_phi:.2f},世界坐标phi:{self.phi:.2f},道路朝向:{self.lane_rad:.2f}")
        transfered_path = self.transfer_path(static_paths, self.lane_rad)
        print(f"new_path_x:{transfered_path[0][0]:.2f},new_path_y:{transfered_path[1][0]:.2f}")
        v,w = 0,0
        # =======================old controler=============================
        # self.ccbf_controller.update_values(0, 0, new_phi, self.u, v, w)
        # self.ccbf_controller.update_waypoints(transfered_path)
        # self.ccbf_controller.construct_clf_sep()

        #====================================================
        #4个障碍物，可以是静止，也可以是动态的
        obs_state = [50.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        obs_state2 = [90.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        obs_state3 = [130.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        obs_state4 = [170.0, -1.0, 0.0, 0.0, 0.0, 0.0]

        if (obs_state != None):
            obstacle = ccbf_controller.Obstacle(obs_state)
            self.controller.set_obstacle(obstacle)
        if (obs_state2 != None):
            obstacle2 = ccbf_controller.Obstacle(obs_state2)
            self.controller.set_obstacle(obstacle2)
        if (obs_state3 != None):
            obstacle3 = ccbf_controller.Obstacle(obs_state3)
            self.controller.set_obstacle(obstacle3)
        if (obs_state4 != None):
            obstacle4 = ccbf_controller.Obstacle(obs_state4)
            self.controller.set_obstacle(obstacle4)
        # ====================================================
        # update ego vehicle's current states
        self.controller.update_values(0, 0, new_phi,
                                 self.u, v, w, 0)

        # update reference point (x, y, phi)
        self.controller.update_reference_point(transfered_path[0],transfered_path[1],transfered_path[2])

        # update desired speed for ego vehicle
        self.controller.update_desired_speed(transfered_path[3])
        self.controller.get_control_law("ccbf")

        accel=self.controller.throttle  # acceleration
        steer=self.controller.steer  # steering angle
        # ====================================================
        # car_leader = self.select_leader()
        #
        # if car_leader:
        #     front_veh_x, front_veh_y=world_to_self_car(car_leader['x'], car_leader['y'], self.x, self.y, self.lane_rad)
        #     print('前车转换后坐标x:{:.2f},前车id:{}'.format(front_veh_x,car_leader['id']))
        #     front_veh_real_x=front_veh_x-0.5*(car_leader['Length']+self.Length)
        #     car_leader_u = car_leader['u']
        #     accel_cbf=self.ccbf_controller.construct_cbf_front_veh(front_veh_real_x,v_l=car_leader_u)   #前方有车时，控制刹车的控制量
        # else:
        #     # if self.dis_to_link_end < 10: accel_cbf = self.ccbf_controller.construct_cbf_front_veh(self.dis_to_link_end+2, v_l=0)
        #     # else:accel_cbf = self.ccbf_controller.construct_cbf_front_veh(50, v_l=self.v_A)
        #     front_veh_real_x,car_leader_u=50,30
        #     accel_cbf = self.ccbf_controller.construct_cbf_front_veh(front_veh_real_x, v_l=car_leader_u)
        # accel,steer = self.ccbf_controller.throttle,self.ccbf_controller.steer
        # ====================================================

        control_bound: tuple = ((-0.4, -self.max_dec), (0.4, self.max_acc))# lower bound -0.4（右转向22度）,-7 为车辆的一般减速度    # upper bound 0.4（左转向22度）， 2.5
        # accel,accel_cbf = np.clip(accel, control_bound[0][1], control_bound[1][1]),np.clip(accel_cbf, control_bound[0][1], control_bound[1][1])
        # assert isinstance(accel_cbf, np.ndarray), f'accel_cbf:{accel_cbf}'
        steer = np.clip(steer, control_bound[0][0], control_bound[1][0])
        assert isinstance(steer,np.ndarray),f'steer:{steer}'
        assert isinstance(accel, np.ndarray), f'accel:{accel}'

        # print('(CCBF) car following control, acc: {:.2f}, steer: {:.2f},acc_cbf:{:.2f},前车实际距离:{:.2f},前车速度:{:.2f},自车朝向:{:.4f}，自车速度:{:.2f},期望速度{:.2f}'.format(accel[0],steer[0],accel_cbf[0],front_veh_real_x,car_leader_u,self.phi,self.u,transfered_path[3][0]))

        print('(CCBF) car following control, acc: {:.2f}, steer: {:.2f},自车朝向:{:.4f}，自车速度:{:.2f},期望速度{:.2f}'.format(
                accel[0], steer[0], self.phi, self.u,transfered_path[3][0]))
        control_value = [accel, steer]
        return control_value

    def safe_evaluation(self, idx, vehs_relation):
        if vehs_relation:
            risk = self.eval_podar( idx, vehs_relation)
            safe_value = (2.3 - risk) / 2.3
        else:safe_value = 1
        return safe_value
    def get_xy_translane(self,idx=2):
        lane_idx=int(self.lane_id[-1])
        if lane_idx == 1:
            self.xy_lane[1] = {'x':self.x,'y': self.y}
            self.xy_lane[2] = {'x':self.x+self.d*math.sin(self.lane_rad),'y':self.y-self.d*math.cos(self.lane_rad)}
            self.xy_lane[3] = {'x': self.x + 2*self.d * math.sin(self.lane_rad),'y': self.y - 2*self.d * math.cos(self.lane_rad)}
        if lane_idx == 2:
            self.xy_lane[1] = {'x':self.x-self.d*math.sin(self.lane_rad),'y':self.y+self.d*math.cos(self.lane_rad)}
            self.xy_lane[2] = {'x':self.x,'y': self.y}
            self.xy_lane[3] = {'x':self.x+self.d*math.sin(self.lane_rad),'y':self.y-self.d*math.cos(self.lane_rad)}
        if lane_idx == 3:
            self.xy_lane[1] = {'x': self.x - 2*self.d * math.sin(self.lane_rad), 'y': self.y + 2*self.d * math.cos(self.lane_rad)}
            self.xy_lane[2] = {'x':self.x-self.d*math.sin(self.lane_rad),'y': self.y+self.d*math.cos(self.lane_rad)}
            self.xy_lane[3] = {'x':self.x,'y': self.y}
        return self.xy_lane[idx]
    def eval_podar(self, idx, ov_concern):
        l, w,ms1,md,x,y,u,phi,acc_lon= self.Length,self.Width,self.Weight,self.max_dec,self.x,self.y,self.u,self.phi,self.acc_lon
        podar = PODAR()
        surr_list = []
        for _ in ov_concern:
            if not _:
                continue
            surr_list.append(_['id'])
        podar.ego_surr_list = {self.id: surr_list}
        # [step 1] set static parameters (only once)
        podar.add_ego(name=self.id, length=l, width=w, mass=ms1, max_dece=md)
        # [step 2] set dynamic information
        xy=self.xy_lane[idx]
        print(f'xy:{xy},idx:{idx}')         #todo　平移时
        podar.update_ego(name=self.id, x0=xy['x'], y0=xy['y'], speed=u, phi0=phi, a0=acc_lon,length=(u / 22) * 10+l if idx!=int(self.lane_id[-1]) else l)
        risk = self.cul_risk( podar, ov_concern)
        return risk

    def cul_risk(self, podar, ov_concern):
        for _ in ov_concern:
            if not _:
                continue
            podar.add_obj(name=_['id'], x0=_['x'], y0=_['y'], speed=_['u'],phi0=_['phi'])  #a0=_['lon_acc'], length=_['Length'], width=_['Width']
        risk, if_real_collision, if_predicted_collision = podar.estimate_risk(self.id)
        risk = min(risk , 2.3)
        risk = risk if risk > 0.3 else 0
        podar.reset()
        return risk

    def traffic_evaluation(self, vehs_relation:list):
        v_d = model_params['target_v'] - 0.1 * model_params['target_v'] * (1 - model_params['driver_nou'])
        car_leader = vehs_relation[0]
        if car_leader:
            length_leader = car_leader['Length']
            dis = car_leader['dis_to_link']-self.dis_to_link_end
            real_dis = abs(dis) - (self.Length + length_leader) / 2
            v_l = car_leader['u']
            d_b = 30    # max(0.0001, 3.6 * v_l)
            def traffic_eval_method1():
                if real_dis > 5:
                    if v_l > d_b:v_d0 = v_d
                    else:v_d0 = min(0.5 * real_dis * v_l / v_d, v_d)
                    return round(v_d0, 2)
                else:return 0
            def traffic_eval_method2():
                if real_dis>3:
                    if v_l>d_b:return v_d
                    else:return min(0.5*real_dis/d_b*v_d+ 0.5*v_l, v_d)
                else:return 0
            v_d0=traffic_eval_method2()
            traffic_value = round(v_d0 / min(v_d,22), 2)
        else:traffic_value=1
        return traffic_value
@dataclass
class Cosim():
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjEzLCJvaWQiOjEwMSwibmFtZSI6IueGiuadsCIsImlkZW50aXR5Ijoibm9ybWFsIiwicGVybWlzc2lvbnMiOlsidGFzay50YXNrLnB1YmxpYy5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wdWJsaWMuQUNUSU9OX0NPUFkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBPUlQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9BREQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fQ09QWSIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fUkVQTEFZIiwidGFzay50YXNrLnByaXZhdGUuQUNUSU9OX1JFUE9SVCIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwidGFzay50YXNrLnBlcnNvbmFsLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucGVyc29uYWwuQUNUSU9OX1JFUE9SVCIsInJlc291cmNlLnZlaGljbGUucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudmVoaWNsZS5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudmVoaWNsZS5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS52ZWhpY2xlLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2UudmVoaWNsZS5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucHVibGljLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9BREQiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2Vuc29yLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnNlbnNvci5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnB1YmxpYy5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VQREFURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9WSUVXIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX0FERCIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS5tYXAucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UubWFwLnBlcnNvbmFsLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnNjZW5hcmlvLnByaXZhdGUuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wcml2YXRlLkFDVElPTl9DT1BZIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS50cmFmZmljX2Zsb3dfY29uZmlnLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9ET1dOTE9BRCIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RPV05MT0FEIl0sImlzcyI6InVzZXIiLCJzdWIiOiJMYXNWU2ltIiwiZXhwIjoxNzA0MDc4MjgwLCJuYmYiOjE3MDM0NzM0ODAsImlhdCI6MTcwMzQ3MzQ4MCwianRpIjoiMTMifQ.LdkObK3RUW50ZBfX0_jGpygOyzbrR3qcUykyXbw6A1g'
    metadata = [('authorization','Bearer ' + token)]
    http = 'qianxing-grpc.risenlighten.com:80'
    # http='127.0.0.1:8290'  #桌面端lasvsim连接grpc用
    max_recv_msg_size = 20 * 1024 * 1024  # 20M
    # 创建channelArguments对象来设置选项
    channel_args = [('grpc.max_receive_message_length', max_recv_msg_size), ('grpc.default_timeout', 100)]
    happended_beha_list=[]
    beha_group = {'A': [1], 'B': [2, 3, 4, 5], 'C': [6]}
    P_dict = {1: 0.0, 2: 0.9, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1,12: 0.1,13:0.1}
    task_id,record_id,vehicle_id = 5045,8059,'ego'
    vehicle_id_list = [vehicle_id]
    progress_times,save_time,data_num,i_beha = 0,0,0,0
    choosed_list,vehicle_id_list_new,veh_id_around,all_vehs_pos,allvehs_info,control_value,used_names_list,abnormal_veh_list = [],[],[],[],[],[],[],[]
    veh_info,agent,ctrl_value,ego_info,abnormal_veh_status,case_info,jam_vehs_l = {},{},{},{},{},{'test_veh_id':vehicle_id},{}
    old_file_path,simulation_id,leader_id = '','',''

    def run(self,num=300):
        with grpc.insecure_channel(self.http, options=self.channel_args) as channel:
            stub = simulation_pb2_grpc.CosimStub(channel)
            self.simulation_start(stub)
            file_path = self.built_file_dir(file_type='pkl')
            for i in range(num):
                start_time=time.time()
                print("=" * 100,'\n',f'i:{i}')
                result = self.step(stub)
                if result:break
                all_vehs_pos, allvehs_info = self.get_allvehs_position(stub)    #存为Pickle
                self.allvehs_info=allvehs_info
                with open(os.path.join(file_path, 'data.pkl'),'ab') as file:
                    pickle.dump({'i':i,'all_vehs_pos':all_vehs_pos,'allvehs_info':allvehs_info},file)
                if i > 0:
                    if not self.vehicle_id_list_new:#todo 找最近10辆控制车
                        ctrl_vehs = []
                        ctrl_l_list,ctrl_f_list = get_ctrl_vehs(all_vehs_pos)
                        self.vehicle_id_list_new = self.vehicle_id_list #+ ctrl_l_list + ctrl_f_list
                        for veh in self.vehicle_id_list_new:
                            vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(simulation_id=self.simulation_id, vehicle_id=veh), metadata=self.metadata)
                            veh_info=dict(Length=vehicle.vehicle.info.base_info.Length,Width=vehicle.vehicle.info.base_info.Width,Weight=vehicle.vehicle.info.base_info.Weight,
                                          max_dec=vehicle.vehicle.info.base_info.max_dec,max_acc=vehicle.vehicle.info.base_info.max_acc,link_id=vehicle.vehicle.info.moving_info.position.link_id)
                            ctrl_vehs.append(Surr(veh, veh_info))

                    def save_veh_values():
                        with open(os.path.join(file_path, 'ctrl_veh_values.pkl'), 'ab') as file:
                            if veh_values['abnormal_status']:
                                status=veh_values['abnormal_status']['status']
                                done=veh_values['abnormal_status']['beha_done']
                                stop = veh_values['abnormal_status']['beha_stop']
                            else:status,done=0,False
                            ctrl_veh_info={'i':i,'veh_id':veh.id,'ctrl_value':veh_values['control_value'],'lane_id':veh.lane_id,'dis_to_link':veh.dis_to_link_end,'u':veh.u,'phi':veh.phi,
                                           'x':veh.x,'y':veh.y,'eval_value':veh_values['eval_value'],'selected_id':veh.selected_path_id,'vehs_relation':veh_values['vehs_relation'],
                                           'bias_lat':veh.bias_lat,'bias_v':veh.bias_v,'step_tag':status,'i_beha':veh.i_beha,'beha_time':veh.abnormal_run_time,'done':done}
                            pickle.dump(ctrl_veh_info, file)
                        if veh.id in list(abnormal_veh_dict.keys()):
                            if done or stop:self.update_abnormal_veh_status(veh.id,veh.i_beha,beha_status={'beha_done':done,'beha_stop':stop})
                        print('id为{}，当前所在车道{}，dis_to_link:{:<3.2f}'.format(veh.id,veh.lane_id,veh.dis_to_link_end), '\n',"*" * 50)
                    def abnormal_beha_info():
                        if veh.id in list(abnormal_veh_dict.keys()) and self.abnormal_veh_status == {}:
                            self.i_beha = i_beha = abnormal_veh_dict[veh.id]  # 即将发生的异常行为
                            self.update_abnormal_veh_status(veh.id, i_beha, {'beha_done':False,'beha_stop':False})
                        elif self.abnormal_veh_status and veh.id == self.abnormal_veh_status['veh_id']:
                            self.i_beha = self.abnormal_veh_status['i_beha']
                        else:
                            self.i_beha = 0

                        self.case_info = self.construct_case()         #制造加塞场景
                        is_ego_leader = veh.id == self.leader_id
                        self.case_info['is_ego_leader'] = is_ego_leader
                        print('case_info:', self.case_info)


                    if not self.abnormal_veh_status:
                        beha_id_list=self.sampler_behavior() #and False #采样异常行为
                        if beha_id_list:
                            print(f'发生异常行为：{abnormal_behv_str[beha_id_list[0]]}')
                            abnormal_veh_dict=self.find_abnormal_veh(beha_id_list,all_vehs_pos)
                            print(f'发生异常行为的车辆为：{abnormal_veh_dict}')
                            if abnormal_veh_dict:   #todo 有新的异常行为类型发生，该类异常行为概率降一半,新的异常行为种类放入发生异常行为列表
                                i_list=list(abnormal_veh_dict.values())
                                new_i_list=[i for i in i_list if i not in self.happended_beha_list]
                                if new_i_list:
                                    for i in new_i_list:
                                        self.P_dict[i] = self.P_dict[i] / 2
                                    self.happended_beha_list=self.happended_beha_list+new_i_list
                            self.abnormal_veh_list=[veh for veh in list(abnormal_veh_dict.keys()) if veh not in self.vehicle_id_list_new]
                            if beha_id_list[0]==1:
                                leader=self.get_case_leader_info()
                                if leader:
                                    if leader.id not in self.vehicle_id_list_new:self.abnormal_veh_list.append(leader.id)
                                    self.leader_id=leader.id
                            for veh in self.abnormal_veh_list:
                                vehicle = stub.GetVehicle(
                                    simulation_pb2.GetVehicleReq(simulation_id=self.simulation_id, vehicle_id=veh),
                                    metadata=self.metadata)
                                veh_info = dict(Length=vehicle.vehicle.info.base_info.Length,
                                                Width=vehicle.vehicle.info.base_info.Width,
                                                Weight=vehicle.vehicle.info.base_info.Weight,
                                                max_dec=vehicle.vehicle.info.base_info.max_dec,
                                                max_acc=vehicle.vehicle.info.base_info.max_acc,
                                                link_id=vehicle.vehicle.info.moving_info.position.link_id)
                                ctrl_vehs.append(Surr(veh, veh_info))
                                self.vehicle_id_list_new.append(veh)
                        else:
                            abnormal_veh_dict={}
                            print('未发生异常行为')
                    print(f'控制车辆列表：{self.vehicle_id_list_new}')
                    if i==num-1:
                        with open(os.path.join(file_path, 'ctrl_vehs.pkl'), 'wb') as file:
                            pickle.dump({'ctrl_vehs': self.vehicle_id_list_new}, file)
                    for num_,veh in enumerate(ctrl_vehs):
                        print(f'控制第{num_ + 1}辆车,id为{veh.id},总共{len(ctrl_vehs)}辆车')
                        # if num_==0:self.make_jam_car(stub, all_vehs_pos)  # todo 固定车辆，发生前方停车事件
                        vehicle, nearest_veh_id_around = self.get_veh_around(stub, veh.id,all_vehs_pos)
                        time_model_begin=time.time()
                        if vehicle.vehicle.info.moving_info.position.link_id=='' :continue
                        #if vehicle.vehicle.info.moving_info.position.dis_to_link_end<15:continue
                        static_paths = stub.GetVehicleReferenceLines(simulation_pb2.GetVehicleReferenceLinesReq(simulation_id=self.simulation_id,vehicle_id=veh.id), metadata=self.metadata)
                        abnormal_beha_info()    #异常行为处理
                        ############################################
                        veh_values = veh.update(allvehs_info,vehicle.vehicle.info,nearest_veh_id_around,static_paths,i_beha=self.i_beha,case_info=self.case_info ) #if veh.id=='ego' else [[0.5],[0]]
                        time_model_end = time.time()
                        print('机动车模型控制单辆车计算耗时{:<1.4f}s'.format(time_model_end-time_model_begin))
                        self.control_veh(stub, veh.id, veh_values['control_value'])
                        ############################################
                        save_veh_values()
                end_time = time.time()
                print(f'单步耗时{round(end_time-start_time,2)}s')
            self.simulation_stop(stub)

    def update_abnormal_veh_status(self,veh_id,i_beha,beha_status):#更新异常行为车状态，如果异常行为未完成，继续执行5s,
        self.abnormal_veh_status={'veh_id':veh_id,'i_beha':i_beha,'beha_done':beha_status}
        if beha_status['beha_done'] or beha_status['beha_stop']:
            self.abnormal_veh_status={}
            self.i_beha=0
            self.leader_id=''

    def make_jam_car(self,stub,all_vehs_pos):
        key_veh_id = 'ego'
        vehs_l_g= {}
        key_veh = list(filter(lambda d: d['id'] == key_veh_id, all_vehs_pos))[0]
        # vehs_l = [item for item in all_vehs_pos if 50 < key_veh['dis_to_link_end'] - item['dis_to_link_end']   < 100
        #           and item['lane_id'][:-1] == key_veh['lane_id'][:-1] ][:3]     #and item['lane_id'][-1] != '3'
        if not self.jam_vehs_l:
            jam_vehs_l = [item for item in all_vehs_pos if  50<item['x'] -key_veh['x']<100
                  and item['lane_id'][:-1] == key_veh['lane_id'][:-1] ]     #and item['lane_id'][-1] != '3'
            num=int(max(jam_vehs_l,key=lambda item:int(item['lane_id'][-1]))['lane_id'][-1])
            for i in range(num):
                # if i==int(key_veh['lane_id'][-1]):continue
                vehs_l = [item for item in jam_vehs_l if  int(item['lane_id'][-1]) ==i+1]
                vehs_l=vehs_l[0] if vehs_l else []
                vehs_l_g[i+1]=vehs_l

            for key,item in vehs_l_g.items():
                if item==[]:continue
                self.control_veh(stub, item['id'], [[-7], [0]])
            self.jam_vehs_l=vehs_l_g
        else:
            for key,item in self.jam_vehs_l.items():
                if item == []: continue
                self.control_veh(stub, item['id'], [[-7], [0]])

        print('控制前车停车',self.jam_vehs_l)
    def construct_case(self,i=2):
        v_ego,v_leader='',''
        if i==1:
            v_ego=3
            v_leader=3
        if i in [2,3,4,5]:
            v_ego=6
        return {'v_ego':v_ego,'v_leader':v_leader,'i_beha':i,'test_veh_id':self.vehicle_id}



    def sampler_behavior(self):#被测车采样异常行为
        #根据被测车的车速选择异常行为集
        beha_group_=[]
        beha_id_list = []

        ego_info=list(filter(lambda d: d.id == self.vehicle_id, self.allvehs_info))[0]
        self.ego_info=ego_info
        ego_info_u = ego_info.moving_info.u
        if ego_info_u<3:beha_group_=self.beha_group['A']       #加塞cutin
        if 3<ego_info_u<15.4:beha_group_=self.beha_group['B']      #0.7*22，向右换道overtake
        if ego_info_u > 15.4: beha_group_ = self.beha_group['C']  # >0.7*22，
        for i in beha_group_:
            r=random.random()
            if r< self.P_dict[i]:
                beha_id_list.append(i)
                print(f'采样概率：{r}，行为采样概率：{self.P_dict[i]}，采样行为：{abnormal_behv_str[i]}')
        return beha_id_list
    def get_case_leader_info(self):
        ego_lane_id = self.ego_info.moving_info.position.lane_id
        ego_dis_to_link = self.ego_info.moving_info.position.dis_to_link_end
        leader_list = [veh for veh in self.allvehs_info if
                       veh.moving_info.position.lane_id == ego_lane_id and 0 > veh.moving_info.position.dis_to_link_end - ego_dis_to_link > -40 and veh.id !=self.vehicle_id]
        if leader_list:return leader_list[max(range(len(leader_list)), key=lambda idx: leader_list[idx].moving_info.position.dis_to_link_end)]
        else: return {}
    def judge_condition(self,i,veh_id):#异常行为触发条件
        '''
        case1:加塞：cutin
        case2:右侧超车：overtake
        '''
        condition1, condition2,condition3 = False, False,False
        veh_info = list(filter(lambda d: d.id == veh_id, self.allvehs_info))[0]
        ego_dis_to_link = self.ego_info.moving_info.position.dis_to_link_end
        ego_lane_id = self.ego_info.moving_info.position.lane_id
        ego_v = self.ego_info.moving_info.u
        veh_lane_id = veh_info.moving_info.position.lane_id
        veh_v=veh_info.moving_info.u
        veh_Length = veh_info.base_info.Length
        veh_dis_to_link=veh_info.moving_info.position.dis_to_link_end
        leader_list = [veh for veh in self.allvehs_info if veh.moving_info.position.lane_id == ego_lane_id and 0>veh.moving_info.position.dis_to_link_end - ego_dis_to_link > -30]
        ego_veh_real_dis=abs(ego_dis_to_link - veh_dis_to_link)   - 0.5 * (self.ego_info.base_info.Length-veh_info.base_info.Length )
        t_THD_define=2
        d_threshold = (ego_v if veh_dis_to_link > ego_dis_to_link else veh_v) * t_THD_define   #veh车在ego后方,否则在ego后方

        if leader_list:
            leader = leader_list[max(range(len(leader_list)), key=lambda idx: leader_list[idx].moving_info.position.dis_to_link_end)]
            leader_dis_to_link = leader.moving_info.position.dis_to_link_end
            leader_Length=leader.base_info.Length
        else:leader_dis_to_link,leader_Length=ego_dis_to_link-100,5
        ego_leader_real_dis = ego_dis_to_link - leader_dis_to_link - 0.5 * (leader_Length + self.ego_info.base_info.Length)
        if i==1:#加塞
            if ego_leader_real_dis>3:condition1 = True
            if 0<np.sign(ego_dis_to_link - veh_dis_to_link)*ego_veh_real_dis < veh_Length: condition2 = True      #ego_leader_real_dis:测试车和前车的实际车距，veh_Length:待控车车长
            if abs(int(veh_lane_id[-1])-int(ego_lane_id[-1]))==1:condition3=True
            if condition1 and condition2 and condition3:return True
        elif i==2:#右侧超车
            if ego_dis_to_link-leader_dis_to_link- 0.5 * (leader_Length + self.ego_info.base_info.Length)>10:condition1 = True
            if 0<np.sign(veh_dis_to_link-ego_dis_to_link )*ego_veh_real_dis<d_threshold:condition2=True
            if int(veh_lane_id[-1])==int(ego_lane_id[-1]) and int(veh_lane_id[-1])<3:condition3=True    #todo 有报错:out of range
            if condition1 and condition2 and condition3:
                return True
        elif i==3:#连续换道
            if np.sign(ego_dis_to_link-veh_dis_to_link)*ego_veh_real_dis < d_threshold:condition1=True
            if ego_v-veh_v>2:condition3=True
            if condition1:
                print('连续换道车id:',veh_info.id)
                return True
        elif i==4:#压线行驶
            if veh_dis_to_link < ego_dis_to_link:condition1=True
            if ego_veh_real_dis < d_threshold:condition2=True
            if int(veh_lane_id[-1])==int(ego_lane_id[-1]):condition3=True
            if condition1 and condition2 and condition3:
                print('压线车id:',veh_info.id)
                return True
        elif i==5:#骑线行驶
            if veh_dis_to_link < ego_dis_to_link:condition1=True
            if ego_veh_real_dis < d_threshold:condition2=True
            if abs(int(veh_lane_id[-1]) - int(ego_lane_id[-1]))<=1: condition3 = True
            if condition1 and condition2 and condition3:
                print('骑线车id:',veh_info.id)
                return True
        elif i==6:#变速行驶
            if veh_dis_to_link < ego_dis_to_link:condition1=True
            if ego_veh_real_dis < d_threshold:condition2=True
            if int(veh_lane_id[-1]) == int(ego_lane_id[-1]): condition3 = True
            if condition1 and condition2 and condition3:
                print('变速车id:',veh_info.id)
                return True
        else:
            return False


    def find_abnormal_veh(self,beha_id_list,all_vehs_pos):#寻找异常行为被控车
        abnormal_veh_dict={}
        candi_vehs=get_ctrl_vehs(all_vehs_pos,3,3)
        candi_vehs_=list(candi_vehs)[0]+list(candi_vehs)[1]
        veh_list=[veh for veh in self.vehicle_id_list_new if veh !=self.vehicle_id]   #self.vehicle_id_list_new
        for i in beha_id_list:
            veh=[veh for veh in candi_vehs_ if self.judge_condition(i,veh )] #candi_vehs,self.vehicle_id_list_new
            if veh:abnormal_veh_dict[veh[0]]=i
        return abnormal_veh_dict
    
    def load_pkl(self,file_name):
        pkl_name={1:'ctrl_vehs.pkl',2:'ctrl_veh_values.pkl',3:'data.pkl'}
        i_list,acc_list,v_list,phi_list,rel_dis_list,y_list,steer_list,selected_id_list=[],[],[],[],[],[],[],[]
        value_dict_list=[{'acc':acc_list,'v':v_list},{'y': y_list, 'phi': phi_list},{'steer': steer_list, 'selected_id': selected_id_list}]
        var_unit_dict={'acc':' [m/s^2]','v':' [m/s]','y':' [m]','phi':'[deg]','steer':'[deg]','selected_id':'','bias_lat':' [m]','bias_v':' [m/s]'}
        ctrl_id='npc83'
        def plot_sub_single(axs,i,key,value_list):
            axs[i].plot(i_list,value_list)
            axs[i].set_title(key+'_sequence')
            axs[i].set_xlabel('steps')
            axs[i].set_ylabel(key+var_unit_dict[key])

        def plot(num):
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(len(value_dict))
            i=0
            for key,value in value_dict.items():
                plot_sub_single(axs,i,key,value)
                if key == 'selected_id': plt.ylim(top=0, bottom=4)
                print(f'{key}列表长度；{len(value)}')
                i+=1
            # plt.text(f'受控车辆{ctrl_id}',horizontalalignment='left', verticalalignment='top')

            plt.tight_layout()
            plt.savefig(os.path.join(document_name,f'{ctrl_id}_{num}.jpg'))
            plt.show()
        for key,pkl_name_selected in pkl_name.items():
            document_name='./projects/pkl/'+file_name
            file_path = os.path.join(document_name, pkl_name_selected)
            # ctrl_id = 'npc16'
            with open(file_path, 'rb') as file:
                if key==3:continue
                if pkl_name_selected!='ctrl_vehs.pkl':
                    step=0
                    while True:
                        try:#9.24新加了保存x,y及后面的信息,0103-4530-3707后，J_value替换成了eval_value，有J值，eval_safe和eval_traffic值
                            data=pickle.load(file)  #if key==3: print(data['i'],data['all_vehs_pos'][0]['id'],data['allvehs_info'][0].id)
                            def process_veh_data():

                                if key==2 and data['veh_id']==ctrl_id:#npc144,ego
                                    print(data['i'],data['veh_id'],'ctrl_value:',[item.tolist() for item in data['ctrl_value']],'x:',data['x'],'y:',data['y'],'u:',data['u'],'phi:',data['phi'],
                                       'lane_id:',data['lane_id'],'dis_to_link:',data['dis_to_link'],'eval_value:',data['eval_value'],'selected_id:',data['selected_id'],'step_tag:',data['step_tag'],'i_beha:',data['i_beha'],'beha_time:',data['beha_time'],'done:',data['done'],'\n',
                                          'vehs_relation_lane1:',data['vehs_relation']['v1l'],data['vehs_relation']['v1f'],'\n','vehs_relation_lane2:',data['vehs_relation']['v2l'],data['vehs_relation']['v2f'],'\n','vehs_relation_lane3:',data['vehs_relation']['v3l'],data['vehs_relation']['v3f'])
                                    i_list.append(data['i'])
                                    acc_list.append([item.tolist() for item in data['ctrl_value']][0][0])
                                    car_leader=data['vehs_relation']['v{}l'.format(data['lane_id'][-1])]
                                    rel_dis_list.append((car_leader['x'] if car_leader else data['x']+30) -data['x'])
                                    y_list.append(data['y'])
                                    v_list.append(data['u'])
                                    phi_list.append(data['phi']*180/math.pi)
                                    steer_list.append([item.tolist() for item in data['ctrl_value']][1][0]*180/math.pi)
                                    selected_id_list.append(data['selected_id'])
                            def process_static_data():
                                if key == 2 and step<10:
                                    with open(os.path.join(document_name, 'bias.csv'),mode= 'a', newline='') as file1:
                                        writer=csv.writer(file1)
                                        writer.writerow([step,data['veh_id'],data['bias_lat'],data['bias_v']])
                                if step==10:print('保存csv成功')

                            process_veh_data()
                            # process_static_data()
                            step+=1
                            # if step>10:break
                        except EOFError:break
                    for num,value_dict in enumerate(value_dict_list):
                        plot(num+1)

                if pkl_name_selected=='ctrl_vehs.pkl':
                    data = pickle.load(file)
                    print(data['ctrl_vehs'])

    def simulation_start(self,stub):
        startResp = stub.Start(simulation_pb2.StartSimulationReq(task_id=self.task_id, record_id=self.record_id),
                               metadata=self.metadata)
        self.simulation_id = startResp.simulation_id

    def step(self,stub):
        time_begin=time.time()
        try:
            stepResult = stub.NextStep(simulation_pb2.NextStepReq(
                simulation_id=self.simulation_id), metadata=self.metadata)
            print('stepResult:', stepResult.state.progress)
            if (stepResult.state.progress <= 0) or (stepResult.state.progress >= 100):
                print(f"仿真结束,状态：{stepResult.state.msg}")
                return True
            else:return False
        except Exception as e:print('step_error:', e)
        print('*' * 100)
        time_end=time.time()
        print(f'step消耗时间为{time_end-time_begin}s')

    def simulation_stop(self,stub):

        stub.Stop(simulation_pb2.StopSimulationReq(
            simulation_id=self.simulation_id), metadata=self.metadata)
        result = stub.GetResults(simulation_pb2.GetResultsReq(
            simulation_id=self.simulation_id), metadata=self.metadata)

    def get_y_list(self,stub,link_id,point):
        time_begin=time.time()
        lanes=self.getLinkinfo(stub,link_id).link.ordered_lanes[1:]
        idx=min(range(len(lanes[0].center_line)),key=lambda index:abs(lanes[0].center_line[index].x-point.x))
        y_list=[item.center_line[idx].y for item in lanes]
        print(f'y_list_idx:{idx},y_list:{y_list},center_line_x:{lanes[0].center_line[idx].x},veh_point:{point.x, point.y}')
        time_end=time.time()
        print('联合仿真获取道路信息接口耗时{:<1.4f}'.format(time_end-time_begin))
        return y_list

    def getLinkinfo(self,stub,link_id)->list:
        link_info=stub.GetLink(simulation_pb2.GetLinkReq(
            simulation_id=self.simulation_id, link_id=link_id), metadata=self.metadata)
        checkError(link_info.error)
        return link_info
    def getLaneinfo(self,stub,lane_id)->list:
        lane_info = stub.GetLane(simulation_pb2.GetLaneReq(
            simulation_id=self.simulation_id,lane_id=lane_id), metadata=self.metadata)
        checkError(lane_info.error)
        return lane_info
    def getallvehicle(self,stub)->list:
        all_vehicles = stub.GetAllVehicles(simulation_pb2.GetAllVehiclesReq(
            simulation_id=self.simulation_id), metadata=self.metadata)
        checkError(all_vehicles.error)
        return all_vehicles

    def get_allvehs_position(self,stub)->tuple:
        all_vehicles = self.getallvehicle(stub)
        allvehs_info_raw = [item.info for item in all_vehicles.vehicles]
        key_veh_id = 'ego'
        key_veh = list(filter(lambda d: d.id == key_veh_id, allvehs_info_raw))[0]
        allvehs_info = [item for item in allvehs_info_raw
                        if item.moving_info.position.link_id == key_veh.moving_info.position.link_id]
        allvehs_position = [{'id': item.id,'x': item.moving_info.position.point.x,'y': item.moving_info.position.point.y,
                             'dis_to_link_end': item.moving_info.position.dis_to_link_end,'lane_id': item.moving_info.position.lane_id}
                            for item in allvehs_info]
        return allvehs_position, allvehs_info_raw

    def control_veh(self,stub, vehicle_id, control_value):
        time_begin=time.time()
        print("车辆id:{},control_value:{:2.2f},{:.3f}".format(vehicle_id,control_value[0][0],control_value[1][0]))
        vehicleControleReult = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
            simulation_id=self.simulation_id, vehicle_id=vehicle_id,
            lon_acc=control_value[0][0], ste_wheel=control_value[1][0]), metadata=self.metadata)
        checkError(vehicleControleReult.error)
        time_end=time.time()
        print('control_veh 花费时间{:<1.4f}s'.format(time_end-time_begin))

    def get_veh_around(self, stub, vehicle_id,all_vehs_pos)->tuple:
        """
        1.整个地图的车辆
        2.根据距离筛选出最近的10辆
        """
        time_begin=time.time()
        vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(
            simulation_id=self.simulation_id, vehicle_id=vehicle_id), metadata=self.metadata)
        checkError(vehicle.error)
        veh_around = [{'id':item['id'], 'x':item['x'], 'y':item['y'], 'lane_id':item['lane_id']} for item
                      in all_vehs_pos if item['id'] != vehicle_id]
        nearest_veh_id_around = []
        if veh_around:
            # 得到指定范围内周车
            nearest_vehs = get_nearest_vehs(vehicle, veh_around, 100, 10)
            nearest_veh_id_around = list(map(lambda x: x['id'], nearest_vehs))
        time_end=time.time()
        print('vehicle接口时间{:<1.4f}s'.format(time_end-time_begin))
        return vehicle, nearest_veh_id_around
    def built_file_dir(self,file_type='csv'):
        current_time = time.strftime("%Y%m%d%H%M%S").replace(':', '')
        folder_name = current_time.replace(':', '')[:-2]  # 移除时间字符串中的冒号，用于文件夹命名
        if file_type=='csv':folder_path = f"./projects/csv/{folder_name}-{self.task_id}-{self.record_id}"  # 文件夹路径
        if file_type=='pkl':folder_path = f"./projects/pkl/{folder_name}-{self.task_id}-{self.record_id}"  # 文件夹路径
        # 创建一个以当前时间命名的文件夹
        if self.old_file_path == '':
            os.makedirs(folder_path)
            self.old_file_path = folder_path
        return self.old_file_path

    def save_data(self):
        self.save_time += 1
        if self.save_time % 2 == 0:
            self.data_num += 1
            files_path = self.built_file_dir()
            for veh in self.vehicle_id_list_new:
                file_name = f'values_{veh}.csv'
                file_path = os.path.join(files_path, file_name)
                with open(file_path, mode='a+', newline='') as file:
                    writer = csv.writer(file)
                    if veh not in self.used_names_list:
                        writer.writerow(self.set_data_title())
                        self.used_names_list.append(veh)
                    writer.writerow(self.set_data(veh))
                    file.close()

    def set_data_title(self) -> list:
        title = ['time','x','y','phi','u','v','w','acc_lon','acc_lat','acc_w','acce_des','theta_des','static_path_num',
            'static_path_index_des', 'lane_id','lat_bias_des','lateral_position','eva_traffic','eva_safety',
            'dis_2_veh_1l','dis_2_veh_1f','dis_2_veh_2l','dis_2_veh_2f','dis_2_veh_3l','dis_2_veh_3f']
        return title

    def set_data(self, ego: Surr) -> list:
        acce_des = ego.control_value[0]  # 'acce_des',
        theta_des = ego.control_value[1]  # 'theta_des',
        static_path_num = len(ego.static_paths),  # 'static_path_num',
        static_path_index_des = ego.selected_id,  # 'static_path_index_des',
        lat_bias_des = ego.bias_lat
        lateral_position = 0  # 'lateral_position',
        eva_traffic = ego.eval_value  # 'eva_traffic',
        eva_safety = ego.eval_value  # 'eva_safety',
        dis_dict = ego.matched_static_path_veh
        save_data = [self.save_time, ego.x, ego.y,ego.phi, ego.u,  ego.v, ego.w,ego.lon_acc,  ego.lat_acc,  ego.w_acc,
            acce_des,  theta_des,  static_path_num,  static_path_index_des,   ego.lane_id,lat_bias_des,lateral_position,
            eva_traffic,  eva_safety,dis_dict[1][0],dis_dict[1][1],dis_dict[2][0],dis_dict[2][1],dis_dict[3][0],dis_dict[3][1]]
        return save_data


def get_nearest_vehs(veh_info, around_vehs, desir_dis=50, veh_num=10):
    host_veh_point = veh_info.vehicle.info.moving_info.position.point
    host_veh_point = np.array([host_veh_point.x, host_veh_point.y])
    nearest_vehs = [item for item in around_vehs if np.linalg.norm(host_veh_point - np.array(
            [item['x'], item['y']])) < desir_dis]
    nearest_vehs = sorted(nearest_vehs,key=lambda item: math.sqrt((item['x'] - host_veh_point[0]) ** 2 +
                                                     (item['y'] - host_veh_point[1]) ** 2))[:min(len(nearest_vehs), veh_num)]
    return nearest_vehs

def order_vehicles(self,vehicles):
    if isinstance(vehicles,list):
        dis_list=np.array([abs(item['dis_to_link']-self.dis_to_link_end) for item in vehicles])
        index = np.argmin(dis_list)
    vehicle=vehicles[index]
    return vehicle

def output_lane_rad_tag(lane_rad):
    tag = 1
    if 0 <= lane_rad < math.pi / 4 or math.pi * 7 / 4 < lane_rad <= math.pi * 2:tag = 1
    if math.pi / 4 < lane_rad <= math.pi * 3 / 4:tag = 2
    if math.pi * 3 / 4 < lane_rad <= math.pi * 5 / 4:tag = 3
    if math.pi * 5 / 4 < lane_rad <= math.pi * 7 / 4:tag = 4
    return tag

def world_to_self_car( world_x, world_y, car_x, car_y, car_heading):
    # Calculate the relative position of the world coordinates with respect to the car
    relative_x = world_x - car_x
    relative_y = world_y - car_y
    # Rotate the relative position based on the car's heading
    rotated_x = relative_x * math.cos(car_heading) + relative_y * math.sin(car_heading)
    rotated_y = -relative_x * math.sin(car_heading) + relative_y * math.cos(car_heading)  # -relative_x修改为relative_x
    return rotated_x, rotated_y

def checkError(err):
    if err is None:return False
    if err.code != 0:
        print(err.msg)
        return True

def get_ctrl_vehs(all_vehs_pos, l_num=3,f_num=3):
    key_veh_id = 'ego'
    key_veh = list(filter(lambda d: d['id'] == key_veh_id, all_vehs_pos))[0]
    vehs_l = [item for item in all_vehs_pos if -100 < item['dis_to_link_end'] - key_veh['dis_to_link_end'] < 0
              and item['lane_id'].split('_')[:2] == key_veh['lane_id'].split('_')[:2]]
    vehs_f = [item for item in all_vehs_pos if 0 < item['dis_to_link_end'] - key_veh['dis_to_link_end'] < 100
              and item['lane_id'].split('_')[:2] == key_veh['lane_id'].split('_')[:2]]
    def _get_nearest_vehs(vehs,num=9):
        return sorted(vehs, key=lambda item: math.sqrt((item['x'] - key_veh['x']) ** 2 +
                                            (item['y'] - key_veh['y']) ** 2))[:min(len(vehs), num)]
    vehs_l_nearest=_get_nearest_vehs(vehs_l)
    # vehs_l_id_nearest=[item['id'] for item in vehs_l_nearest]
    vehs_l_id_nearest=[item['id'] for item in vehs_l_nearest if abs(int(item['lane_id'][-1])-int(key_veh['lane_id'][-1]))<=min(l_num-1,2)]
    vehs_f_nearest = _get_nearest_vehs(vehs_f)
    vehs_f_id_nearest = [item['id'] for item in vehs_f_nearest]
    return vehs_l_id_nearest[:l_num],vehs_f_id_nearest[:f_num]
def rotation(l, w, phi):
    """phi: rad"""
    diff_x = l * np.cos(phi) - w * np.sin(phi)
    diff_y = l * np.sin(phi) + w * np.cos(phi)
    return (diff_x, diff_y)

def render(frame:Surr,eval_value,ego_name: str = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import colors
    # define colors
    cmap = plt.cm.jet
    mycmap = cmap.from_list('Custom cmap',[[0 / 255, 255 / 255, 0 / 255], [255 / 255, 255 / 255, 0 / 255],[255 / 255, 0 / 255, 0 / 255]], cmap.N)
    c_norm = colors.Normalize(vmin=0, vmax=10, clip=True)

    def _draw_rotate_rec(veh, ec, fc: str = 'white'):
        diff_x, diff_y = rotation(-veh['Length'] / 2, -veh['Width'] / 2, veh['phi'])
        rec = patches.Rectangle((veh['x'] + diff_x, veh['y'] + diff_y), veh['Length'], veh['Width'],angle=veh['phi'] / np.pi * 180, ec=ec, fc=fc)
        return rec
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    x_min, x_max, y_min, y_max = -10000, 10000, -10000, 10000
    rec_handle = _draw_rotate_rec(frame.ego,  'red','white' )
    ax.add_patch(rec_handle)
    for _key, _veh in {**frame.matched_static_path_veh}.items():
        for i in range(2):
            rec_handle = _draw_rotate_rec(_veh[i], 'black' , mycmap(c_norm(3)))
            ax.add_patch(rec_handle)
            ax.text(_veh[i]['x'], _veh[i]['y'], 'id={}, J_value={:.2f}, v={:.1f}'.format(_veh[i]['id'],
                    eval_value['J_value'][_key] ,_veh[i]['u']),va='center',ha='center',rotation=45,fontsize=12)
            ax.scatter(_veh[i]['x'], _veh[i]['y'], c='black', s=5)
            x_min, x_max, y_min, y_max = np.min([x_min, _veh[i]['x']]), np.max([x_max,_veh[i]['x']]), np.min(
                [y_min, _veh[i]['y']]), np.max([y_max, _veh[i]['y']])
            print('id={:<4} x={:<10.2f} y={:<10.2f} v={:<6.2f} phi={:<6.2f} J_value={:<8.2f} eval_safe={:<1.2f} eval_traffic={:<1.2f} l={:<6.1f} w={:<6.1f} dis_to_link={:<6.1f}'.format(
                _veh[i]['id'], _veh[i]['x'],_veh[i]['y'], _veh[i]['u'], _veh[i]['phi'] / np.pi * 180,
                eval_value['J_value'][_key] ,eval_value['val_safe'][_key] ,eval_value['val_traffic'][_key] ,_veh[i]['Length'], _veh[i]['Width'],_veh[i]['dis_to_link']))
    plt.xlim(x_min - 10, x_max + 10)
    plt.ylim(y_min - 10, y_max + 10)
    plt.axis('equal')

if __name__ == '__main__':
    Cosim().run()
    # Cosim().load_pkl('202312172129-5044-7971')
    def test_eval_value(Surr):
        import matplotlib.pyplot as plt
        import numpy as np
        ego_dict = dict( Length=4.5, Width=1.8,Weight=1800,max_acc=2.5,max_dec=7)
        Surr = Surr('0',ego_dict)
        path1 = [np.ones(40) * -3.2, np.arange(0, 20, 0.5)]
        path2=[np.zeros(40),np.arange(0,20,0.5)]
        path3 = [np.ones(40) * 3.2, np.arange(0, 20, 0.5)]
        Surr.matched_static_path_array={1:path1,2:path2,3:path3}
        Surr.lane_id = '2'
        Surr.lane_rad=0.5*math.pi
        Surr.get_xy_translane()


        ego_simple = dict(id='0', x=0, y=0, u=15, phi=np.pi / 2, Length=4.5, Width=1.8, dis_to_link_end=100)
        ego = dict(id='0',x=0, y=0, u=15, phi=np.pi / 2,Length=4.5,Width=1.8,dis_to_link_end=100,lane_id='lk2',d=3.2,lane_rad=np.pi / 2,ego=ego_simple)
        def update_ego(**kwargs):
            for key,value in kwargs.items():
                exec('Surr.{0}=value'.format(key),{'Surr':Surr,'value':value})
        update_ego(**ego)
        car1 = dict(id='1',x=-3.2, y=10, u=25, phi=np.pi / 2,Length=4.5,Width=1.8)
        car1['dis_to_link']=100-car1['y']
        car2 = dict(id='2',x=-3.2, y=-10, u=20, phi=np.pi / 2,Length=4.5,Width=1.8)
        car2['dis_to_link'] = 100 - car2['y']
        car3 = dict(id='3', x=0, y=15, u=20, phi=np.pi / 2,Length=4.5,Width=1.8)
        car3['dis_to_link'] = 100 - car3['y']
        car4 = dict(id='4', x=0, y=-10, u=15, phi=np.pi / 2,Length=4.5,Width=1.8)
        car4['dis_to_link'] = 100 - car4['y']
        car5 = dict(id='5', x=3.2, y=10, u=30, phi=np.pi / 2,Length=4.5,Width=1.8)
        car5['dis_to_link'] = 100 - car5['y']
        car6 = dict(id='6', x=3.2, y=-10, u=15, phi=np.pi / 2,Length=4.5,Width=1.8)
        car6['dis_to_link'] = 100 - car6['y']
        Surr.matched_static_path_veh={1:[car1,car2],2:[car3,car4],3:[car5,car6]}
        eval_value=Surr.eval_function()
        Surr.get_track_speed(0,{})
        render(Surr,eval_value)
        plt.show()
    def check_csv():
        import csv
        document_name = './projects/pkl/202310041302-4531-3714'
        with open(os.path.join(document_name,'bias.csv'), 'r') as infile, open(os.path.join(document_name,'bias1.csv'), 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                if any(row):
                    writer.writerow(row)
    def concat_csv():
        import os
        import glob
        import pandas as pd

        # 指定包含CSV文件的文件夹路径
        folder_path = './projects/pkl'  # 请将'path_to_folder'替换为实际的文件夹路径

        # 获取文件夹中所有CSV文件的路径
        csv_files = glob.glob(os.path.join(folder_path, '**', 'bias.csv'), recursive=True)

        # 创建一个空的DataFrame用于存储合并后的数据
        merged_data = pd.DataFrame()

        # 遍历CSV文件并合并数据
        for file in csv_files:
            # 读取CSV文件
            data = pd.read_csv(file,header=None)
            # 检查数据是否为空，如果不是空行，则将其添加到合并的DataFrame中
            if not data.empty:
                merged_data = pd.concat([merged_data, data])

                # 将合并后的数据写入新的CSV文件
        # os.makedirs(folder_path+'/merged_csv/')
        merged_data.to_csv(os.path.join(folder_path+'/merged_csv/','merged_data1.csv'), index=False)
    def plot_csv():
        import pandas as pd
        import matplotlib.pyplot as plt
        folder_path = './projects/pkl/merged_csv/merged_data0.csv'
        folder_path1 = './projects/pkl/merged_csv/y_data.csv'
        data = pd.read_csv(folder_path)
        data1 = pd.read_csv(folder_path1)
        print(data.head())
        print(data1.head())
        values = data['bias_y']
        values1 = np.array(data1['y'])-np.mean(data1['y'])
        # values1 = np.array(data1['y']) - np.mean(data1['y'])
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        fig=plt.figure(figsize=(8,6))
        plt.hist(list(values), bins=50,range=(-1,1), edgecolor='black',color='blue', alpha=0.5,label='仿真模型')  # 20个bin（条形）
        plt.hist(values1[:len(values)], bins=50, range=(-1,1),edgecolor='black',color='red', alpha=0.5,label='high_D数据集')  # 20个bin（条形）
        # plt.title('Histogram')
        # plt.xlabel('v_bias [m/s]')
        plt.xlabel('y_bias [m]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
    # plot_csv()
    # concat_csv()
    # check_csv()
    # test_eval_value(Surr)
