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
    bias_lat_var=0.3,      #纵向位置偏差方差
    bias_v_miu=0,        #纵向位置偏差均值
    bias_v_var=0.5,      #纵向位置偏差方差
    driver_nou=1,          #驾驶员激进系数（增大），期望速度（变大），期望间距（变小），转向期望间距（变小）
)       #6个参数

driver_delay=min(1/model_params['driver_nou'],3),  #驾驶员反应时间


#控制车辆类
class Surr(object):
    def __init__(self,id,vehicle):
        self.id=id
        self.lane_change_time,self.lane_change_times,self.LC_desir_time=0,0,0
        self.Length,self.Width,self.Weight=vehicle['Length'],vehicle['Width'],vehicle['Weight']     #传vehicle信息，改车的长宽质量
        self.max_dec,self.max_acc,self.x,self.y,self.phi,self.u,self.acc_lon,self.lane_rad,self.dis_to_link_end,self.selected_path_id=vehicle['max_dec'],vehicle['max_acc'],0,0,0,0,0,0,0,0
        self.static_path,self.all_veh_info=[],[]
        self.matched_static_path,self.matched_static_path_array,self.matched_static_path_veh,self.xy_lane,self.ego = {},{},{},{},{}
        self.lane_id,self.old_lane_id= '',''
        self.link_id=vehicle['link_id']     #认为车辆link不变,后续需要修改成变化的link
        self.ccbf_controller = ccbf_controller.CCBF_Controller(VehicleSpec())
        (P1, P2, P3, P4, P5, P6) = CCBFOption.weight
        self.ccbf_controller.update_lyapunov_parameter(P1, P2, P3, P4, P5, P6)
        self.bias_lat = random.gauss(model_params['bias_lat_miu'], model_params['bias_lat_var'])  # 驾驶道路中心线横向偏差
        self.bias_v = random.gauss(model_params['bias_v_miu'], model_params['bias_v_var'])  # 驾驶道路速度偏差

    def update(self,all_veh_info:list,vehicle:object,nearest_veh_id_around:list,y_list):
        '''

        :return:
        '''
        self.all_veh_info = all_veh_info
        self.lane_change_time+=1
        self.d=abs(y_list[0]-y_list[1])              #车道宽度,abs(y_list[0]-y_list[1]),3.2m
        self.x=vehicle.moving_info.position.point.x
        self.y=vehicle.moving_info.position.point.y
        self.phi=vehicle.moving_info.position.phi
        self.u=vehicle.moving_info.u
        self.acc_lon=vehicle.moving_info.lon_acc
        self.lane_id=vehicle.moving_info.position.lane_id
        self.link_id=vehicle.moving_info.position.link_id
        self.dis_to_link_end=vehicle.moving_info.position.dis_to_link_end
        print(f'x:{self.x},y:{self.y},u:{self.u},phi:{self.phi},cl_time:{self.lane_change_time},lane_id:{self.lane_id}')
        self.static_path=vehicle.static_path
        self.static_path_match(y_list)
        self.lane_rad=self.cul_lane_rad()
        vehs_relation=self.find_vehs(nearest_veh_id_around,y_list)
        i_behavior=1
        if i_behavior==1:
            eval_value=self.eval_function()
        else:
            self.selected_path_id=self.get_path_id(i_behavior)
        ref_path=self.matched_static_path_array[self.selected_path_id]
        v_A = self.get_track_speed()
        new_ref_path = self.gen_new_static_path(ref_path, v_A,0)
        print(f"new_ref_path_x:{new_ref_path[0][0]},new_ref_path_y:{new_ref_path[1][0]},静态路径数量：{len(self.static_path)}")
        control_value=self.ccbf_controler(new_ref_path)
        self.reset()
        return {'control_value':control_value,'eval_value':eval_value,'vehs_relation':vehs_relation}

    def reset(self):
        self.matched_static_path.clear()
        self.matched_static_path_array.clear()
        self.matched_static_path_veh.clear()

    def get_path_id(self,i_behavior):
        if i_behavior==2:#右侧超车:1.判断与前车距离在阈值范围内，前车速度低于0.5倍限速即40km/h 2.选择的路径为当前路径的右侧 3.换道到右侧后，加速，等换道到左侧的条件， 4.选择当前路径的左侧
            if True:self.selected_path_id=int(self.lane_id[-1])+1
            else:self.selected_path_id=int(self.lane_id[-1])


    def cul_lane_rad(self):
        waypoints = self.matched_static_path[2]
        cosest_idx = min(range(len(waypoints)), key=lambda closest_idx: math.sqrt(
            (waypoints[closest_idx].x - self.x) ** 2 + (waypoints[closest_idx].y - self.y) ** 2))
        waypoints = waypoints[cosest_idx:cosest_idx + 41]
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

    def find_vehs(self,vehs_list,y_list):

        nearest_vehs=[self.get_veh_info_from_traffic(veh_id) for veh_id in vehs_list]
        vehs_1l,vehs_2l,vehs_3l,vehs_1f,vehs_2f,vehs_3f=[],[],[],[],[],[]
        vehs_d = dict(v1l=vehs_1l,v2l=vehs_2l,v3l=vehs_3l,v1f=vehs_1f,v2f=vehs_2f,v3f=vehs_3f)
        def cul_min_dis(idx):
            if veh['dis_to_link'] >self.dis_to_link_end:return abs((y_list[idx] - math.sin(self.lane_rad) * abs(self.dis_to_link_end - veh['dis_to_link'])) - veh['y'])
            else:return abs((y_list[idx] + math.sin(self.lane_rad) * abs(self.dis_to_link_end - veh['dis_to_link'])) - veh['y'])

        for veh in nearest_vehs:
            idx_cul=min(range(len(y_list)),key=lambda idx:cul_min_dis(idx))+1
            idx=int(veh['lane_id'][-1])
            # assert idx_cul==idx,'idx_cul:{} is not equal to idx:{}'.format(idx_cul,idx)
            vehs_l,vehs_f=vehs_d[f'v{idx_cul}l'],vehs_d[f'v{idx_cul}f']
            if veh['dis_to_link'] >self.dis_to_link_end:exec('vehs.append(value)',{'vehs':vehs_f,'value':veh})
            else:exec('vehs.append(value)',{'vehs':vehs_l,'value':veh})
            vehs_l, vehs_f = [], []
        for key,vehs in vehs_d.items():
            vehs_d[key]=order_vehicles(self,vehs) if vehs else []
        self.matched_static_path_veh={1:[vehs_d['v1l'],vehs_d['v1f']],2:[vehs_d['v2l'],vehs_d['v2f']],3:[vehs_d['v3l'],vehs_d['v3f']]}
        print(f'vehs_relation:{vehs_d}')
        return vehs_d
    def static_path_match(self,y_list):
        static_path=copy.deepcopy(self.static_path[0].point)
        closest_idx = min(range(len(static_path)), key=lambda closest_idx: abs(static_path[closest_idx].x - self.x))
        for i,y in enumerate(y_list):
            idx=min(range(len(self.static_path)), key=lambda index: abs(self.static_path[index].point[closest_idx].y - y))
            if (self.lane_id[-1]=='1' and i==2) or (self.lane_id[-1]=='3' and i==0):continue #1号车道时，不需要评估3号车道路径，3号车道时，不需要1号车道路径
            self.matched_static_path[i + 1] = self.static_path[idx].point[closest_idx:closest_idx+min(41,len(self.static_path[idx].point)-closest_idx)]
            self.matched_static_path_array[i+1] =self.out_static_path(self.matched_static_path[i + 1])

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

        if self.lane_change_times==0 or self.lane_change_time>15 :#换道后，time置为0，不评估静态路径，每次time累加1，到20以后再评估静态路径
            for idx, path in self.matched_static_path.items():  # self.cul_real_dis可以计算与前车的实际距离
                val_safe = self.safe_evaluation(idx, self.matched_static_path_veh[idx])
                val_traffic = self.traffic_evaluation(self.matched_static_path_veh[idx])    #todo safe,<1时都归零，
                J_value[idx], val_safe_d[idx], val_traffic_d[idx] = val_safe + val_traffic, val_safe, val_traffic
                print(f'idx:{idx},val_safe:{val_safe},val_traffic:{val_traffic}')   #todo condition,加一个累积时间1s,以后考虑相同激进系数，不同类别阈值
            J_max_id = max(list(J_value.keys()), key=lambda index: J_value[index])

            if (J_value[J_max_id] - J_value[int(self.lane_id[-1])]) / J_value[int(self.lane_id[-1])] > 0.1 :#0.3，换道效益提高阈值
                self.LC_desir_time+=1
                if self.lane_change_times == 0: self.selected_path_id = J_max_id
                elif self.LC_desir_time>5:
                    self.selected_path_id = J_max_id    #10
                    self.LC_desir_time = 0
                print(f'效益提升累计时间{self.LC_desir_time}')
            else:
                if J_max_id !=int(self.lane_id[-1]):self.LC_desir_time = 0
                if self.lane_change_times == 0: self.selected_path_id = int(self.lane_id[-1])
        else:
            for idx, path in self.matched_static_path.items():  # self.cul_real_dis可以计算与前车的实际距离
                val_safe_d[idx] = self.safe_evaluation(idx, self.matched_static_path_veh[idx])

        if self.selected_path_id!=int(self.lane_id[-1]) and self.old_lane_id=='':self.old_lane_id=int(self.lane_id[-1])
        if val_safe_d[self.selected_path_id]<0.5 and isinstance(self.old_lane_id,int):self.selected_path_id=self.old_lane_id
        if int(self.lane_id[-1])!=self.old_lane_id and isinstance(self.old_lane_id,int):#实际换道后时间清零
            self.lane_change_time = 0
            self.lane_change_times += 1
            self.old_lane_id = ''

        print(f'评估函数值{J_value},选择静态路径索引{self.selected_path_id},lane_change_time:{self.lane_change_time},换道次数：{self.lane_change_times},效益提升累计时间：{self.LC_desir_time}')
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
    def get_track_speed(self):#todo 留速度，加一个速度偏置
        v_static_path = model_params['target_v'] * model_params['driver_nou']
        v_A = v_static_path + self.bias_v
        print('v_static_path={:<2.2f} bias_v={:<2.2f} v_A={:<2.2f}'.format(v_static_path,self.bias_v,v_A))
        return v_A

    def gen_new_static_path(self, path, target_v, target_phi):#todo 加一个横向偏置
        path.append(target_phi * np.ones(41))
        path.append(target_v * np.ones(41))
        path[1]=path[1]+self.bias_lat
        print('bias_lat:',self.bias_lat)
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

    def ccbf_controler(self,static_paths):
        if abs(self.lane_rad - 2 * math.pi) * 180 / math.pi < 0.2 or abs(self.lane_rad * 180 / math.pi < 0.2):self.lane_rad = 0
        new_phi = self.phi - self.lane_rad
        print(f"new_phi:{new_phi}")
        transfered_path = self.transfer_path(static_paths, self.lane_rad)
        print(f"new_path_x:{transfered_path[0][0]},new_path_y:{transfered_path[1][0]}")
        v,w = 0,0
        self.ccbf_controller.update_values(0, 0, new_phi, self.u, v, w)
        self.ccbf_controller.update_waypoints(transfered_path)
        self.ccbf_controller.construct_clf_sep()
        car_leader = self.matched_static_path_veh[int(self.lane_id[-1])][0]
        if car_leader:
            front_veh_x, front_veh_y=world_to_self_car(car_leader['x'], car_leader['y'], self.x, self.y, self.lane_rad)
            print(f'前车转换后坐标x:{front_veh_x}')
            front_veh_real_x=front_veh_x-0.5*(car_leader['Length']+self.Length)
            accel_cbf=self.ccbf_controller.construct_cbf_front_veh(front_veh_real_x,v_l=car_leader['u'])   #前方有车时，控制刹车的控制量
        else:
            accel_cbf = [0]
            accel_cbf = self.ccbf_controller.construct_cbf_front_veh(50, v_l=30)
        accel,steer = self.ccbf_controller.throttle,self.ccbf_controller.steer
        control_bound: tuple = ((-0.4, -self.max_dec), (0.4, self.max_acc))# lower bound -0.4（右转向22度）,-7 为车辆的一般减速度    # upper bound 0.4（左转向22度）， 2.5
        accel,accel_cbf = np.clip(accel, control_bound[0][1], control_bound[1][1]),np.clip(accel_cbf, control_bound[0][1], control_bound[1][1])
        steer = np.clip(steer, control_bound[0][0], control_bound[1][0])
        print(f'(CCBF) car following input, acc: {accel}, steer: {steer},acc_cbf:{accel_cbf}')
        # control_value = [accel if accel_cbf[0]==0 else accel_cbf, steer]
        control_value = [accel, steer]
        return control_value

    def safe_evaluation(self, idx, vehs_relation):
        if vehs_relation:
            risk = self.eval_podar( idx, vehs_relation)
            safe_value = (2.3 - risk) / 2.3
        else:safe_value = 1
        return safe_value
    def get_xy_translane(self,idx):
        lane_idx=int(self.lane_id[-1])
        if lane_idx == 1:
            self.xy_lane[1] = {'x':self.x,'y': self.y}
            self.xy_lane[2] = {'x':self.x+self.d*math.sin(self.lane_rad),'y':self.y-self.d*math.cos(self.lane_rad)}
        if lane_idx == 2:
            self.xy_lane[1] = {'x':self.x-self.d*math.sin(self.lane_rad),'y':self.y+self.d*math.cos(self.lane_rad)}
            self.xy_lane[2] = {'x':self.x,'y': self.y}
            self.xy_lane[3] = {'x':self.x+self.d*math.sin(self.lane_rad),'y':self.y-self.d*math.cos(self.lane_rad)}
        if lane_idx == 3:
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
        xy=self.get_xy_translane(idx)
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
        v_d = model_params['target_v'] * model_params['driver_nou']
        car_leader = vehs_relation[0]
        if car_leader:
            length_leader = car_leader['Length']
            dis = car_leader['dis_to_link']-self.dis_to_link_end
            real_dis = abs(dis) - (self.Length + length_leader) / 2
            v_l = car_leader['u']
            d_b = max(0.0001, 3.6 * v_l)
            def traffic_eval_method1():
                if real_dis > 5:
                    if v_l > d_b:v_d0 = v_d
                    else:v_d0 = min(0.5 * real_dis * v_l / v_d, v_d)
                    return round(v_d0, 2)
                else:return 0
            def traffic_eval_method2():
                d_b=80
                return min(real_dis/d_b , 1)*v_d
            v_d0=traffic_eval_method2()
            traffic_value = round(v_d0 / 22.2, 2)
        else:traffic_value=1
        return traffic_value
@dataclass
class Cosim():
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjEzLCJvaWQiOjEwMSwibmFtZSI6IueGiuadsCIsImlkZW50aXR5Ijoibm9ybWFsIiwicGVybWlzc2lvbnMiOlsidGFzay50YXNrLnB1YmxpYy5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wdWJsaWMuQUNUSU9OX0NPUFkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBPUlQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9BREQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fQ09QWSIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fUkVQTEFZIiwidGFzay50YXNrLnByaXZhdGUuQUNUSU9OX1JFUE9SVCIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwidGFzay50YXNrLnBlcnNvbmFsLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucGVyc29uYWwuQUNUSU9OX1JFUE9SVCIsInJlc291cmNlLnZlaGljbGUucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudmVoaWNsZS5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudmVoaWNsZS5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS52ZWhpY2xlLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2UudmVoaWNsZS5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucHVibGljLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9BREQiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2Vuc29yLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnNlbnNvci5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnB1YmxpYy5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VQREFURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9WSUVXIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX0FERCIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS5tYXAucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UubWFwLnBlcnNvbmFsLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnNjZW5hcmlvLnByaXZhdGUuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wcml2YXRlLkFDVElPTl9DT1BZIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS50cmFmZmljX2Zsb3dfY29uZmlnLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9ET1dOTE9BRCIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RPV05MT0FEIl0sImlzcyI6InVzZXIiLCJzdWIiOiJMYXNWU2ltIiwiZXhwIjoxNjk4NjM4MzMwLCJuYmYiOjE2OTgwMzM1MzAsImlhdCI6MTY5ODAzMzUzMCwianRpIjoiMTMifQ.FBRtXhtc42paYAnj6Ilf6BLkDeWsAdIBXIjhnjhcyks'
    metadata = [('authorization','Bearer ' + token)]
    http = 'qianxing-grpc.risenlighten.com:80'
    # http='127.0.0.1:8290'  #桌面端lasvsim连接grpc用
    max_recv_msg_size = 20 * 1024 * 1024  # 20M
    # 创建channelArguments对象来设置选项
    channel_args = [('grpc.max_receive_message_length', max_recv_msg_size), ('grpc.default_timeout', 100)]

    task_id,record_id,vehicle_id = 4645,4201,'ego'
    vehicle_id_list = [vehicle_id]
    progress_times,save_time,data_num = 0,0,0
    choosed_list,vehicle_id_list_new,veh_id_around,all_vehs_pos,allvehs_info,control_value,used_names_list = [],[],[],[],[],[],[]
    veh_info,agent,ctrl_value = {},{},{}
    old_file_path,simulation_id = '',''

    def run(self):
        with grpc.insecure_channel(self.http, options=self.channel_args) as channel:
            stub = simulation_pb2_grpc.CosimStub(channel)
            self.simulation_start(stub)
            file_path = self.built_file_dir(file_type='pkl')
            for i in range(300):
                start_time=time.time()
                print("=" * 100,'\n',f'i:{i}')
                result = self.step(stub)
                if result:break
                all_vehs_pos, allvehs_info = self.get_allvehs_position(stub)    #存为Pickle
                with open(os.path.join(file_path, 'data.pkl'),'ab') as file:
                    pickle.dump({'i':i,'all_vehs_pos':all_vehs_pos,'allvehs_info':allvehs_info},file)
                if i > 0:
                    if not self.vehicle_id_list_new:#todo 找最近10辆控制车
                        ctrl_vehs = []
                        ctrl_l_list,ctrl_f_list = get_ctrl_vehs(all_vehs_pos)
                        self.vehicle_id_list_new = self.vehicle_id_list #+ ctrl_l_list
                        for veh in self.vehicle_id_list_new:
                            vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(simulation_id=self.simulation_id, vehicle_id=veh), metadata=self.metadata)
                            veh_info=dict(Length=vehicle.vehicle.info.base_info.Length,Width=vehicle.vehicle.info.base_info.Width,Weight=vehicle.vehicle.info.base_info.Weight,
                                          max_dec=vehicle.vehicle.info.base_info.max_dec,max_acc=vehicle.vehicle.info.base_info.max_acc,link_id=vehicle.vehicle.info.moving_info.position.link_id)
                            ctrl_vehs.append(Surr(veh, veh_info))
                        with open(os.path.join(file_path, 'ctrl_vehs.pkl'), 'wb') as file:
                            pickle.dump({'ctrl_vehs': self.vehicle_id_list_new}, file)
                    print(f'控制车辆列表：{self.vehicle_id_list_new}')
                    for num,veh in enumerate(ctrl_vehs):
                        print(f'控制第{num + 1}辆车,id为{veh.id},总共{len(ctrl_vehs)}辆车')
                        vehicle, nearest_veh_id_around = self.get_veh_around(stub, veh.id,all_vehs_pos)
                        if vehicle.vehicle.info.moving_info.position.link_id=='' \
                                or vehicle.vehicle.info.moving_info.position.dis_to_link_end<15:continue
                        y_list=self.get_y_list(stub,vehicle.vehicle.info.moving_info.position.link_id,vehicle.vehicle.info.moving_info.position.point)
                        veh_values = veh.update(allvehs_info,vehicle.vehicle.info,nearest_veh_id_around,y_list) #if veh.id=='ego' else [[0.5],[0]]
                        self.control_veh(stub, veh.id, veh_values['control_value'])
                        with open(os.path.join(file_path, 'ctrl_veh_values.pkl'), 'ab') as file:
                            ctrl_veh_info={'i':i,'veh_id':veh.id,'ctrl_value':veh_values['control_value'],'lane_id':veh.lane_id,'dis_to_link':veh.dis_to_link_end,'u':veh.u,'phi':veh.phi,
                                           'x':veh.x,'y':veh.y,'eval_value':veh_values['eval_value'],'selected_id':veh.selected_path_id,'vehs_relation':veh_values['vehs_relation'],
                                           'bias_lat':veh.bias_lat,'bias_v':veh.bias_v}
                            pickle.dump(ctrl_veh_info, file)
                        print(f'id为{veh.id}，当前所在车道{veh.lane_id}，dis_to_link:{veh.dis_to_link_end}','\n',"*"*50)
                end_time = time.time()
                print(f'单步耗时{round(end_time-start_time,2)}s')
            self.simulation_stop(stub)

    def load_pkl(self):
        pkl_name={1:'ctrl_vehs.pkl',2:'ctrl_veh_values.pkl',3:'data.pkl'}
        acc_list,v_list,phi_list,rel_dis_list,y_list,steer_list,selected_id_list=[],[],[],[],[],[],[]
        # value_dict={'acc':acc_list,'v':v_list}
        # value_dict = {'y': y_list, 'phi': phi_list}
        # value_dict = {'steer': steer_list, 'selected_id': selected_id_list}
        value_dict_list=[{'acc':acc_list,'v':v_list},{'y': y_list, 'phi': phi_list},{'steer': steer_list, 'selected_id': selected_id_list}]
        vdi=2
        value_dict=value_dict_list[vdi]
        var_unit_dict={'acc':' [m/s^2]','v':' [m/s]','y':' [m]','phi':'[deg]','steer':'[deg]','selected_id':'','bias_lat':' [m]','bias_v':' [m/s]'}

        def plot_sub_single(axs,i,key,value_list):
            axs[i].plot(value_list)
            axs[i].set_title(key+'_sequence')
            axs[i].set_xlabel('steps')
            axs[i].set_ylabel(key+var_unit_dict[key])
        def plot():
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(len(value_dict))
            i=0
            for key,value in value_dict.items():
                plot_sub_single(axs,i,key,value)
                i+=1

            plt.tight_layout()
            plt.savefig(os.path.join(document_name,f'{vdi+1}.jpg'))
            plt.show()
        for key,pkl_name_selected in pkl_name.items():
            document_name='./projects/pkl/202310172011-4644-4200'
            file_path = os.path.join(document_name, pkl_name_selected)
            with open(file_path, 'rb') as file:
                if key==3:continue
                if pkl_name_selected!='ctrl_vehs.pkl':
                    step=0
                    while True:
                        try:#9.24新加了保存x,y及后面的信息,0103-4530-3707后，J_value替换成了eval_value，有J值，eval_safe和eval_traffic值
                            data=pickle.load(file)  #if key==3: print(data['i'],data['all_vehs_pos'][0]['id'],data['allvehs_info'][0].id)
                            def process_veh_data():
                                if key==2 and data['veh_id']=='ego':#npc144,ego
                                    print(data['i'],data['veh_id'],'ctrl_value:',[item.tolist() for item in data['ctrl_value']],'x:',data['x'],'y:',data['y'],'u:',data['u'],'phi:',data['phi'],
                                       'lane_id:',data['lane_id'],'dis_to_link:',data['dis_to_link'],'eval_value:',data['eval_value'],'selected_id:',data['selected_id'],'\n','vehs_relation_lane1:',data['vehs_relation']['v1l'],data['vehs_relation']['v1f'],'\n','vehs_relation_lane2:',data['vehs_relation']['v2l'],data['vehs_relation']['v2f'],'\n','vehs_relation_lane3:',data['vehs_relation']['v3l'],data['vehs_relation']['v3f'])
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
                    plot()

                if pkl_name_selected=='ctrl_vehs.pkl':
                    data = pickle.load(file)
                    print(data['ctrl_vehs'])

    def simulation_start(self,stub):
        startResp = stub.Start(simulation_pb2.StartSimulationReq(task_id=self.task_id, record_id=self.record_id),
                               metadata=self.metadata)
        self.simulation_id = startResp.simulation_id

    def step(self,stub):
        try:
            stepResult = stub.NextStep(simulation_pb2.NextStepReq(
                simulation_id=self.simulation_id), metadata=self.metadata)
            print('stepResult:', stepResult.state.progress)
            if (stepResult.state.progress <= 0) or (stepResult.state.progress >= 100):
                return True
                print(f"仿真结束,状态：{stepResult.state.msg}")
            else:return False
        except Exception as e:print('step_error:', e)
        print('*' * 100)

    def simulation_stop(self,stub):

        stub.Stop(simulation_pb2.StopSimulationReq(
            simulation_id=self.simulation_id), metadata=self.metadata)
        result = stub.GetResults(simulation_pb2.GetResultsReq(
            simulation_id=self.simulation_id), metadata=self.metadata)

    def get_y_list(self,stub,link_id,point):
        lanes=self.getLinkinfo(stub,link_id).link.ordered_lanes[1:]
        idx=min(range(len(lanes[0].center_line)),key=lambda index:abs(lanes[0].center_line[index].x-point.x))
        y_list=[item.center_line[idx].y for item in lanes]
        print(f'y_list_idx:{idx},y_list:{y_list},center_line_x:{lanes[0].center_line[idx].x},veh_point:{point.x, point.y}')
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
        print(f"control_value:{control_value}")
        vehicleControleReult = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
            simulation_id=self.simulation_id, vehicle_id=vehicle_id,
            lon_acc=control_value[0][0], ste_wheel=control_value[1][0]), metadata=self.metadata)
        checkError(vehicleControleReult.error)

    def get_veh_around(self, stub, vehicle_id,all_vehs_pos)->tuple:
        """
        1.整个地图的车辆
        2.根据距离筛选出最近的10辆
        """
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

def get_ctrl_vehs(all_vehs_pos, l_num=9,f_num=3):
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
    vehs_l_id_nearest=[item['id'] for item in vehs_l_nearest]
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
    # Cosim().run()
    Cosim().load_pkl()
    def test_eval_value(Surr):
        import matplotlib.pyplot as plt
        import numpy as np
        ego_dict = dict( Length=4.5, Width=1.8,Weight=1800,max_acc=2.5,max_dec=7)
        Surr = Surr('0',ego_dict)
        path1 = [np.ones(40) * -3.2, np.arange(0, 20, 0.5)]
        path2=[np.zeros(40),np.arange(0,20,0.5)]
        path3 = [np.ones(40) * 3.2, np.arange(0, 20, 0.5)]
        Surr.matched_static_path={1:path1,2:path2,3:path3}
        ego_simple = dict(id='0', x=0, y=0, u=15, phi=np.pi / 2, Length=4.5, Width=1.8, dis_to_link_end=100)
        ego = dict(id='0',x=0, y=0, u=15, phi=np.pi / 2,Length=4.5,Width=1.8,dis_to_link_end=100,lane_id='lk2',d=3.2,lane_rad=np.pi / 2,ego=ego_simple)
        def update_ego(**kwargs):
            for key,value in kwargs.items():
                exec('Surr.{0}=value'.format(key),{'Surr':Surr,'value':value})
        update_ego(**ego)
        car1 = dict(id='1',x=-3.2, y=5, u=25, phi=np.pi / 2,Length=4.5,Width=1.8)
        car1['dis_to_link']=100-car1['y']
        car2 = dict(id='2',x=-3.2, y=-10, u=15, phi=np.pi / 2,Length=4.5,Width=1.8)
        car2['dis_to_link'] = 100 - car2['y']
        car3 = dict(id='3', x=0, y=15, u=20, phi=np.pi / 2,Length=4.5,Width=1.8)
        car3['dis_to_link'] = 100 - car3['y']
        car4 = dict(id='4', x=0, y=-10, u=15, phi=np.pi / 2,Length=4.5,Width=1.8)
        car4['dis_to_link'] = 100 - car4['y']
        car5 = dict(id='5', x=3.2, y=5, u=30, phi=np.pi / 2,Length=4.5,Width=1.8)
        car5['dis_to_link'] = 100 - car5['y']
        car6 = dict(id='6', x=3.2, y=-10, u=15, phi=np.pi / 2,Length=4.5,Width=1.8)
        car6['dis_to_link'] = 100 - car6['y']
        Surr.matched_static_path_veh={1:[car1,car2],2:[car3,car4],3:[car5,car6]}
        eval_value=Surr.eval_function()
        Surr.get_track_speed()
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
