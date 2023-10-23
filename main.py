import grpc
import copy

from risenlighten.lasvsim.process_task.api.cosim.v1 import simulation_pb2
from risenlighten.lasvsim.process_task.api.cosim.v1 import simulation_pb2_grpc
from dataclasses import dataclass
import time
import random
import numpy as np
import math
from traffic_model.monitor import Monitor
from traffic_model.traffic_model import Surr, IDC_decision
from traffic_model.data_class import data
import csv
import datetime
import os
import json
import pickle
from typing import List


@dataclass
class parametes_set:
    desir_dis: float = 50
    veh_num: int = 10

    def update_decay_param(self, value):
        self.desir_dis = value


@dataclass
class co_sim:
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjEzLCJvaWQiOjEwMSwibmFtZSI6IueGiuadsCIsImlkZW50aXR5Ijoibm9ybWFsIiwicGVybWlzc2lvbnMiOlsidGFzay50YXNrLnB1YmxpYy5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wdWJsaWMuQUNUSU9OX0NPUFkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBPUlQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9BREQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fQ09QWSIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fUkVQTEFZIiwidGFzay50YXNrLnByaXZhdGUuQUNUSU9OX1JFUE9SVCIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwidGFzay50YXNrLnBlcnNvbmFsLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucGVyc29uYWwuQUNUSU9OX1JFUE9SVCIsInJlc291cmNlLnZlaGljbGUucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudmVoaWNsZS5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudmVoaWNsZS5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS52ZWhpY2xlLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2UudmVoaWNsZS5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucHVibGljLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9BREQiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2Vuc29yLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnNlbnNvci5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnB1YmxpYy5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VQREFURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9WSUVXIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX0FERCIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS5tYXAucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UubWFwLnBlcnNvbmFsLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnNjZW5hcmlvLnByaXZhdGUuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wcml2YXRlLkFDVElPTl9DT1BZIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS50cmFmZmljX2Zsb3dfY29uZmlnLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9ET1dOTE9BRCIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RPV05MT0FEIl0sImlzcyI6InVzZXIiLCJzdWIiOiJMYXNWU2ltIiwiZXhwIjoxNjk3NzcyMjM2LCJuYmYiOjE2OTcxNjc0MzYsImlhdCI6MTY5NzE2NzQzNiwianRpIjoiMTMifQ.cPrbpgBcIUKQLaUaNIHLMipGXavtX7gCoGpYXm5GFK8'
    metadata = [('authorization',
                 'Bearer ' + token)]
    http = 'qianxing-grpc.risenlighten.com:80'
    # http='127.0.0.1:8290'  #桌面端lasvsim连接grpc用
    max_recv_msg_size = 20 * 1024 * 1024  # 20M
    # 创建channelArguments对象来设置选项
    channel_args = [('grpc.max_receive_message_length', max_recv_msg_size), ('grpc.default_timeout', 100)]
    params = parametes_set()

    # 复杂交通:[4070,2035]
    # 水平复杂场景：[4084,2070]
    # 垂直场景：[4085,2075]
    # 金枫大道:[4092,2100]
    # U型道路:[4184,2439]
    # N型道路:[4185,2445]
    # S型道路:[4186,2451]
    # U型道路:[4214,2628]
    # 水平三道路:[4224,2866]
    # 水平三道路新复杂:[4244,2891-2896]
    # 水平三道路:[4345,3341-3346]
    # 水平三道路-长-自车位置距起点100m:[4361:3383-3392]
    # 水平三道路-短-自车位置距起点100m:[4362:3395,3398,3401,3402]
    # 水平三道路-短-自车位置距起点100m:[4393:3488,3489,3491,3492,3495,3496,3497]

    task_id = 4393
    record_id = 3497
    vehicle_id = 'ego'
    vehicle_id_list = [vehicle_id]
    vehicle_id_list_new = []
    progress_times = 0
    choosed_list = []
    simulation_id = ''
    veh_id_around = []
    veh_around_obj = []
    all_vehs_pos = []
    allvehs_info = []

    veh_info = {}
    agent = {}
    control_value = []
    ctrl_value = {}
    IDC_value = {}
    old_file_name = ''
    old_file_path = ''
    save_time = 0
    old_time = None
    data_num = 0
    surr = None
    used_names_list = []

    step_i = 0
    new_turn_step_i = 0

    def simulation_start(self, stub):
        startResp = stub.Start(simulation_pb2.StartSimulationReq(task_id=self.task_id, record_id=self.record_id),
                               metadata=self.metadata)
        self.simulation_id = startResp.simulation_id

    def update_info(self, stub, vehicle_id):
        UpdateVehicleInfoReult = stub.UpdateVehicleInfo(simulation_pb2.UpdateVehicleInfoReq(
            simulation_id=self.simulation_id, vehicle_id=vehicle_id,
        ), metadata=self.metadata)

    def simulation_stop(self, stub):

        stub.Stop(simulation_pb2.StopSimulationReq(
            simulation_id=self.simulation_id), metadata=self.metadata)
        result = stub.GetResults(simulation_pb2.GetResultsReq(
            simulation_id=self.simulation_id), metadata=self.metadata)
        # print(f"仿真结束,结果:{result.results}")

    def get_allvehs_list(self, stub):
        """
        1.得到地图所有车辆id，得到所有车辆的位置信息
        2.根据控制的车辆的位置，得到车辆一定范围内周围车辆，再根据周围车辆id获得其所在车道，dis_to_end信息，确定控制车辆与周车的位置关系。
        3.输出结果
        :param stub:
        :return:
        """
        vehicle_id_list = stub.GetVehicleIdList(simulation_pb2.GetVehicleIdListReq(
            simulation_id=self.simulation_id), metadata=self.metadata)

        return vehicle_id_list

    # todo 修改方法
    def get_veh_around(self, stub, vehicle_id):
        """
        1.整个地图的车辆
        2.根据距离筛选出最近的10辆

        :param stub:
        :param vehicle_id:
        :return:
        """
        # 获取自车以及自车的周车列表
        vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(
            simulation_id=self.simulation_id, vehicle_id=vehicle_id), metadata=self.metadata)
        self.checkError(vehicle.error)

        veh_around = [[item['id'], item['x'], item['y'], item['lane_id']] for item
                      in self.all_vehs_pos if item['id'] != vehicle_id]

        nearest_veh_id_around = []

        if veh_around:
            self.veh_id_around = list(map(lambda x: x[0], veh_around))
            # 得到指定范围内周车

            nearest_vehs = self.get_nearest_vehs(vehicle, veh_around, self.params.desir_dis, self.params.veh_num)
            nearest_veh_id_around = list(map(lambda x: x[0], nearest_vehs))

            if len(nearest_veh_id_around) > 30:
                self.params.update_decay_param(value=30)
            else:
                self.params.update_decay_param(value=50)

        return vehicle, nearest_veh_id_around

    def control_near_objs(self, stub, nearest_veh_id_around, surr, i):
        # 筛选相同路段的车辆
        if nearest_veh_id_around:

            leader_list = surr.get_leader_list()
            follower_list = surr.get_follower_list()
            # print(f'leader_list:{leader_list}')
            # print(f'follower_list:{follower_list}')
            choosed_list = []

            leader_info_list = surr.get_veh_acc(leader_list, target_v=6)
            print(f'leader_info_list:{leader_info_list}')
            follower_info_list = surr.get_veh_acc(follower_list, target_v=3)
            print(f'follower_info_list:{follower_info_list}')

            if i < 100 and i % 10 == 0:
                if len(follower_info_list) > 2:
                    choosed_list = random.sample(follower_info_list, 2)
                elif len(leader_info_list) > 2:
                    choosed_list = random.sample(leader_info_list, 2)
                else:
                    choosed_list = leader_info_list
            if i >= 100:
                choosed_list = leader_info_list + follower_info_list
            choosed_list = choosed_list if choosed_list else follower_info_list
            print(f'choosed_list:{choosed_list}')
            for leader_info in choosed_list:
                leader_veh = leader_info[0]
                leader_acc = leader_info[1]
                vehicleControleReult_others = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
                    simulation_id=self.simulation_id, vehicle_id=leader_veh,
                    lon_acc=leader_acc, ste_wheel=0), metadata=self.metadata)
                self.checkError(vehicleControleReult_others.error)

    def control_veh(self, stub, vehicle_id, control_value):
        start_time = time.time()

        # if i>50:
        #     control_value=[[-2],[0.1]]
        print(f"control_value:{control_value}")

        vehicleControleReult = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
            simulation_id=self.simulation_id, vehicle_id=vehicle_id,
            lon_acc=control_value[0][0], ste_wheel=control_value[1][0]), metadata=self.metadata)
        self.checkError(vehicleControleReult.error)

        end_time = time.time()
        print(f'单步control_veh花费时间:{round((end_time - start_time), 2)}s')
        # 状态不正确，结束

    def step(self, stub):
        try:
            stepResult = stub.NextStep(simulation_pb2.NextStepReq(
                simulation_id=self.simulation_id), metadata=self.metadata)
        except Exception as e:
            print('step_error:', e)

        print('*' * 100)
        print('stepResult:', stepResult.state.progress)
        if (stepResult.state.progress <= 0) or (stepResult.state.progress >= 100):
            # or progress_times>10:
            return True

            print(f"仿真结束,状态：{stepResult.state.msg}")
        else:
            return False

    def getallvehicle(self, stub):
        all_vehicles = stub.GetAllVehicles(simulation_pb2.GetAllVehiclesReq(
            simulation_id=self.simulation_id), metadata=self.metadata)
        self.checkError(all_vehicles.error)
        return all_vehicles

    # todo 加耗时计算
    def run_singel(self, stub, vehicle_id):

        vehicle, nearest_veh_id_around = self.get_veh_around(stub, vehicle_id)
        nearest_veh_id_around = nearest_veh_id_around + [vehicle_id]

        input_params = self.process_input_params( vehicle, nearest_veh_id_around)

        veh_list_edited = input_params['veh_list_edited']
        # if not bool(traffic_info):
        #     break
        # 调用机动车行为模型
        surr = Surr(input_params, self.step_i, self.new_turn_step_i)
        self.surr = surr
        control_value = surr.update(input_params, veh_list_edited)
        self.new_turn_step_i = surr.IDC.turn_step_i if surr.IDC.turn_step_i > 0 else 0
        if self.new_turn_step_i > 0:
            print(f"self.new_turn_step_i:{self.new_turn_step_i}")
            print('0' * 100)
        self.control_value = control_value
        self.control_veh(stub, vehicle_id=vehicle_id, control_value=control_value)
        return vehicle,control_value,surr.IDC

    def run(self, iter_num=300):

        with grpc.insecure_channel(self.http, options=self.channel_args) as channel:
            stub = simulation_pb2_grpc.CosimStub(channel)
            self.simulation_start(stub)
            for i in range(iter_num):
                self.step_i = i
                start_time = time.time()
                result = self.step(stub)
                if result:
                    break
                if i > 1:
                    self.all_vehs_pos, self.allvehs_info = self.get_allvehs_position(stub)
                    if not self.vehicle_id_list_new:
                        ctrl_l_list, ctrl_f_list = self.get_ctrl_vehs()
                        self.vehicle_id_list_new = self.vehicle_id_list + ctrl_l_list + ctrl_f_list
                        # self.vehicle_id_list_new = self.vehicle_id_list + ctrl_l_list
                    # else:
                    #     self.vehicle_id_list_new = self.vehicle_id_list
                    for item in self.vehicle_id_list_new:
                        found_item = [veh for veh in self.all_vehs_pos if veh['id'] == item]
                        try:
                            found_item_info = found_item[0]
                            if found_item_info['lane_id'] == '':
                                self.vehicle_id_list_new.remove(item)
                        except:
                            self.vehicle_id_list_new.remove(item)

                    print('控制车辆列表；', self.vehicle_id_list_new)

                    # self.ctrl_vehs_name.append(self.vehicle_id_list_new)

                    idx = 0

                    for vehicle_id in self.vehicle_id_list_new:
                        start_time_single = time.time()
                        print(f'当前控制车辆id:{vehicle_id},第{idx + 1}辆，共{len(self.vehicle_id_list_new)}辆')

                        vechicle,ctrl_value,IDC=self.run_singel(stub, vehicle_id)

                        timestamp = i * 0.1
                        print(f'timestamp:{timestamp}')

                        end_time_single = time.time()
                        idx += 1
                        print(f'单个step中单辆车控制花费时间:{round((end_time_single - start_time_single), 2)}s')

                        self.agent[vehicle_id] = vechicle.vehicle.info.moving_info
                        self.ctrl_value[vehicle_id] = [item[0] for item in ctrl_value]
                        self.IDC_value[vehicle_id]=IDC



                    end_time_for_model = time.time()

                    print(f'单个step中交通流模型花费时间:{round((end_time_for_model - start_time), 2)}s')

                    # 保存数据
                    self.save_data()

                # if i==8:
                #     self.simulation_stop(stub)
                print('progress_i:', i)
            self.simulation_stop(stub)

    # todo 数据重新梳理
    def save_data(self):
        self.save_time += 1

        if self.save_time % 2 == 0:
            self.data_num += 1
            current_time = time.strftime("%Y%m%d%H%M%S").replace(':', '')
            files_path = self.built_file_dir(current_time)

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

    def set_data_title(self) -> List[str]:
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
            'acc_w',
            'safe_LC',
            'safe_pass',
            'u_cal',
            'u_des',
            'acce_des',
            'theta_des',
            'static_path_num',
            'static_path_index_des',
            'lane_index',
            'lane_id',
            'lat_bias_des',
            'lateral_position',
            'eva_traffic',
            'eva_safety',
            'dis_2_veh_cl',
            'dis_2_veh_cf',
            'dis_2_veh_ll',
            'dis_2_veh_rl',
            'dis_2_veh_lf',
            'dis_2_veh_rf'

        ]

        return title

    def get_vehs_info_around(self,veh_id):
        veh_relation_info=self.IDC_value[veh_id]

        rel_str_list=np.array(['veh_cl','veh_cf','veh_ll','veh_lf','veh_rl','veh_rf'])


        dis_2_veh_cl = veh_relation_info.vehs_relation['veh_cl']  # 'dis_2_veh_cl'
        dis_2_veh_cf = veh_relation_info.vehs_relation['veh_cf']  # 'dis_2_veh_cf'
        dis_2_veh_ll = veh_relation_info.vehs_relation['veh_ll']  # 'dis_2_veh_ll'
        dis_2_veh_rl = veh_relation_info.vehs_relation['veh_rl']  # 'dis_2_veh_rl'
        dis_2_veh_lf = veh_relation_info.vehs_relation['veh_lf']  # 'dis_2_veh_lf'
        dis_2_veh_rf = veh_relation_info.vehs_relation['veh_rf']  # 'dis_2_veh_rf'


        dis_dict = dict(dis_2_veh_cl=dis_2_veh_cl,
                        dis_2_veh_cf=dis_2_veh_cf,
                        dis_2_veh_ll=dis_2_veh_ll,
                        dis_2_veh_rl=dis_2_veh_rl,
                        dis_2_veh_lf=dis_2_veh_lf,
                        dis_2_veh_rf=dis_2_veh_rf
                        )
        return dis_dict

    # todo 所有的surr.IDC都对应不上
    def set_data(self, ego_name: str) -> List[str]:

        safe_pass = self.IDC_value[ego_name].safe_pass
        safe_LC = self.IDC_value[ego_name].safe_LC
        u_des = self.IDC_value[ego_name].u_des
        u_cal = self.IDC_value[ego_name].u_cal

        acce_des = self.ctrl_value[ego_name][0]  # 'acce_des',
        theta_des = self.ctrl_value[ego_name][1]  # 'theta_des',

        static_path_num = len(self.IDC_value[ego_name].static_paths),  # 'static_path_num',
        static_path_index_des = self.IDC_value[ego_name].path_idx_des,  # 'static_path_index_des',

        lane_index = self.IDC_value[ego_name].current_idx  # 'lane_index',
        lat_bias_des = self.IDC_value[ego_name].bias_lat
        lateral_position = 0  # 'lateral_position',
        eva_traffic = self.IDC_value[ego_name].value_traffic  # 'eva_traffic',
        eva_safety = self.IDC_value[ego_name].value_safety  # 'eva_safety',

        dis_dict = self.get_vehs_info_around(ego_name)

        save_data = [

            self.save_time,  # 'time',
            self.agent[ego_name].position.point.x,  # 'x',
            self.agent[ego_name].position.point.y,  # 'y',
            self.agent[ego_name].position.phi,  # 'phi',
            self.agent[ego_name].u,  # 'u',
            self.agent[ego_name].v,  # 'v',
            self.agent[ego_name].w,  # 'w',
            self.agent[ego_name].lon_acc,  # 'acc_lon',
            self.agent[ego_name].lat_acc,  # 'acc_lat',
            self.agent[ego_name].w_acc,  # 'acc_w',
            safe_LC,
            safe_pass,
            u_cal,
            u_des,  # 'u_des'
            acce_des,  # 'acce_des',
            theta_des,  # 'theta_des',
            static_path_num,  # 'static_path_num',
            static_path_index_des,  # 'static_path_index_des',
            lane_index,  # 'lane_index',
            self.agent[ego_name].position.lane_id,  # 'lane_id',
            lat_bias_des,
            lateral_position,  # 'lateral_position',
            eva_traffic,  # 'eva_traffic',
            eva_safety,  # 'eva_safety',
            dis_dict['dis_2_veh_cl'],  # 'dis_2_veh_cl',
            dis_dict['dis_2_veh_cf'],  # 'dis_2_veh_cf',
            dis_dict['dis_2_veh_ll'],  # 'dis_2_veh_ll',
            dis_dict['dis_2_veh_rl'],  # 'dis_2_veh_rl',
            dis_dict['dis_2_veh_lf'],  # 'dis_2_veh_lf',
            dis_dict['dis_2_veh_rf'],  # 'dis_2_veh_rf',

        ]
        return save_data

    def cul_sec(self, time: str):
        assert len(time) == 14, 'time str lenth is not 14'
        hour = int(time[8:10]) * 60 * 60
        minute = int(time[10:12]) * 60
        second = int(time[12:14])

        total_second = hour + minute + second
        return total_second

    def built_file_dir(self, current_time):

        folder_name = current_time.replace(':', '')[:-2]  # 移除时间字符串中的冒号，用于文件夹命名
        folder_path = f"./projects/data/{folder_name}-{self.task_id}-{self.record_id}"  # 文件夹路径

        # 创建一个以当前时间命名的文件夹
        if self.old_file_name == '':
            os.makedirs(folder_path)
            self.old_file_name = folder_name
            self.old_time = time.time()
            self.old_file_path = folder_path

        # 等待10分钟
        delay = 30  # 10分钟
        delay_second = delay * 60
        # current_time_tosec=self.cul_sec(current_time)
        # old_time_tosec=self.cul_sec(self.old_file_name)
        # if current_time_tosec - old_time_tosec<0:
        #     current_time_tosec+=3600
        new_time = time.time()

        # 检查当前时间和文件夹名称的时间差是否超过20分钟
        if new_time - self.old_time >= delay_second:
            # 如果超过20分钟，创建以当前时间为名称的文件夹
            # 获取当前时间并重新命名文件夹
            # new_folder_path = folder_path
            # os.makedirs(new_folder_path)
            # os.makedirs(self.old_file_path)
            print("创建成功！")
            self.old_time = new_time
            # return new_folder_path
            return self.old_file_path
        else:
            print(f"未超过{delay}分钟，无需创建。")
            return self.old_file_path

    def get_ctrl_vehs(self, l_num=8, f_num=1):

        key_veh_id = 'ego'

        key_veh = list(filter(lambda d: d['id'] == key_veh_id, self.all_vehs_pos))[0]

        vehs_l = [item['id'] for item in self.all_vehs_pos if
                  -100 < item['dis_to_link_end'] - key_veh['dis_to_link_end'] < 0
                  and item['lane_id'].split('_')[:2] == key_veh['lane_id'].split('_')[:2]]

        vehs_f = [item['id'] for item in self.all_vehs_pos if
                  0 < item['dis_to_link_end'] - key_veh['dis_to_link_end'] < 100
                  and item['lane_id'].split('_')[:2] == key_veh['lane_id'].split('_')[:2]]

        return vehs_l[:l_num], vehs_f[:f_num]

    # todo 获得所有车辆位置
    def get_allvehs_position(self, stub):
        all_vehicles = self.getallvehicle(stub)

        allvehs_info_raw = [item.info for item in all_vehicles.vehicles]
        key_veh_id = 'ego'

        key_veh = list(filter(lambda d: d.id == key_veh_id, allvehs_info_raw))[0]

        allvehs_info = [item for item in allvehs_info_raw
                        if item.moving_info.position.link_id == key_veh.moving_info.position.link_id]
        allvehs_position = [{'id': item.id,
                             'x': item.moving_info.position.point.x,
                             'y': item.moving_info.position.point.y,
                             'dis_to_link_end': item.moving_info.position.dis_to_link_end,
                             'lane_id': item.moving_info.position.lane_id}
                            for item in allvehs_info]
        return allvehs_position, allvehs_info_raw

    def checkError(self, err):
        if err is None:
            return False

        if err.code != 0:
            print(err.msg)
            return True
        return False

    def get_allvehs_info_new(self, veh_list):
        all_vehs = {}
        veh_list_edited = copy.deepcopy(veh_list)

        all_vehs = [item for item in self.allvehs_info if item.id in veh_list]

        return all_vehs, veh_list_edited

    # 主要获取车辆真实位置信息
    def get_allvehs_info(self, stub, veh_list):
        # veh_list = stub.GetVehicleIdList(simulation_pb2.GetVehicleIdListReq(
        #     simulation_id=startResp.simulation_id))
        # print(veh_list)

        all_vehs = {}
        veh_list_edited = copy.deepcopy(veh_list)

        for veh in veh_list:
            vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(
                simulation_id=self.simulation_id, vehicle_id=veh), metadata=self.metadata)

            result = self.checkError(vehicle.error)
            if result:
                print(f'{veh} is not in map')
                veh_list_edited.remove(veh)

                continue
            id = vehicle.vehicle.info.id
            all_vehs[id] = [vehicle.vehicle.info.moving_info, vehicle.vehicle.info.base_info]

        return all_vehs, veh_list_edited

    def process_input_params(self,  vehicle, veh_list):
        start_time = time.time()
        vehicle_info = vehicle.vehicle.info
        print("=" * 100)

        static_paths = vehicle_info.static_path
        print("=" * 100)
        # print('static_paths:', static_paths)

        # all_moving_objs,veh_list_edited = self.get_allvehs_info(stub, veh_list)
        all_moving_objs, veh_list_edited = self.get_allvehs_info_new(veh_list)

        # print(all_moving_objs)

        input_params = {'veh_info': vehicle_info, 'traffic_info': all_moving_objs, 'static_paths': static_paths,
                        'veh_list_edited': veh_list_edited}

        self.veh_info = vehicle_info.moving_info

        end_time_process_input = time.time()
        print(f'单个step中process_input花费时间:{round((end_time_process_input - start_time), 2)}s')

        return input_params

    def order_paths(self, static_paths):

        y_list = []
        for path in static_paths:
            y_value = path.point[-1].y
            y_list.append(y_value)
        # idx1 = y_list.index(max(y_list))
        idx1 = np.array(y_list).argmax()
        new_paths = []
        new_paths.append(static_paths[idx1])
        m_paths = copy.deepcopy(y_list)
        m_paths = m_paths.pop(y_list[idx1])
        if len(m_paths) == 1:
            # idx2 = y_list.index(min(y_list))
            idx2 = np.array(y_list).argmin()
            new_paths.append(static_paths[idx2])

        y_2 = max(m_paths)
        idx2 = y_list.index(y_2)
        new_paths.append(static_paths[idx2])

        while len(m_paths) > 1:
            y_value_next = max(m_paths)
            idx_next = y_list.index(y_value_next)
            new_paths.append(static_paths[idx_next])
            m_paths.pop(y_value_next)

        return new_paths

    def get_nearest_vehs(self, veh_info, around_vehs, desir_dis=50, veh_num=10):
        vehicle_id = veh_info.vehicle.info.id
        host_veh_point = veh_info.vehicle.info.moving_info.position.point
        host_veh_point = np.array([host_veh_point.x, host_veh_point.y])
        lane_id_list = veh_info.vehicle.info.moving_info.position.lane_id.split('_')
        # print(f'lane_id_list:{lane_id_list},len为{len(lane_id_list)}')
        seg = lane_id_list[0]

        # 采用欧式距离计算两点距离
        # nearest_vehs = [item for item in around_vehs
        #                 if np.linalg.norm(host_veh_point - np.array(
        #         [item[1], item[2]])) < desir_dis]

        # math.sqrt((item[1] - host_veh_point[0]) ** 2 +(item[2] - host_veh_point[1]) ** 2)

        # 方法二，采用曼哈顿距离计算两点距离
        if len(lane_id_list) > 1:
            link = lane_id_list[1]
            nearest_vehs = [item for item in around_vehs
                            if (abs(item[1] - host_veh_point[0]) + abs(item[2] - host_veh_point[1])) < desir_dis
                            and item[3].split('_')[0] == seg and item[3].split('_')[1] == link]
        else:
            nearest_vehs = [item for item in around_vehs
                            if (abs(item[1] - host_veh_point[0]) + abs(item[2] - host_veh_point[1])) < desir_dis
                            and item[3].split('_')[0] == seg]
        nearest_vehs = sorted(nearest_vehs,
                              key=lambda item: (abs(item[1] - host_veh_point[0]) + abs(item[2] - host_veh_point[1])))[
                       :min(len(nearest_vehs), veh_num)]
        return nearest_vehs

    def get_same_seg_vehs_new(self, host_veh, around_vehs, traffic_info):
        host_veh_link = host_veh.moving_info.position.link_id
        list_traffic = [item.id for item in traffic_info]

        print(f'same_seg_vehs中traffic_info:{list_traffic},around_vehs:{around_vehs}')
        new_vehs = [item.id for item in traffic_info if item.moving_info.position.link_id == host_veh_link
                    and item.id in around_vehs]

        # print(list_traffic,new_list)
        print(f"new_vehs:{new_vehs}")
        return new_vehs

    def get_same_seg_vehs(self, host_veh, around_vehs, traffic_info):
        host_veh_link = host_veh.moving_info.position.link_id

        list_traffic = list(traffic_info.keys())
        print(f'same_seg_vehs中traffic_info:{list_traffic},around_vehs:{around_vehs}')
        new_vehs = [x for x in around_vehs if traffic_info[x][0].position.link_id == host_veh_link]

        # print(list_traffic,new_list)
        print(f"new_vehs:{new_vehs}")
        return new_vehs


def run():
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjEzLCJvaWQiOjEwMSwibmFtZSI6IueGiuadsCIsImlkZW50aXR5Ijoibm9ybWFsIiwicGVybWlzc2lvbnMiOlsidGFzay50YXNrLnB1YmxpYy5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wdWJsaWMuQUNUSU9OX0NPUFkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucHVibGljLkFDVElPTl9SRVBPUlQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9BREQiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fQ09QWSIsInRhc2sudGFzay5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJ0YXNrLnRhc2sucHJpdmF0ZS5BQ1RJT05fUkVQTEFZIiwidGFzay50YXNrLnByaXZhdGUuQUNUSU9OX1JFUE9SVCIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fVklFVyIsInRhc2sudGFzay5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwidGFzay50YXNrLnBlcnNvbmFsLkFDVElPTl9SRVBMQVkiLCJ0YXNrLnRhc2sucGVyc29uYWwuQUNUSU9OX1JFUE9SVCIsInJlc291cmNlLnZlaGljbGUucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudmVoaWNsZS5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudmVoaWNsZS5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS52ZWhpY2xlLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnZlaGljbGUucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2UudmVoaWNsZS5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnZlaGljbGUucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucHVibGljLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9BREQiLCJyZXNvdXJjZS5zZW5zb3IucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2Vuc29yLnByaXZhdGUuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnNlbnNvci5wcml2YXRlLkFDVElPTl9VU0UiLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zZW5zb3IucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnNlbnNvci5wZXJzb25hbC5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnB1YmxpYy5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VQREFURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9WSUVXIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX0FERCIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9VUERBVEUiLCJyZXNvdXJjZS5tYXAucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2UubWFwLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLm1hcC5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UubWFwLnBlcnNvbmFsLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnNjZW5hcmlvLnByaXZhdGUuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wcml2YXRlLkFDVElPTl9DT1BZIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fREVMRVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fVVNFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX1ZJRVciLCJyZXNvdXJjZS5zY2VuYXJpby5wZXJzb25hbC5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2Uuc2NlbmFyaW8ucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHVibGljLkFDVElPTl9WSUVXIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wdWJsaWMuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fVVBEQVRFIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wcml2YXRlLkFDVElPTl9ERUxFVEUiLCJyZXNvdXJjZS50cmFmZmljX2Zsb3dfY29uZmlnLnByaXZhdGUuQUNUSU9OX1VTRSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucHJpdmF0ZS5BQ1RJT05fQUREIiwicmVzb3VyY2UudHJhZmZpY19mbG93X2NvbmZpZy5wZXJzb25hbC5BQ1RJT05fVklFVyIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX1VQREFURSIsInJlc291cmNlLnRyYWZmaWNfZmxvd19jb25maWcucGVyc29uYWwuQUNUSU9OX0RFTEVURSIsInJlc291cmNlLm1hcC5wcml2YXRlLkFDVElPTl9ET1dOTE9BRCIsInJlc291cmNlLm1hcC5wdWJsaWMuQUNUSU9OX0RPV05MT0FEIl0sImlzcyI6InVzZXIiLCJzdWIiOiJMYXNWU2ltIiwiZXhwIjoxNjkyODg4NDUzLCJuYmYiOjE2OTI4MDIwNTMsImlhdCI6MTY5MjgwMjA1MywianRpIjoiMTMifQ.3-Z8llEMl0bSZ4MI1mUp9RmrStSPHCDH4ZuNo1X1DF8'

    metadata = [('authorization',
                 'Bearer ' + token)]
    # with grpc.insecure_channel('qianxing-grpc.risenlighten.com:80') as channel:

    http3 = 'https://qianxing.risenlighten.com/'
    http1 = '8.146.201.197:31244'
    # http='127.0.0.1:8290'  #桌面端lasvsim连接grpc用
    http = 'qianxing-grpc.risenlighten.com:80'
    http2 = 'http://qianxing-api.risenlighten.com'

    with grpc.insecure_channel(http) as channel:

        # 复杂交通:[4070,2035]
        # 水平复杂场景：[4084,2070]
        # 垂直场景：[4085,2075]
        # 金枫大道:[4092,2100]
        # U型道路:[4184,2439]
        # N型道路:[4185,2445]
        # S型道路:[4186,2451]
        # U型道路:[4214,2628]

        params = parametes_set()

        task_id = 4185
        record_id = 2445
        vehicle_id = 'ego'
        progress_times = 0
        choosed_list = []

        stub = simulation_pb2_grpc.CosimStub(channel)
        startResp = stub.Start(simulation_pb2.StartSimulationReq(task_id=task_id, record_id=record_id),
                               metadata=metadata)

        UpdateVehicleInfoReult = stub.UpdateVehicleInfo(simulation_pb2.UpdateVehicleInfoReq(
            simulation_id=startResp.simulation_id, vehicle_id=vehicle_id,
        ), metadata=metadata)
        # parametes_set=params.parametes_set()

        for i in range(250):

            start_time = time.time()

            # 获取自车以及自车的周车列表
            vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(
                simulation_id=startResp.simulation_id, vehicle_id=vehicle_id), metadata=metadata)
            checkError(vehicle.error)
            # print(vehicle)
            veh_around = vehicle.vehicle.around_moving_objs
            # veh_id_around=[x.id for x in veh_around]
            nearest_veh_id_around = []
            if veh_around:
                veh_id_around = list(map(lambda x: x.id, veh_around))
                # 得到指定范围内周车
                nearest_vehs = get_nearest_vehs(vehicle, veh_around, params.desir_dis, params.veh_num)
                nearest_veh_id_around = list(map(lambda x: x.id, nearest_vehs))

                if len(nearest_veh_id_around) > 30:
                    params.update_decay_param(value=30)
                else:
                    params.update_decay_param(value=50)

            stepResult = stub.NextStep(simulation_pb2.NextStepReq(
                simulation_id=startResp.simulation_id), metadata=metadata)
            print('*' * 100)
            print('stepResult:', stepResult.state.progress)

            end_time_nextstep = time.time()
            print(f'单个step中nextstep花费时间:{round((end_time_nextstep - start_time), 2)}s')

            if stepResult.state.progress == 1:
                progress_times += 1
            # if stepResult.state.progress in [0,200] or i>10:
            #     break
            #     tag=True

            if checkError(stepResult.error):
                break

            if i > 1:
                input_params = process_input_params(vehicle, stub, startResp, nearest_veh_id_around, metadata)
                # print(input_params)
                # control_value = [[-2], [0]]
                veh_info = input_params['veh_info']
                traffic_info = input_params['traffic_info']
                # if not bool(traffic_info):
                #     break
                # 筛选相同路段的车辆
                surr = Surr(input_params)

                new_vehs_list = []
                if nearest_veh_id_around:
                    new_vehs_list = get_same_seg_vehs(veh_info, nearest_veh_id_around, traffic_info)

                    print(f"考虑的周车的数量:{len(nearest_veh_id_around)},相同路段的周车数量:{len(new_vehs_list)}")

                    leader_list = surr.get_leader_list()
                    follower_list = surr.get_follower_list()
                    # print(f'leader_list:{leader_list}')
                    # print(f'follower_list:{follower_list}')

                    leader_info_list = surr.get_veh_acc(leader_list, target_v=6)
                    print(f'leader_info_list:{leader_info_list}')
                    follower_info_list = surr.get_veh_acc(follower_list, target_v=3)
                    print(f'follower_info_list:{follower_info_list}')

                    if i < 100 and i % 10 == 0:
                        if len(follower_info_list) > 2:
                            choosed_list = random.sample(follower_info_list, 2)
                        elif len(leader_info_list) > 2:
                            choosed_list = random.sample(leader_info_list, 2)
                        else:
                            choosed_list = leader_info_list
                    if i >= 100:
                        choosed_list = leader_info_list + follower_info_list
                    choosed_list = choosed_list if choosed_list else follower_info_list
                    print(f'choosed_list:{choosed_list}')
                    for leader_info in choosed_list:
                        leader_veh = leader_info[0]
                        leader_acc = leader_info[1]
                        vehicleControleReult_others = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
                            simulation_id=startResp.simulation_id, vehicle_id=leader_veh,
                            lon_acc=leader_acc, ste_wheel=0), metadata=metadata)
                        checkError(vehicleControleReult_others.error)

                control_value = surr.update(input_params, new_vehs_list)
                # if i>50:
                #     control_value=[[-2],[0.1]]
                print(f"control_value:{control_value}")

                vehicleControleReult = stub.SetVehicleControl(simulation_pb2.SetVehicleControlReq(
                    simulation_id=startResp.simulation_id, vehicle_id=vehicle_id,
                    lon_acc=control_value[0][0], ste_wheel=control_value[1][0]), metadata=metadata)
                checkError(vehicleControleReult.error)
            end_time_for_model = time.time()
            print(f'单个step中交通流模型花费时间:{round((end_time_for_model - start_time), 2)}s')

            print('progress_times:', progress_times)

            end_time = time.time()
            print(f'单个step花费时间:{round((end_time - start_time), 2)}s')
            # 状态不正确，结束
            if (stepResult.state.progress <= 0) or (stepResult.state.progress >= 100):
                # or progress_times>10:

                print(
                    f"仿真结束,状态：{stepResult.state.msg}")
                break

        print(f"id：{startResp.simulation_id}")

        stub.Stop(simulation_pb2.StopSimulationReq(
            simulation_id=startResp.simulation_id), metadata=metadata)
        result = stub.GetResults(simulation_pb2.GetResultsReq(
            simulation_id=startResp.simulation_id), metadata=metadata)
        # print(f"仿真结束,结果:{result.results}")


def checkError(err):
    if err is None:
        return False

    if err.code != 0:
        print(err.msg)
        return True
    return False


# 主要获取车辆真实位置信息
def get_allvehs_info(stub, startResp, veh_list, metadata):
    # veh_list = stub.GetVehicleIdList(simulation_pb2.GetVehicleIdListReq(
    #     simulation_id=startResp.simulation_id))
    # print(veh_list)

    all_vehs = {}

    for veh in veh_list:
        vehicle = stub.GetVehicle(simulation_pb2.GetVehicleReq(
            simulation_id=startResp.simulation_id, vehicle_id=veh), metadata=metadata)
        checkError(vehicle.error)
        id = vehicle.vehicle.info.id
        all_vehs[id] = [vehicle.vehicle.info.moving_info, vehicle.vehicle.info.base_info]

    return all_vehs


# todo 修改周车获取方法
def process_input_params(vehicle, stub, startResp, veh_list, metadata):
    start_time = time.time()
    vehicle_info = vehicle.vehicle.info
    # around_moving_objs = vehicle.vehicle.around_moving_objs
    print("=" * 100)
    # print('around_moving_objs:', around_moving_objs)

    static_paths = vehicle_info.static_path
    print("=" * 100)
    # print('static_paths:', static_paths)

    all_moving_objs = get_allvehs_info(stub, startResp, veh_list, metadata)
    # print(all_moving_objs)

    input_params = {'veh_info': vehicle_info, 'traffic_info': all_moving_objs, 'static_paths': static_paths}

    end_time_process_input = time.time()
    print(f'单个step中process_input花费时间:{round((end_time_process_input - start_time), 2)}s')

    return input_params


def order_paths(static_paths):
    y_list = []
    for path in static_paths:
        y_value = path.point[-1].y
        y_list.append(y_value)
    idx1 = y_list.index(max(y_list))
    new_paths = []
    new_paths.append(static_paths[idx1])
    m_paths = copy.deepcopy(y_list)
    m_paths = m_paths.pop(y_list[idx1])
    if len(m_paths) == 1:
        idx2 = y_list.index(min(y_list))
        new_paths.append(static_paths[idx2])

    y_2 = max(m_paths)
    idx2 = y_list.index(y_2)
    new_paths.append(static_paths[idx2])

    while len(m_paths) > 1:
        y_value_next = max(m_paths)
        idx_next = y_list.index(y_value_next)
        new_paths.append(static_paths[idx_next])
        m_paths.pop(y_value_next)

    return new_paths


def get_nearest_vehs(veh_info, around_vehs, desir_dis=50, veh_num=10):
    host_veh_point = veh_info.vehicle.info.moving_info.position.point
    host_veh_point = np.array([host_veh_point.x, host_veh_point.y])

    nearest_vehs = [item for item in around_vehs
                    if np.linalg.norm(host_veh_point - np.array(
            [item.moving_info.position.point.x, item.moving_info.position.point.y])) < desir_dis]
    # nearest_vehs = [item for item in around_vehs
    #                 if math.sqrt((item.moving_info.position.point.x-host_veh_point.x)**2+
    #                              (item.moving_info.position.point.y-host_veh_point.y)**2) < desir_dis]
    nearest_vehs = sorted(nearest_vehs,
                          key=lambda item: math.sqrt((item.moving_info.position.point.x - host_veh_point[0]) ** 2 +
                                                     (item.moving_info.position.point.y - host_veh_point[1]) ** 2))[
                   :min(len(nearest_vehs), veh_num)]
    return nearest_vehs


def get_same_seg_vehs(host_veh, around_vehs, traffic_info):
    host_veh_link = host_veh.moving_info.position.link_id
    # list_traffic=[traffic_info[x][0].position.link_id for x in around_vehs]
    # new_list=[x for x in list_traffic if x==host_veh_link]
    new_vehs = [x for x in around_vehs if traffic_info[x][0].position.link_id == host_veh_link]
    # print(list_traffic,new_list)
    print(f"new_vehs:{new_vehs}")
    return new_vehs


def pyplot(data):
    pass


if __name__ == '__main__':
    # run()
    co_sim = co_sim()
    co_sim.run()
