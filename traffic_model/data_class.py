from traffic_model.traffic_model import Surr,IDC_decision
from dataclasses import dataclass,field
from typing import Dict
from traffic_model.vehicle_data import traffic_info,veh_info

@dataclass
class data:
    surr: Surr
    IDC: IDC_decision
    veh_info: veh_info
    ego_veh_name: traffic_info

    def updata_data(self):
        pass


if __name__=='main':
    print('ok')