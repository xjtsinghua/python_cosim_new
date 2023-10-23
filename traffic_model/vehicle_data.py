from dataclasses import dataclass, field
from typing import Dict, List
from dataclasses_json import dataclass_json
import time



@dataclass
class traffic_info:
    id:str='xff'
    Length:float=4.5
    Width:float=1.5
    Height:float=1.5
    Weight:float=1800
    max_acc:float=2.5252
    max_dec: float = 7

    static_paths:List[str]=field(default_factory=list)

    x: float = 0
    y: float = 0
    phi:float = 0

    dis_to_link_end:float = 0
    link_id: str = ''
    lane_id:str=''

    u:float = 0
    lon_acc:float = 0
    timestamp:int=0






@dataclass
class veh_info:

    id: str = 'xff'
    Length: float = 4.5
    Width: float = 1.5
    Height: float = 1.5
    Weight: float = 1800
    max_acc: float = 2.5252
    max_dec: float = 7

    static_paths: List[str] = field(default_factory=list)

    x: float = 0
    y: float = 0
    phi: float = 0

    dis_to_link_end: float = 0
    link_id: str = ''
    lane_id: str = ''

    u: float = 0
    lon_acc: float = 0
    timestamp: int = 0

def __get_info():
    pass



if __name__=='__main__':

    print(traffic_info.id)
    print(veh_info.id)