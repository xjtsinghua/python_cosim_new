# This Python file uses the following encoding: utf-8
# coding=utf-8

# ===================================================================
# update the evaluation results.
# author: Chen Chen
# 2023.01.12 --------------
# separate the podar into the indepnedent file
# -------------------------
# ===================================================================

from math import sin, cos, pi, sqrt, atan2
import numpy as np
from shapely.geometry import Polygon
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import copy

config = dict(
    description='use new A and B, use abs delta_v',
    step_interval=0.1,  # trajectory prediction step interval, second
    traj_pre_ver=None,  # no use
    pred_hori=4,  # prediction horizon
    A=1.04,  # 0.17,  # determine the attenuation curve in temporal dimension
    B=1.94,  # 0.58,  # determine the attenuation curve in temporal dimension
    Alpha=0.5,  # balance coef for delta_v and abs_v, recommende not to change
    stochastic=False,  # no use
    consider_max_dece=False,  # recommende not to change
    distance_method='shapely',  # or 'circle', use manhatton distance instead of Euclidean distance
    damage_form='fix damage loglog',  # or 'kinetic form', the way to calculate damage
    delta_v_method='norm',  # no use
    use_delta_v_vector=True,  # direct use abs(delta v)
    relative_direction_method='both center',  # no use
    velocity_method='signal plus'
)


@dataclass
class Veh_obj:
    """Basic class of a object
    risk will be assessed and saved to each instance of this class
    """
    # basic information
    name: str = None  # vehicle name
    type: str = 'car'  # vehicle type, determine shapes and sensitive
    x0: float = 0.  # position X, unit: m
    y0: float = 0.  # position Y, unit: m
    speed: float = 0.  # speed, unit: m/s
    phi0: float = 0.  # heading angle, to right is zero. unit: rad
    a0: float = 0.  # acceleration, unit: m/s^2
    phi_a0: float = 0.  # yaw rate, nuit: rad/s
    mass: float = -1.  # mass, unit: ton
    sensitive: float = -1.  # how vulnerable the obj is
    length: float = 4.5  # shape L
    width: float = 1.8  # shape W
    fix_damage: float = None  # for version 8
    max_dece: float = 7.5  # maximum deceleration values, unit: m/s^2

    # intermediate variables from trajectory prediction
    x_pred: np.ndarray = np.ndarray(0, )  # predicted position X
    y_pred: np.ndarray = np.ndarray(0, )  # predicted position Y
    v_pred: np.ndarray = np.ndarray(0, )  # predicted speed
    phi_pred: np.ndarray = np.ndarray(0, )  # predicted heading angle
    virtual_center_x: np.ndarray = np.ndarray(
        0, )  # surrogate point, for surr. veh., it is the center of rear bumper; for ego veh., it is the centers of rear/front bumper
    virtual_center_y: np.ndarray = np.ndarray(0, )
    future_position: List[Polygon] = field(
        default_factory=list)  # list of Shapely Polygon obj. to calculate the minimum distance between ego and surr. veh.

    # media variables for safety evaluation
    risk_curve: np.ndarray = np.ndarray(0, )
    dis_t: np.ndarray = np.ndarray(0, )  # 各时刻的距离
    time_t: np.ndarray = np.ndarray(0, )  # 各个时刻的时间折减原始值
    damage: np.ndarray = np.ndarray(0, )  # 各时刻预测损伤
    dis_de_curve: np.ndarray = np.ndarray(0, )  # 各时刻距离衰减曲线
    weight_t: np.ndarray = np.ndarray(0, )  # 各时刻时间衰减曲线
    delta_v: np.ndarray = np.ndarray(0, )  # 各时刻速度差
    abs_v: np.ndarray = np.ndarray(0, )  # 各时刻速度和
    v_: np.ndarray = np.ndarray(0, )  # 各个时刻的替代速度

    # results of risk assessment, useful only for ego vehicle
    risk: List[float] = field(default_factory=list)  # final determined risk values, scalar
    collided: List[bool] = field(default_factory=list)  # if will be a collision in predicted horizon
    rc: List[bool] = field(default_factory=list)  # if a collision occurs at current (real collision)

    # results of risk assessment for ego vehicle to each obj. useful only for ego vehicle
    risk2obj: Dict[str, list] = field(default_factory=dict)
    collided2obj: Dict[str, list] = field(default_factory=dict)
    rc2obj: Dict[str, list] = field(default_factory=dict)

    # used for regulatory compliance eva
    prev_laneid: str = ''  # lane id of vehicle on previous step
    prev_roadid: bool = False  # road id of vehicle on previous step
    regulatory_penalty: int = 0  # penalty due to break regulatory

    # used for comfort eva
    acce_recorder: List[float] = field(default_factory=list)
    acce_lat_recorder: List[float] = field(default_factory=list)

    # used for traffic efficiency eva
    surr_speed: List[float] = field(default_factory=list)  # speed list of surr vehicle

    # used for statistical results
    mileage: float = 0.
    st_risk: List[float] = field(default_factory=list)  # risk recorde
    st_coll: int = 0  # num of collision recorde
    st_comf: List[float] = field(default_factory=list)  # comfort recorde
    st_comf_lvl: int = 0  # steps comfort > level 3 recorde
    st_traf: List[float] = field(default_factory=list)  # recorde
    st_ener: float = 0.  # sum
    st_regu: List[float] = field(default_factory=list)  # sum

    def set_ms(self):
        """set default parameters
        """
        if self.length < 1 and self.width < 1:  # ped
            if self.mass == -1.: self.mass = 0.07
            if self.sensitive == -1.: self.sensitive = 50
            if self.fix_damage == None: self.fix_damage = 1.2
        elif self.length < 2.5:  # bic
            if self.mass == -1.: self.mass = 0.09
            if self.sensitive == -1.: self.sensitive = 50
            if self.fix_damage == None: self.fix_damage = 1.2
        elif self.length < 7.5:  # car
            if self.mass == -1.: self.mass = 1.8
            if self.sensitive == -1.: self.sensitive = 1
            if self.fix_damage == None: self.fix_damage = 1
        else:  # truck
            if self.mass == -1.: self.mass = 4.5
            if self.sensitive == -1.: self.sensitive = 1
            if self.fix_damage == None: self.fix_damage = 1.5

    def update(self, **kwargs):
        """update related parameters during calculation
        """
        for key, val in kwargs.items():
            assert key in vars(self), '{} is not a class attr'.format(key)
            exec("self.{0}=val".format(key), {'self': self, 'val': val})

    def update_risk(self, **kwargs):
        """should only be called by ego veh, to save risk ,rc, collided information
        """
        if 'obj_name' in kwargs.keys():
            flag = 1  # if save risk2ego dict
            obj_name = kwargs['obj_name']
            del kwargs['obj_name']
        else:
            flag = 0
        for key, val in kwargs.items():
            exec("self.{0}.append(val)".format(key),
                 {'self': self, 'val': val})  # for ego: record the risks to each obj
            if flag == 1:
                exec("self.{0}[obj_name] = val".format(key + '2obj'), {'self': self, 'obj_name': obj_name, 'val': val})

    def reset(self):
        self.risk.clear()
        self.collided.clear()
        self.rc.clear()
        self.risk2obj.clear()
        self.collided2obj.clear()
        self.rc2obj.clear()
        self.regulatory_penalty = 0
        self.surr_speed.clear()
        self.acce_recorder = self.acce_recorder[-20:]
        self.acce_lat_recorder = self.acce_lat_recorder[-20:]


@dataclass
class PODAR:
    """used to deal risk assessment for ego vehicle
    """
    ego: Dict[str, Veh_obj] = field(default_factory=dict)  # ego (self-controlled) vehicle list
    obj: Dict[str, Veh_obj] = field(default_factory=dict)  # all surrouding vehicle list without ego vehicles
    ego_surr_list: Dict[str, list] = field(default_factory=dict)  # surr vehicle list for specific ego vehicle

    def add_ego(self, name, **kwargs):
        """add ego vehicle information
        """
        _o = Veh_obj(name=name)
        for key, val in kwargs.items():
            assert key in vars(_o), '{} is not a class attr'.format(key)
            exec("_o.{0}=val".format(key), {'_o': _o, 'val': val})
        _o.set_ms()
        self.ego[name] = _o

    def update_ego(self, name, **kwargs):
        self.ego[name].update(**kwargs)
        traj_predition(self.ego[name])
        get_future_position_shapely(self.ego[name], ego_flag=True)

    def add_obj(self, name, **kwargs):
        """add surr. vehicle
        """
        has_dealed_flag = 0
        _o = Veh_obj(name=name)
        for key, val in kwargs.items():
            assert key in vars(_o), '{0} is not a class attr'.format(key)
            exec("_o.{0}=val".format(key), {'_o': _o, 'val': val})
        _o.set_ms()
        for _name, _list in self.ego_surr_list.items():
            if _o.name in _list:
                if not has_dealed_flag:
                    traj_predition(_o)
                    get_future_position_shapely(_o)
                    has_dealed_flag = 1
                get_risk_to_obj(self.ego[_name], _o)
        self.obj[name] = _o

    def reset(self):
        for _ego in self.ego.values(): _ego.reset()
        self.obj.clear()
        self.ego_surr_list.clear()

    def estimate_risk(self, ego_name) -> list:
        """run the risk evaluation
        """
        if len(self.ego[ego_name].risk) == 0:
            self.ego[ego_name].st_risk.append(0)
            self.ego[ego_name].st_coll += 0
            return (0, 0, 0)

        risk = np.max(self.ego[ego_name].risk)  # the max risk value among all vehicles is regarded as final risk
        collided = 1 if np.sum(self.ego[ego_name].collided) > 0 else 0
        rc = 1 if np.sum(self.ego[ego_name].rc) > 0 else 0

        self.ego[ego_name].st_risk.append(copy.deepcopy(risk))
        self.ego[ego_name].st_coll += rc

        return (risk, rc, collided)


def traj_predition(veh: Veh_obj):
    """predict the future position and heading angle of an object
    prediciton horzion is 3 second

    Parameters
    ----------
    veh : Veh_obj
        ego or surr. veh. instance
    """
    step_interval = config['step_interval']
    pred_hori = config['pred_hori']

    x, y, v, phi, a, a_v, L = veh.x0, veh.y0, veh.speed, veh.phi0, veh.a0, veh.phi_a0, veh.length
    veh_w, veh_l = veh.width, veh.length
    x_pre, y_pre, v_pre, phi_pre = [x], [y], [v], [phi]
    veh.pred_steps = int(pred_hori / step_interval) + 1
    for i in range(int(pred_hori / step_interval)):
        v_pre.append(np.clip(v_pre[i] + step_interval * a, 0, None))
        x_pre.append(x_pre[i] + step_interval * v_pre[i] * np.cos(phi_pre[i]))
        y_pre.append(y_pre[i] + step_interval * v_pre[i] * np.sin(phi_pre[i]))
        phi_pre.append(phi_pre[i] + step_interval * v_pre[i] * np.tan(a_v) / L)

    veh.update(x_pred=np.array(x_pre), y_pred=np.array(y_pre), v_pred=np.array(v_pre), phi_pred=np.array(phi_pre))


def get_future_position_shapely(veh: Veh_obj, ego_flag=False):
    """get Shapely instance to calculate relative distance

    Parameters
    ----------
    veh : Veh_obj
    ego_flag : bool, optional
        to determine if the veh is ego vehicle, due to the virtual_center are not the same, by default False
    """
    traj_x_true, traj_y_true, traj_heading_true, veh_w, veh_l = \
        veh.x_pred, veh.y_pred, veh.phi_pred, veh.width, veh.length
    assert len(traj_x_true) > 0, 'there is no predicted traj'
    if config['distance_method'] == 'shapely':
        shapely_results = []        #
        beta = atan2(veh_w / 2, veh_l / 2)  # vehicle center-four point angle
        r = sqrt(pow(veh_w, 2) + pow(veh_l, 2)) / 2  # rotation radius

        x_c1 = traj_x_true + r * np.cos(beta + traj_heading_true)  # top-left
        y_c1 = traj_y_true + r * np.sin(beta + traj_heading_true)
        x_c2 = traj_x_true + r * np.cos(beta - traj_heading_true)  # top-right
        y_c2 = traj_y_true - r * np.sin(beta - traj_heading_true)
        x_c5 = traj_x_true - r * np.cos(beta - traj_heading_true)  # bottom-left
        y_c5 = traj_y_true + r * np.sin(beta - traj_heading_true)
        x_c6 = traj_x_true - r * np.cos(beta + traj_heading_true)  # bottom-right
        y_c6 = traj_y_true - r * np.sin(beta + traj_heading_true)

        for i in range(len(traj_x_true)):
            shapely_results.append(Polygon(((x_c1[i], y_c1[i]),
                                            (x_c2[i], y_c2[i]),
                                            (x_c6[i], y_c6[i]),
                                            (x_c5[i], y_c5[i]))))

    elif config['distance_method'] == 'circle':
        shapely_results = []
        n_circle = int(veh.length // (veh.width)) + 1 if veh.length > veh.width else 1
        diss = np.max((0, (veh.length - veh.width))) / np.max((1, (n_circle - 1)))

        for i in range(n_circle):
            diff_x, diff_y = rotation(-diss * i + np.max((0, (veh.length - veh.width))) / 2, 0, veh.phi_pred)
            shapely_results.append([veh.x_pred - diff_x, veh.y_pred - diff_y])
    else:
        raise KeyError

    if config['relative_direction_method'] == 'both center' or not config['use_delta_v_vector']:
        virtual_center_x, virtual_center_y = traj_x_true, traj_y_true
    elif ego_flag or config['relative_direction_method'] == 'obj two point':
        virtual_center_x = [traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * 1,
                            traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * -1]
        virtual_center_y = [traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * 1,
                            traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * -1]
    elif config['relative_direction_method'] == 'obj one point':
        virtual_center_x = traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * -1
        virtual_center_y = traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * -1
    else:
        raise KeyError('relative_direction_method cannot find')

    veh.update(future_position=shapely_results, virtual_center_x=virtual_center_x, virtual_center_y=virtual_center_y)


def get_risk_to_obj(ego: Veh_obj, obj: Veh_obj):
    """risk assessment for ego vehicle and one surr. veh.

    Parameters
    ----------
    ego : Veh_obj
    obj : Veh_obj
    """
    step_interval = config['step_interval']
    A = config['A']
    B = config['B']
    Alpha = config['Alpha']
    pred_hori = config['pred_hori']
    consider_max_dece = config['consider_max_dece']

    t_step = int(pred_hori / step_interval)
    dis_t = []
    # assert len(obj.future_position) > 0, 'Should get future position first'
    if config['distance_method'] == 'shapely':
        for i in range(t_step + 1):  # get the distances in predicted horizon  # 1ms
            dis_t.append(ego.future_position[i].distance(obj.future_position[i]))
    elif config['distance_method'] == 'circle':
        for i in range(t_step + 1):
            rec = (0, 0)
            d_min = 9999
            for _n in range(len(ego.future_position)):
                for _m in range(len(obj.future_position)):
                    d_man = np.abs(ego.future_position[_n][0][i] - obj.future_position[_m][0][i]) + np.abs(
                        ego.future_position[_n][1][i] - obj.future_position[_m][1][i])
                    if d_man < d_min:
                        d_min = d_man
                        rec = (_n, _m)
            dis_t.append(np.clip(np.sqrt(
                (ego.future_position[rec[0]][0][i] - obj.future_position[rec[1]][0][i]) ** 2 + (
                            ego.future_position[rec[0]][1][i] - obj.future_position[rec[1]][1][i]) ** 2) - (
                                             ego.width + obj.width) / 2, 0, None))
    else:
        raise KeyError
    dis_t = np.array(dis_t)

    if config['use_delta_v_vector']:
        if obj.type != 'markings':
            vx0, vx1 = ego.v_pred * np.cos(ego.phi_pred), obj.v_pred * np.cos(obj.phi_pred)
            vy0, vy1 = ego.v_pred * np.sin(ego.phi_pred), obj.v_pred * np.sin(obj.phi_pred)
            vec_v_x = vx1 - vx0
            vec_v_y = vy1 - vy0
            if config['relative_direction_method'] == 'both center':
                vec_dir_x = ego.virtual_center_x - obj.virtual_center_x
                vec_dir_y = ego.virtual_center_y - obj.virtual_center_y
                if config['delta_v_method'] == 'norm':
                    modd = np.linalg.norm([vec_dir_x, vec_dir_y], axis=0) + 0.00001
                elif config['delta_v_method'] == 'manhattan_norm':
                    modd = np.abs(vec_dir_x) + np.abs(vec_dir_y)
                else:
                    raise KeyError('delta_v_method form cannot find.')
                vec_dir_x, vec_dir_y = vec_dir_x / modd, vec_dir_y / modd
                delta_v = vec_v_x * vec_dir_x + vec_v_y * vec_dir_y
            elif config['relative_direction_method'] == 'obj one point':
                # ego vehicle use the front and rear points and other vehicle use the rear point
                vec_dir_x_f = ego.virtual_center_x[0] - obj.virtual_center_x
                vec_dir_y_f = ego.virtual_center_y[0] - obj.virtual_center_y
                vec_dir_x_r = ego.virtual_center_x[1] - obj.virtual_center_x
                vec_dir_y_r = ego.virtual_center_y[1] - obj.virtual_center_y
                if config['delta_v_method'] == 'norm':
                    modd_f = np.linalg.norm([vec_dir_x_f, vec_dir_y_f], axis=0) + 0.00001
                    modd_r = np.linalg.norm([vec_dir_x_r, vec_dir_y_r], axis=0) + 0.00001
                elif config['delta_v_method'] == 'manhattan_norm':
                    modd_f = np.abs(vec_dir_x_f) + np.abs(vec_dir_y_f)
                    modd_r = np.abs(vec_dir_x_r) + np.abs(vec_dir_y_r)
                else:
                    raise KeyError('delta_v_method form cannot find.')
                vec_dir_x_f, vec_dir_y_f = vec_dir_x_f / modd_f, vec_dir_y_f / modd_f
                vec_dir_x_r, vec_dir_y_r = vec_dir_x_r / modd_r, vec_dir_y_r / modd_r

                delta_v_f = vec_v_x * vec_dir_x_f + vec_v_y * vec_dir_y_f
                delta_v_r = vec_v_x * vec_dir_x_r + vec_v_y * vec_dir_y_r

                delta_v = np.max([delta_v_f, delta_v_r], axis=0)
            elif config['relative_direction_method'] == 'obj two point':
                vec_dir_x_ff = ego.virtual_center_x[0] - obj.virtual_center_x[0]
                vec_dir_y_ff = ego.virtual_center_y[0] - obj.virtual_center_y[0]
                vec_dir_x_fr = ego.virtual_center_x[1] - obj.virtual_center_x[0]
                vec_dir_y_fr = ego.virtual_center_y[1] - obj.virtual_center_y[0]
                vec_dir_x_rf = ego.virtual_center_x[0] - obj.virtual_center_x[1]
                vec_dir_y_rf = ego.virtual_center_y[0] - obj.virtual_center_y[1]
                vec_dir_x_rr = ego.virtual_center_x[1] - obj.virtual_center_x[1]
                vec_dir_y_rr = ego.virtual_center_y[1] - obj.virtual_center_y[1]
                if config['delta_v_method'] == 'norm':
                    modd_ff = np.linalg.norm([vec_dir_x_ff, vec_dir_y_ff], axis=0) + 0.00001
                    modd_fr = np.linalg.norm([vec_dir_x_fr, vec_dir_y_fr], axis=0) + 0.00001
                    modd_rf = np.linalg.norm([vec_dir_x_rf, vec_dir_y_rf], axis=0) + 0.00001
                    modd_rr = np.linalg.norm([vec_dir_x_rr, vec_dir_y_rr], axis=0) + 0.00001
                elif config['delta_v_method'] == 'manhattan_norm':
                    modd_ff = np.abs(vec_dir_x_ff) + np.abs(vec_dir_y_ff)
                    modd_fr = np.abs(vec_dir_x_fr) + np.abs(vec_dir_y_fr)
                    modd_rf = np.abs(vec_dir_x_rf) + np.abs(vec_dir_y_rf)
                    modd_rr = np.abs(vec_dir_x_rr) + np.abs(vec_dir_y_rr)
                else:
                    raise KeyError('delta_v_method form cannot find.')
                vec_dir_x_ff, vec_dir_y_ff = vec_dir_x_ff / modd_ff, vec_dir_y_ff / modd_ff
                vec_dir_x_fr, vec_dir_y_fr = vec_dir_x_fr / modd_fr, vec_dir_y_fr / modd_fr
                vec_dir_x_rf, vec_dir_y_rf = vec_dir_x_rf / modd_rf, vec_dir_y_rf / modd_rf
                vec_dir_x_rr, vec_dir_y_rr = vec_dir_x_rr / modd_rr, vec_dir_y_rr / modd_rr

                delta_v_ff = vec_v_x * vec_dir_x_ff + vec_v_y * vec_dir_y_ff
                delta_v_fr = vec_v_x * vec_dir_x_fr + vec_v_y * vec_dir_y_fr
                delta_v_rf = vec_v_x * vec_dir_x_rf + vec_v_y * vec_dir_y_rf
                delta_v_rr = vec_v_x * vec_dir_x_rr + vec_v_y * vec_dir_y_rr

                delta_v = np.max([delta_v_ff, delta_v_fr, delta_v_rf, delta_v_rr], axis=0)
                ...
            else:
                raise KeyError('relative_direction_method cannot find.')
        else:
            delta_v = ego.v_pred
    else:
        delta_v = np.abs(ego.v_pred - obj.v_pred)

    abs_v = ego.v_pred + obj.v_pred  # speed amplitude

    if config['velocity_method'] == 'plus':
        v_ = delta_v * Alpha + abs_v * (1 - Alpha)
    elif config['velocity_method'] == 'signal plus':
        assert config['use_delta_v_vector'], 'velocity_method: signal plus must use with use_delta_v_vector'
        dv_flag = np.zeros(t_step + 1) + 1
        dv_flag[(delta_v < 0) & (dis_t > 0)] = -0.5
        v_ = np.abs(ego.v_pred - obj.v_pred) * dv_flag * Alpha + abs_v * (1 - Alpha)
    else:
        raise KeyError('velocity_method cannot find.')

    if config['damage_form'] == 'kinetic energy':
        damage = 0.5 * (ego.mass * ego.sensitive + obj.mass * obj.sensitive) * v_ * np.abs(v_) * 1 / 50
    elif config['damage_form'] == 'fix damage':
        v_flag = np.zeros(t_step + 1) + 1
        v_flag[v_ < 0] = -1
        p_flag = np.zeros(t_step + 1) + 1
        p_flag[v_ < 0] = 0  # 负值的速度，不考虑质量影响
        damage = ((ego.fix_damage + obj.fix_damage) * p_flag + 1) * np.log((v_ + 1) ** 2) * v_flag
    elif config['damage_form'] == 'fix damage loglog':
        v_flag = np.zeros(t_step + 1) + 1
        v_flag[v_ < 0] = -1
        p_flag = np.zeros(t_step + 1) + 1
        p_flag[v_ < 0] = 0  # 负值的速度，不考虑质量影响
        damage = np.log(np.log(((ego.fix_damage + obj.fix_damage) * p_flag + 1) * 0.5 * (v_ + 1) ** 2) + 2)
        # damage = np.log(np.log((ego.fix_damage + obj.fix_damage + 1) * 0.5 * (v_ + 1)**2) + 2)
    else:
        raise KeyError('damage form cannot find.')

    time_t = np.linspace(0, pred_hori, t_step + 1)
    if consider_max_dece:
        ii = int(ego.v_pred[0] / abs(ego.max_dece) / step_interval)
        time_t = np.concatenate((np.zeros(ii), time_t[:t_step - ii + 1]))

    weight_t = np.exp(-1 * time_t * A)

    dis_t[dis_t < 0] = 0
    dis_de_curve = np.exp(-1 * dis_t * B)

    time_t = np.linspace(0, pred_hori, t_step + 1)
    if consider_max_dece:
        ii = int(ego.v_pred[0] / abs(ego.max_dece) / step_interval)
        time_t = np.concatenate((np.zeros(ii), time_t[:t_step - ii + 1]))

    weight_t = np.exp(-1 * time_t * A)

    dis_t[dis_t < 0] = 0
    dis_de_curve = np.exp(-1 * dis_t * B)
    # ============================================================================
    risk = damage * (dis_de_curve * weight_t)

    if np.sum(np.where(risk >= 0)) > 0:  # if the estimated damage values exist at least one positive
        risk_tmp = np.max(risk)  # use the max value
    else:  # if all the estimated damage are negative, meanning the obj is moving far away from host vehicle
        risk = damage * (1 + 1 - (dis_de_curve * weight_t))  # deal with the risk values
        risk_tmp = np.max(risk)  # modified, 20220104

    if min(dis_t) <= 0:  # if there exist a collision in predicted horizon
        if np.min(np.where(dis_t == 0)) != 0:  # if no collision occurs at present
            ego.update_risk(risk=risk_tmp, collided=1, rc=0, obj_name=obj.name)  # predicted collision
        else:
            ego.update_risk(risk=risk_tmp, collided=1, rc=1, obj_name=obj.name)  # actual collision
    else:
        ego.update_risk(risk=risk_tmp, collided=0, rc=0, obj_name=obj.name)  # no collision

    obj.update(risk_curve=risk, damage=damage, dis_de_curve=dis_de_curve, weight_t=weight_t, delta_v=delta_v,
               abs_v=abs_v, dis_t=dis_t, time_t=time_t, v_=v_)


def rotation(l, w, phi):
    """phi: rad"""
    diff_x = l * np.cos(phi) - w * np.sin(phi)
    diff_y = l * np.sin(phi) + w * np.cos(phi)
    return (diff_x, diff_y)


def podar_render(frame: PODAR, ego_name: str = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import colors
    # define colors
    cmap = plt.cm.jet
    mycmap = cmap.from_list('Custom cmap',
                            [[0 / 255, 255 / 255, 0 / 255], [255 / 255, 255 / 255, 0 / 255],
                             [255 / 255, 0 / 255, 0 / 255]], cmap.N)
    c_norm = colors.Normalize(vmin=0, vmax=10, clip=True)

    def _draw_rotate_rec(veh, ec, fc: str = 'white'):
        diff_x, diff_y = rotation(-veh.length / 2, -veh.width / 2, veh.phi0)
        rec = patches.Rectangle((veh.x0 + diff_x, veh.y0 + diff_y), veh.length, veh.width,
                                angle=veh.phi0 / np.pi * 180, ec=ec, fc=fc)
        return rec

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()

    x_min, x_max, y_min, y_max = -10000, 10000, -10000, 10000
    ego_name = list(frame.ego.keys())[0] if ego_name == None else ego_name
    for _name, _veh in {**frame.ego, **frame.obj}.items():
        rec_handle = _draw_rotate_rec(_veh, 'black' if _name != ego_name else 'red',
                                      'white' if _name == ego_name else mycmap(
                                          c_norm(frame.ego[ego_name].risk2obj[_name])))
        ax.add_patch(rec_handle)
        ax.text(_veh.x0, _veh.y0, 'id={}, r={:.2f}, v={:.1f}'.format(_name,
                                                                     (frame.ego[ego_name].risk2obj[
                                                                          _name] if _name != ego_name else np.max(
                                                                         frame.ego[ego_name].risk)),
                                                                     _veh.speed))
        ax.scatter(_veh.x0, _veh.y0, c='black', s=5)
        ax.plot(_veh.x_pred, _veh.y_pred, linestyle='--')
        x_min, x_max, y_min, y_max = np.min([x_min, _veh.x0]), np.max([x_max, _veh.x0]), np.min(
            [y_min, _veh.y0]), np.max([y_max, _veh.y0])
        print('id={:<4} x={:<10.2f} y={:<10.2f} v={:<6.2f} phi={:<6.2f} risk={:<8.5f} l={:<6.1f} w={:<6.1f}'.format(
            _veh.name, _veh.x0, _veh.y0, _veh.speed, _veh.phi0 / np.pi * 180,
            frame.ego[ego_name].risk2obj[_name] if _name != ego_name else np.max(frame.ego[ego_name].risk),
            _veh.length, _veh.width))

    plt.xlim(x_min - 10, x_max + 10)
    plt.ylim(y_min - 10, y_max + 10)
    plt.axis('equal')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    podar = PODAR()
    podar.ego_surr_list = {'0': ['1', '2', '3']}
    podar.add_ego(type='car', name='0')
    podar.update_ego(name='0', x0=0, y0=0, speed=15, phi0=np.pi / 2)
    podar.add_obj(type='car', name='1', x0=-2.8, y0=0, speed=5, phi0=np.pi / 2)
    podar.add_obj(type='car', name='2', x0=3.0, y0=4, speed=8, phi0=np.pi / 2)
    podar.add_obj(type='car', name='3', x0=0.0, y0=0, speed=10, phi0=np.pi / 2)

    podar.estimate_risk('0')
    print(podar.estimate_risk('0'))

    for v in podar.obj.values():
        print("name={}, risk={:.3f}, minDis={:.3f}".format(v.name, podar.ego['0'].risk2obj[v.name], np.min(v.dis_t), ))

    podar_render(podar)
    plt.show()