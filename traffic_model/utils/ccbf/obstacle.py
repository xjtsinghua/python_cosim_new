import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
# from dynamics import GeqiangNonLinearBicycleModel

class Obstacle():
    def __init__(self, obs_state, size:float=20.0, var_x:float=0.2, var_y:float=0.5, DT=0.05):
        self.x1_obs = obs_state[0] # x
        self.x2_obs = obs_state[1] # y

        self.state = obs_state
        # self.dynamic = GeqiangNonLinearBicycleModel(x=self.state[0], y=self.state[1], yaw=self.state[2], vx=self.state[3], dt=DT)
        self.throttle = 0 # acc, steer
        self.steer = 0

        self.C = size
        self.var_x = var_x
        self.var_y = var_y

    # Calculate the barrier function of obstacle. (real_num, symbolic)
    def generate_obstacle_barrier(self, x1_ego, x2_ego, obstacle, var_x:float=0.2, var_y:float=0.5):
        # var_x, var_y : How big the influence is in x and y direction.
        return (obstacle.C/np.sqrt(2*np.pi))*np.exp(-( ((x1_ego-obstacle.x1_obs)**2)/(2*obstacle.var_x) + ((x2_ego-obstacle.x2_obs)**2)/(2*obstacle.var_y)))

    def update_input(self, throttle=0, steer=0):
        self.throttle = throttle
        self.steer = steer

    # update the movement of the obstacle
    # def step(self):
    #     self.dynamic.update(self.throttle, self.steer)
    #     self.x1_obs = self.dynamic.x
    #     self.x2_obs = self.dynamic.y
    #     self.state = [self.dynamic.x, self.dynamic.y, self.dynamic.yaw, self.dynamic.vx, self.dynamic.vy, self.dynamic.omega]



if __name__ == '__main__':

    # visualize
    obs_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 5 meter safe zone setting
    obstacle = Obstacle(obs_state, size=100, var_x=1.0, var_y = 1.0)

    x1_min = -10.0
    x1_max = 10.0
    x2_min = -10.0
    x2_max = 10.0

    x1, x2 = np.meshgrid(np.arange(x1_min,x1_max, 0.1), np.arange(x2_min,x2_max, 0.1))

    y = obstacle.generate_obstacle_barrier(x1, x2, obstacle)
    plt.imshow(y,extent=[x1_min,x1_max,x2_min,x2_max], cmap=cm.jet, origin='lower')

    plt.colorbar()
    plt.title("How to evaluate a 2D function using a python grid?" , fontsize=8)
    plt.savefig("evaluate_2d_function_using_meshgrid_03.png", bbox_inches='tight')
    plt.show()

    # plot 3d map
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x1, x2, y, cmap = plt.cm.cividis)

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()