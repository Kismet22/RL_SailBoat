# 运行，包括风帆控制和舵向控制
from heading_controller import heading_controller
from simulation import *
from numpy import *
import matplotlib.pyplot as plt
from sail_angle import sail_angle
# from time import clock
from time import process_time

deq = solve


# 常微分方程类声明
class simple_integrator(object):

    def __init__(self, deq):
        self.t = 0
        self.deq = deq
        self.max_sample_time = .01
        pass

    def set_initial_value(self, x0, t0):
        self.y = x0
        self.t = t0

    def integrate(self, end_time):
        while end_time - self.t > 1.1 * self.max_sample_time:
            self.integrate(self.t + self.max_sample_time)

        self.y += self.deq(self.t, self.y) * (end_time - self.t)
        self.t = end_time


# 常微分方程修改
def deq_sparse(time, state):
    # ignore pitch and heave oscillations (not really needed)
    # 修改常微分方程
    diff = deq(time, state)
    diff[VEL_Z] = 0
    diff[PITCH_RATE] = 0
    diff[ROLL_RATE] = 0
    return diff


# 状态初始化
def init_data_arrays(n_states, N_steps, x0):
    x = zeros((n_states, N_steps + 1))  # n_states行，N_steps + 1列
    r = zeros(N_steps + 1)
    sail = zeros(N_steps + 1)
    t = zeros(N_steps + 1)
    # x0为12个state的初始状态
    x[:, 0] = x0  # 第一列

    ref_heading = zeros(N_steps + 1)
    # 返回状态、时间、奖励、帆状态和前进角度
    return x, t, r, sail, ref_heading


# 微分方程初始化
def init_integrator(x0, sampletime, sparse=False):
    # init integrator
    # 是否稀疏化处理
    fun = deq if not sparse else deq_sparse
    # 选择模型方法
    integrator = ode(fun).set_integrator('dopri5')
    # 初始化状态输入
    integrator.set_initial_value(x0, 0)
    # init heading controller
    controller = heading_controller(sampletime, max_rudder_angle=15 * pi / 180)
    controller.calculate_controller_params(YAW_TIMECONSTANT)

    return integrator, controller


# 平滑处理
def smooth_reference(ref_heading, n_filter=5):
    ################ potential smoothing of heading reference
    # 拷贝前进角度
    smoothed = ref_heading.copy()
    N = ref_heading.shape[0]
    for i in range(N):
        # 改为整数除法，避免报错
        ind_low = max(0, i - n_filter // 2)
        ind_high = min(N, i + (n_filter - n_filter // 2))
        smoothed[i] = np.mean(ref_heading[ind_low:ind_high])

    return smoothed


def scenario_1(save=False):
    n_states = 12
    # 在哪里声明??
    if actor_dynamics:
        n_states = 14
    ## environment
    wind = TrueWind(0, 5, 5, pi / 2)
    environment[SAIL_ANGLE] = 0. / 180 * pi
    environment[RUDDER_ANGLE] = 0
    environment[WAVE] = Wave(50, 0, 0)
    sim_wave = False
    if sim_wave:
        environment[WAVE] = Wave(length=100., amplitude=.5, direction=0)
        append = "_wave"
    else:

        append = ""
    environment[TRUE_WIND] = wind

    # simulation params
    t_end = 150.  # 模拟终止时间
    # t_end = 57.
    sampletime = .3  # 采样间隔
    sail_sampletime = 2.
    N_steps = int(t_end / sampletime)  # 步长
    # initial values
    x0 = zeros(n_states)
    x0[VEL_X] = 0.

    if actor_dynamics:
        x0[SAIL_STATE] = 48 * pi / 180

    # 状态初始化
    x, t, r, sail, ref_heading = init_data_arrays(n_states, N_steps, x0)
    print('初始化完毕')

    if True:
        x2, t2, r2, sail2, ref_heading = init_data_arrays(n_states, N_steps, x0)

    integrator, controller = init_integrator(x0, sampletime)
    integrator2 = simple_integrator(deq_sparse)
    integrator2.set_initial_value(x0, 0)

    # 在中间改变两次前进方向
    ref_heading[int(40 / sampletime):] = 1.2 * pi
    ref_heading[int(90 / sampletime):] = .35 * pi

    sail_angle = None

    x, t, separation, keel, sail_force, sail, r = simulate(N_steps, x, t, r, sail, environment, controller,
                                                           integrator, sampletime, sail_sampletime, ref_heading, wind,
                                                           sail_angle)


def simulate(N_steps, x, t, r, sail, environment, controller, integrator, sampletime, sail_sampletime, ref_heading,
             wind=None, predefined_sail_angle=None):
    start_time = process_time()
    for i in range(N_steps):
        # rudder_angle calculation
        print(f"time: {i}")
        print('state', x[:, i])
        print('heading', ref_heading[i])
        # 速度
        speed = sqrt(x[VEL_X, i] ** 2 + x[VEL_Y, i] ** 2)
        # 角度
        drift = arctan2(x[VEL_Y, i], x[VEL_X, i])
        environment[RUDDER_ANGLE] = controller.controll(ref_heading[i], x[YAW, i], x[YAW_RATE, i], speed, x[ROLL, i],
                                                        drift_angle=drift)
        # 船体舵角(action1)
        r[i] = environment[RUDDER_ANGLE]
        print('heading', x[YAW, i])
        print('rudder_angle', r[i])

        if not predefined_sail_angle is None:
            sail[i] = predefined_sail_angle[i]
            environment[SAIL_ANGLE] = sail[i]  # 目标角度
        else:
            if not i % int(sail_sampletime / sampletime):
                apparent_wind = calculate_apparent_wind(x[YAW, i], x[VEL_X, i], x[VEL_Y, i], wind)
                environment[SAIL_ANGLE] = sail_angle(apparent_wind.angle, apparent_wind.speed, SAIL_STRETCHING)

                sail[i] = environment[SAIL_ANGLE]

            else:
                sail[i] = sail[i - 1]
        # 船体帆角(action2)
        print('sail_angle:', sail[i])
        print('integrator.t', integrator.t)
        integrator.integrate(integrator.t + sampletime)
        t[i + 1] = integrator.t
        x[:, i + 1] = integrator.y
        print('integrator.y after integration:', integrator.y)
    end_time = process_time()
    print('computation time', end_time - start_time)

    # forces
    sail_force = np.zeros((2, x.shape[1]))
    keel = np.zeros((2, x.shape[1]))
    separation = np.zeros((x.shape[1]))
    for i in range(x.shape[1]):
        apparent_wind = calculate_apparent_wind(x[YAW, i], x[VEL_X, i], x[VEL_Y, i], wind)
        force = calculate_sail_force(x[ROLL, i], apparent_wind, np.sign(apparent_wind.angle) * sail[i])
        sail_force[:, i] = [force.x, force.y]
        speed = np.sqrt(x[VEL_X, i] ** 2 + x[VEL_Y, i] ** 2)
        lateral_force, lateral_separation = calculate_lateral_force(x[VEL_X, i], x[VEL_Y, i], x[YAW, i], speed)
        keel[:, i] = [lateral_force.x, lateral_force.y]

        separation[i] = lateral_separation

    return x, t, separation, keel, sail_force, sail, r


def main():
    save = True
    scenario_1(save=save)


if __name__ == "__main__":
    main()
