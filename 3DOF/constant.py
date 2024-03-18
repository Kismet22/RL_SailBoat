###################################################################################################
# Standard import
from math import pi, sin, cos, atan, atan2, sqrt, copysign, degrees, radians, modf
from collections import namedtuple


###################################################################################################
# Library import
import yaml


###################################################################################################
# Load simulation parameters
# sim_params = open('sim_params_config.yaml')
# param_dict = yaml.load(sim_params)
with open('sim_params_config.yaml', 'r') as stream:
    param_dict = yaml.load(stream, Loader=yaml.FullLoader)

# 船体信息
# Boat
BOAT_LENGTH = param_dict['boat']['length']  # 船长，未使用
MASS = param_dict['boat']['mass']  # 船的质量
HULL_SPEED = param_dict['boat']['hull_speed']  # 船体指在水里的最大速度（与船体长度相关），用来计算水的阻抗
LATERAL_AREA = param_dict['boat']['lateral_area']  # 船侧面积，算波浪的时候用的
WATERLINE_AREA = param_dict['boat']['waterline_area']  # 船入水后水线包围的面积，算静浮力用的
HEIGHT_BOUYANCY = param_dict['boat']['height_bouyancy']  # 浮心在重心以上多高

# MOI : moment of inertia
GEOMETRICAL_MOI_X = param_dict['boat']['geometrical_moi_x']  # 绕x轴的几何转动惯量
GEOMETRICAL_MOI_Y = param_dict['boat']['geometrical_moi_y']  # 绕y轴的几何转动惯量
MOI_X = param_dict['boat']['moi_x']  # 船绕轴的真正转动惯量
MOI_Y = param_dict['boat']['moi_y']
MOI_Z = param_dict['boat']['moi_z']

# COG : center of gravity
DISTANCE_COG_RUDDER = param_dict['boat']['distance_cog_rudder']  # 舵受力点离船重心水平多远(sqrt(x^2+y^2))
DISTANCE_COG_SAIL_PRESSURE_POINT = param_dict['boat']['distance_cog_sail_pressure_point']  # 帆受力点离船重心水平多远(sqrt(x^2+y^2))
DISTANCE_COG_KEEL_PRESSURE_POINT = param_dict['boat']['distance_cog_keel_pressure_point']  # 龙骨受力点离船重心水平多远(sqrt(
# x^2+y^2))
DISTANCE_MAST_SAIL_PRESSURE_POINT = param_dict['boat']['distance_mast_sail_pressure_point']

# Sail
SAIL_LENGTH = param_dict['boat']['sail']['length']  # [m]  用来算帆的摩擦阻力
SAIL_HEIGHT = param_dict['boat']['sail']['height']  # [m]  未使用
SAIL_AREA = param_dict['boat']['sail']['area']  # [m^2]  计算帆受力用
SAIL_STRETCHING = param_dict['boat']['sail']['stretching']  # 计算派生阻力时用到
SAIL_PRESSURE_POINT_HEIGHT = param_dict['boat']['sail']['pressure_point_height']  # [m]
# 帆受力点高度（z，相对重心），将帆上的受力集中在一个点上，方便计算

# Keel
KEEL_LENGTH = param_dict['boat']['keel']['length']  # 龙骨长
KEEL_HEIGHT = param_dict['boat']['keel']['height']  # 龙骨高
KEEL_STRETCHING = param_dict['boat']['keel']['stretching']  # 计算水的派生阻力时候用到了

# Rudder
RUDDER_BLADE_AREA = param_dict['boat']['rudder']['area']  # 面积
RUDDER_STRETCHING = param_dict['boat']['rudder']['stretching']  # 展弦比

# Damping
# 阻尼
ALONG_DAMPING = param_dict['boat']['along_damping']  # 纵向阻尼,前后水平晃动
TRANSVERSE_DAMPING = param_dict['boat']['transverse_damping']  # 横向阻尼
DAMPING_Z = param_dict['boat']['damping_z']  # 竖直阻尼
# 由三个角度产生的阻尼
YAW_TIMECONSTANT = param_dict['boat']['yaw_timeconstant']
PITCH_DAMPING = param_dict['boat']['pitch_damping']
ROLL_DAMPING = param_dict['boat']['roll_damping']

# Physical constants
WATER_DENSITY = param_dict['environment']['water_density']
WATER_VISCOSITY = param_dict['environment']['water_viscosity']
AIR_VISCOSITY = param_dict['environment']['air_viscosity']
AIR_DENSITY = param_dict['environment']['air_density']
GRAVITY = param_dict['environment']['gravity']

###################################################################################################
# Invariants
# Wave impedance
# 波浪阻抗。不变
WAVE_IMPEDANCE_INVARIANT = (WATER_DENSITY / 2) * LATERAL_AREA  # (水密度/2) * 船侧面积

# Hydrostatic force
# 静水压力
HYDROSTATIC_EFF_X = HEIGHT_BOUYANCY + (WATER_DENSITY / MASS) * GEOMETRICAL_MOI_X
HYDROSTATIC_EFF_Y = HEIGHT_BOUYANCY + (WATER_DENSITY / MASS) * GEOMETRICAL_MOI_Y
# 浮心相对重心高度 + (水密度/ 质量) * 转动惯量

HYDROSTATIC_INVARIANT_Z = - WATER_DENSITY * WATERLINE_AREA * GRAVITY
# z轴静水压力，不变

GRAVITY_FORCE = MASS * GRAVITY
# 重力

# Damping
# 阻尼 invariant(不改变的)
DAMPING_INVARIANT_X = -MASS / ALONG_DAMPING  # x方向阻尼
DAMPING_INVARIANT_Y = -MASS / TRANSVERSE_DAMPING  # y方向阻尼
DAMPING_INVARIANT_Z = -.5 * DAMPING_Z * sqrt(WATER_DENSITY * WATERLINE_AREA * GRAVITY * MASS)  # z方向阻尼
DAMPING_INVARIANT_YAW = -(MOI_Z / YAW_TIMECONSTANT)  # 前进方向夹角
DAMPING_INVARIANT_PITCH = -2 * PITCH_DAMPING * sqrt(MOI_Y * MASS * GRAVITY * HYDROSTATIC_EFF_Y)
DAMPING_INVARIANT_ROLL = -2 * ROLL_DAMPING * sqrt(MOI_X * MASS * GRAVITY * HYDROSTATIC_EFF_X)

###################################################################################################
# Structured data
# Environment
TrueWind = namedtuple('TrueWind', 'x, y, strength, direction')  # 真实风向
ApparentWind = namedtuple('ApparentWind', 'x, y, angle, speed')  # 视风
Wave = namedtuple('Wave', 'length, direction, amplitude')  # 波浪
WaveVector = namedtuple('WaveVector', 'x, y')
WaveInfluence = namedtuple('WaveInfluence', 'height, gradient_x, gradient_y')
# Forces
RudderForce = namedtuple('RudderForce', 'x, y')
LateralForce = namedtuple('LateralForce', 'x, y')  # 横向力
SailForce = namedtuple('SailForce', 'x, y')
HydrostaticForce = namedtuple('HydrostaticForce', 'x, y, z')  # 静水压力
Damping = namedtuple('Damping', 'x, y, yaw')

###################################################################################################
# State description
# 全局位置x,y
# 艏向角(YAW)
# 速度该变量vx,vy(以船体为坐标系)
# 艏向角变化率r
POS_X, POS_Y, \
YAW, \
VEL_X, VEL_Y, \
YAW_RATE = range(6)

###################################################################################################
# Environment index description
# 创建环境信息枚举
SAIL_ANGLE, RUDDER_ANGLE, TRUE_WIND, WAVE = range(4)