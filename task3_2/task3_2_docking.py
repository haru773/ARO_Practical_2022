import subprocess, math, time, sys, os, numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.2

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.2, 90, -22.8, (-0.12, -0.01, 0.99))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": False,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    global taleId, cubeId, targetId, obstacle
    finalTargetPos = np.array([0.35,0.38,1.0])
    # compile target urdf
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_2_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)

    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition          = [0.8, 0, 0],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/2]),                                  
        useFixedBase          = True,             
        globalScaling         = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_dumb_bell.urdf", 
        basePosition          = [0.5, 0, 1.1],            
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,0]),                                  
        useFixedBase          = False,             
        globalScaling         = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1,1,0,1])
    
    targetId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
        basePosition          = finalTargetPos,             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )
    obstacle = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_obstacle.urdf",
        basePosition          = [0.43,0.275,0.9],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )
   # traj = sim.inverseKinematics('LARM_JOINT5', [0.35,0.275, 1.1], orientation=None,interpolationSteps = 300, threshold=None,)
    #xreal_prev = [0] * len(traj[0])
   # for _ in range(300):
        #sim.chestMovement = sim.chestMovement + traj[_][0]
        #sim.arm0Movement = sim.arm0Movement + traj[_][1]
    #    for i in range(20):
     #       xreal_prev = sim.tick(traj[_], 'LARM_JOINT5', xreal_prev)
    #    time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution():
    endEffector1 = 'LARM_JOINT5'
    endEffector2 = 'RARM_JOINT5'

    sim.move_without_PD(endEffector1, [0.45, 0, 1.1], speed=0.1, orientation=None,
                        threshold=1e-3, maxIter=10, debug=False, verbose=False)
    sim.orientationAdjust(endEffector2)

    sim.move_without_PD(endEffector2, [0.6, 0, 1.05], speed=.2, orientation=None,
                        threshold=1e-3, maxIter=10, debug=False, verbose=False)
    sim.move_without_PD(endEffector1, [0.28, .45, 1.1], speed=0.2, orientation=None,
        threshold=1e-3, maxIter=10, debug=False, verbose=False)
    sim.move_without_PD(endEffector2, [0.35, .281, 1.1], speed=0.12, orientation=None,
                        threshold=1e-3, maxIter=10, debug=False, verbose=False)
 
      
    while True:
      pass

tableId, cubeId, targetId = getReadyForTask()
solution()