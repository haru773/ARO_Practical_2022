from ntpath import join
from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base


# TODO: Rename class name after copying this file
class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1, 0, 0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1]),
        # 'RHAND'      : np.array([0, 0, 0]),
        # 'LHAND'      : np.array([0, 0, 0])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.array([0, 0, 0.85]),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335]),
        # 'RHAND'      : np.array([0, 0, 0]), # optional
        # 'LHAND'      : np.array([0, 0, 0]) # optional
    }

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta. ─
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        # TODO modify from here
        # Hint: the output should be a 3x3 rotational matrix as a numpy array
        # return np.matrix()
        if not theta:
            theta = self.getJointPos(jointName)
        Rx = np.matrix([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        Ry = np.matrix([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
        Rz = np.matrix([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        axis = self.jointRotationAxis[jointName]
        if np.array_equal(axis, np.array([0, 0, 1])):
            return Rz
        elif np.array_equal(axis, np.array([0, 1, 0])):
            return Ry
        else:
            return Rx

    def getTransformationMatrices(self):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}
        # TODO modify from here
        # Hint: the output should be a dictionary with joint names as keys and
        # their corresponding homogeneous transformation matrices as values.
        for joint in self.joints[2:]:
            # forming transformation matrix by composing rotational matrix and translation matrix and a addition row
            p_array = [[value] for value in self.frameTranslationFromParent[joint]]
            arr = np.append(self.getJointRotationalMatrix(joint, self.getJointPos(joint)), p_array, axis=1)
            transformationMatrices[joint] = np.append(arr, [[0, 0, 0, 1]], axis=0)
        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix odotf a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        # TODO modify from here
        # Hint: return two numpy arrays, a 3x1 array for the position vector,
        # and a 3x3 array for the rotation matrix
        indexx = [self.joints.index(x) for x in self.functionJoints(jointName)]
        transMatrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.85], [0, 0, 0, 1]]
        for i in indexx:
            transMatrix = transMatrix * self.getTransformationMatrices()[self.joints[i]]
        return transMatrix[0:3, 3].T, transMatrix[0:3, 0:3]

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointPositions(self, endEffector, angles):
        transformationMatrices = []
        joints = self.functionJoints(endEffector)
        for i in range(len(angles)):
            p_array = [[value] for value in self.frameTranslationFromParent[joints[i]]]
            arr = np.append(self.getJointRotationalMatrix(joints[i], angles[i]), p_array, axis=1)
            matrix = np.append(arr, [[0, 0, 0, 1]], axis=0)
            transformationMatrices.append(matrix)
        transMatrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.85], [0, 0, 0, 1]]
        positions = []
        for i in range(len(transformationMatrices)):
            transMatrix = transMatrix * transformationMatrices[i]
            positions.append(transMatrix[0:3, 3].T)
        return positions

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self, endEffector, pos=None, jointsPosition=None):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        # TODO modify from here
        # You can implement the cross product yourself or use calculateJacobian().
        # Hint: you should return a numpy array for your Jacobian matrix. The
        # size of the matrix will depend on your chosen convention. You can have
        # a 3xn or a 6xn Jacobian matrix, where 'n' is the number of joints in
        # your kinematic chain.
        # return np.array()
        jacobianMatrix = np.array([])
        if pos is None:
            pos = self.getJointPosition(endEffector)
        if jointsPosition is None:
            for joint in self.functionJoints(endEffector):
                # compute [a_n x (p_eff-p_n)] of each joint
                jacobianMatrix = np.append(jacobianMatrix, np.array(
                    np.cross(self.jointRotationAxis[joint], (pos - self.getJointPosition(joint)))).T)
        else:
            joints = self.functionJoints(endEffector)
            for i in range(len(joints)):
                jacobianMatrix = np.append(jacobianMatrix, np.array(
                    np.cross(self.jointRotationAxis[joints[i]], (pos - jointsPosition[i]))).T)
        jacobianMatrix = jacobianMatrix.reshape((len(self.functionJoints(endEffector)), 3))
        return jacobianMatrix.T

    def functionJoints(self, endEffector):
        index = self.joints.index(endEffector)
        if 'L' in endEffector:
            return [self.joints[2]] + self.joints[5:index + 1]
            # return self.joints[5:index+1]
        elif 'R' in endEffector:
            return [self.joints[2]] + self.joints[11:index + 1]
        else:
            return self.joints[2:index + 1]

    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, endEffector, targetPosition, orientation, interpolationSteps, threshold):
        curr_jointPosition = list(map(lambda x: self.getJointPos(x, 19), self.functionJoints(endEffector)))
        traj = np.array([curr_jointPosition])
        y_0 = self.getJointPosition(endEffector)
        y_curr = self.getJointPosition(endEffector)
        for t in range(1, int(interpolationSteps) + 1):
            jacobian = self.jacobianMatrix(endEffector, y_curr, self.getJointPositions(endEffector, curr_jointPosition))
            #print(jacobian)
            y_target = y_0 + (t / interpolationSteps) * (targetPosition - y_0)
            dy = y_target - y_curr
            dtheta = (np.array(np.linalg.pinv(jacobian) @ (dy).T).T).flatten()
            curr_jointPosition = curr_jointPosition + dtheta
            traj = np.append(traj, curr_jointPosition)
            y_curr = y_target
        traj = traj.reshape((interpolationSteps + 1, len(self.functionJoints(endEffector))))
        return traj

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
                        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # TODO add your code here
        # iterate through joints and update joint states based on IK solver

        iteration = int(
            np.linalg.norm(targetPosition - self.getJointPosition(endEffector)) * self.updateFrequency / speed)
        iteration = max(iteration, maxIter)
        pltDistance = np.array([])
        traj = self.inverseKinematics(endEffector, targetPosition, orientation=orientation,
                                      interpolationSteps=iteration, threshold=threshold)
        print('start updating ')
        for i in range(1, iteration + 1):
           # traj = self.inverseKinematics(endEffector, targetPosition, orientation=orientation,
            #                              interpolationSteps=iteration, threshold=threshold)
            self.tick_without_PD(traj[i], endEffector)
            pltDistance = np.append(pltDistance, np.linalg.norm(targetPosition - self.getJointPosition(endEffector)))
        return np.arange(0, iteration, 1), pltDistance

    def tick_without_PD(self, pos, endEffector):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
        # For each joint, you can use the shared variable self.jointTargetPos.
        index = [self.joints.index(x) for x in self.functionJoints(endEffector)]
        for i in range(len(index)):
            self.p.resetJointState(self.robot, self.jointIds[self.joints[index[i]]], pos[i])
        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller

    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        # TODO: Add your code here
        error = x_ref - x_real
        # print(error)
        d_err = dx_ref - dx_real
        integral = 0

        u_t = (kp * error) + (kd * d_err) + (ki * integral)

        return u_t

        # Task 2.2 Joint Manipulation

    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """

        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            # torque = 0.0
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
            pltTarget.append(targetPosition)
            pltPosition.append(xreal)
            pltVelocity.append(dxreal)
            ### To here ###

            pltTorque.append(torque)
            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)
        xreal = float(self.getJointPos(joint))
        xreal1 = 0
        # print(dxreal)
        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []
        steps = 0
        for i in range(1000):
            #print(abs(targetPosition - xreal))
            # print(targetPosition)
            # print(xreal)
            xreal = float(self.getJointPos(joint))
            dxreal = (xreal -xreal1) *1000
            toy_tick(targetPosition, xreal, targetVelocity, dxreal, integral=0)
            xreal1 = xreal
            steps += 1
        pltTime = np.arange(0, steps, 1)
        pltTorqueTime = np.arange(0, steps, 1)
        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
                     threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

        # return pltTime, pltDistance
        iteration = int(
            np.linalg.norm(targetPosition - self.getJointPosition(endEffector)) * self.updateFrequency / speed)
        iteration = max(iteration, maxIter)
        pltDistance = np.array([])
        traj = self.inverseKinematics(endEffector, targetPosition, orientation=orientation,
                                      interpolationSteps=iteration,
                                      threshold=threshold)
        xreal_prev = [0] * len(traj[0])
        for i in range(1, iteration + 1):
            xreal_prev = self.tick(traj[i], endEffector, xreal_prev)
            pltDistance = np.append(pltDistance, np.linalg.norm(targetPosition - self.getJointPosition(endEffector)))
        return np.arange(0, iteration, 1), pltDistance

    def tick(self, pos, endEffector, xreal_prev,integral=0):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        index = [self.joints.index(x) for x in self.functionJoints(endEffector)]

        for i in range(len(pos)):
            # skip dummy joints (world to base joint)
            joint = [self.joints[index[i]]][0]
            #print(joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Implement your code from here ... ###
            targetPos = pos[i]
            #print(targetPos)
            targetVel = 0
            xreal = float(self.getJointPos(joint))
            #print(xreal)
            #calc velocity
            dxreal = (xreal - xreal_prev[i]) * 1000
            #update prev velocity lst
            xreal_prev[i] = xreal

            # TODO: obtain torque from PD controller
            torque = self.calculateTorque(targetPos, xreal, targetVel, dxreal, integral, kp, ki, kd)
            ### ... to here ###

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)
        return xreal_prev

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        # TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array),
        # sampled from a cubic spline defined by 'points' and a boundary condition.
        # You may use methods found in scipy.interpolate

        # return xpoints, ypoints
        pass

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
                          threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

### END
