import numpy as np
from abc import ABC, abstractmethod


class Controller(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate(self,                  
                 current_pose_T, # in T
                 current_joint_angles,
                 f_des,
                 f_meas = None,
                 ):
        pass

    @abstractmethod
    def reset(self):
        pass


class PIDForceController(Controller):

    def __init__(self,
                 n_outputs,
                 kp = np.array([100]),
                 kd = np.array([10]),
                 ki = np.array([1]),
                 dt = 0.01,
                 ):
        
        self.pid_controller = PID(n_outputs=n_outputs,
                                    kp=kp,
                                    kd=kd,
                                    ki=ki,
                                    dt=dt)

    def calculate(self,
                f_des,
                f_meas,
                ):
        return self.pid_controller.calculate(f_meas,f_des)

    
    def reset(self):
        self.pid_controller.reset()
    
class PID(Controller):
    
    def __init__(self,
                 n_outputs,
                 kp = np.array([100, 100, 80, 40, 20, 10]),
                 kd = np.array([10, 10, 8, 4, 2, 1]),
                 ki = np.array([1, 1, 0.8, 0.4, 0.2, 0.1]),
                 dt = 0.01):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt
        
        self.error_shape = (n_outputs)
        self.error_prev = np.zeros(self.error_shape)
        self.error_integral = np.zeros(self.error_shape)

    def reset(self):
        self.error_prev = np.zeros(self.error_shape)
        self.error_integral = np.zeros(self.error_shape)

    def calculate(self, q, qdes):
        error = qdes - q
        errordot = (error - self.error_prev)/self.dt
        self.error_integral += error*self.dt
        self.error_prev = error.copy()
        u = (self.kp * error + 
                self.kd * errordot + 
                self.ki * self.error_integral)
        return u
    
class Admittance(Controller):
    
    def __init__(self,
                 n_outputs,
                 virtual_inertia = np.zeros(6),
                 virtual_damping = np.ones(6)*0.05,
                #  virtual_stiffness = np.ones(6)*1,
                #  vlim = 4,
                 deadband = None,
                 direction = 1,
                #  virtual_stiffness = np.zeros(6),
                 dt = 0.01):

        self.dt = dt
        self.Mv = virtual_inertia
        if np.sum(np.abs(self.Mv)) > 0.001: 
            self.Mv_inv = np.linalg.pinv(self.Mv)
        else: self.Mv_inv = None

        self.Dv = virtual_damping
        if np.sum(np.abs(self.Dv)) > 0.00001: 
            self.Dv_inv = np.linalg.pinv(self.Dv)
        else: self.Dv_inv = None

        # self.vlim = vlim
        
        self.error_shape = (n_outputs)
        self.twist_ref = np.zeros(self.error_shape)
        self.deadband = deadband
        self.direction = direction
        self.vref_buffer = []

    def reset(self):
        self.twist_ref = np.zeros(self.error_shape)

    def calculate(self, 
                        Fmeas, 
                        Fdes, 
                        ):
        """ solve for D*vdot = Fmeas-Fdes
        (note that error is flipped from normal desired-measured because to reduce force,
        velocity goes opp direction.)

        """
        
        error = Fmeas - Fdes

        # error *= self.direction
        if self.deadband is not None:
            for i,e in enumerate(error):
                if np.abs(e) < self.deadband:
                    error[i] = 0
        
        if self.Mv_inv is not None:
            vdot_des = self.Mv_inv @ (error - self.Dv @ self.twist_ref )
            vadd = vdot_des * self.dt

            vadd[np.isnan(vadd)] = 0

            self.twist_ref += vadd
        
        else:
            self.twist_ref = self.Dv_inv @ error
        
        return self.twist_ref.copy()
