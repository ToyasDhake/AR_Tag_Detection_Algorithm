import numpy as np


class KF2d():
    # Initialize
    def __init__(self, dt=1):
        super(KF2d, self).__init__()
        self.dt = dt
        self.A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=np.float)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        self.Q = 0.9 * np.eye(4, dtype=np.float)
        self.R = np.array([
            [100, 0],
            [0, 100]
        ], dtype=np.float)

        self.zero_cnt = 0
        self.flg_disappear = True

    # Main function that computes prediction
    def compute_kalman(self, x, P, z):
        xp = self.A @ x
        Pp = self.A @ P @ self.A.T + self.Q

        HPH = np.array([
            [Pp[0, 0], Pp[0, 2]],
            [Pp[2, 0], Pp[2, 2]],
        ])
        HPHR = HPH + self.R
        inv_HPHR = np.array([
            [HPHR[1, 1], -HPHR[0, 1]],
            [-HPHR[1, 0], HPHR[0, 0]]
        ])
        inv_HPHR /= (HPHR[0, 0] * HPHR[1, 1] - HPHR[0, 1] * HPHR[1, 0])
        K = Pp @ self.H.T @ inv_HPHR
        # Predicted mean
        x = xp + K @ (z - self.H @ xp)
        # covariance of state
        P = Pp - K @ self.H @ Pp
        return x, P

    # Pre processing of state
    def pre_process(self, x, P, z):
        if z[0] != 0 and z[1] != 0 and self.flg_disappear == True:
            self.flg_disappear = False
            x[0] = z[0]
            x[1] *= 0.1
            x[2] = z[1]
            x[3] *= 0.1
            P = 0 * np.eye(4, dtype=np.float)

        if x[0] != 0 and x[2] != 0 and z[0] == 0 and z[1] == 0:
            z[0] = x[0]
            z[1] = x[2]
            self.zero_cnt += 1
        else:
            self.zero_cnt = 0

        if self.zero_cnt >= 5:
            self.zero_cnt = 0
            self.flg_disappear = True
            x = np.array([0, 0, 0, 0], dtype=np.float)
            P = 0 * np.eye(4, dtype=np.float)

        if abs(x[1]) > 5 or abs(x[3]) > 5:
            x[0] = z[0]
            x[1] *= 0.1
            x[2] = z[1]
            x[3] *= 0.1
            P = 0 * np.eye(4, dtype=np.float)
        return x, P, z

    # Post processing of data
    def post_process(self, x, P):

        if abs(x[1]) > 10 or abs(x[3]) > 10:
            output = (0, 0)
        else:
            output = (int(round(x[0])), int(round(x[2])))
        if x[0] < 10 or x[2] < 10:
            output = (0, 0)
        return x, P, output

    # Function to execute all required actions in order
    def process(self, x, P, z):
        output = None
        # Pre process
        x, P, z = self.pre_process(x, P, z)
        # Predict
        x, P = self.compute_kalman(x, P, z)
        # Post process
        x, P, output = self.post_process(x, P)
        return x, P, output
