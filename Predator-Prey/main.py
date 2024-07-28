import numpy as np
import matplotlib.pyplot as plt

class LotkaVolterra:
    """
    Lotka-Volterra Model
    dR/dt = alpha * R - beta * R * P
    dP/dt = delta * R * P - gamma * P
    """
    def __init__(self, R0, P0, alpha, beta, delta, gamma):
        self.R0 = R0
        self.P0 = P0
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        
    def dR_dt(self, R, P):
        """
        Return the growth rate of prey population
        
        Parameters:
        R: number of preys
        P: number of predators
        """
        return self.alpha * R - self.beta * R * P

    def dP_dt(self, R, P):
        """
        Return the growth rate of predator population

        Parameters:
        R: number of preys
        P: number of predators
        """
        return self.delta * R * P - self.gamma * P

    def solve(self, h, T):
        """
        Solve the Lotka-Volterra model using Runge-Kutta 4th order method

        Parameters:
        h: time step
        T: total time
        """
        N = int(T / h)
        R = np.zeros(N+1)
        P = np.zeros(N+1)
        t = np.zeros(N+1)

        R[0] = self.R0
        P[0] = self.P0
        t[0] = 0

        for i in range(N):
            k1_R = h * self.dR_dt(R[i], P[i])
            k1_P = h * self.dP_dt(R[i], P[i])

            k2_R = h * self.dR_dt(R[i] + 0.5 * k1_R, P[i] + 0.5 * k1_P)
            k2_P = h * self.dP_dt(R[i] + 0.5 * k1_R, P[i] + 0.5 * k1_P)

            k3_R = h * self.dR_dt(R[i] + 0.5 * k2_R, P[i] + 0.5 * k2_P)
            k3_P = h * self.dP_dt(R[i] + 0.5 * k2_R, P[i] + 0.5 * k2_P)

            k4_R = h * self.dR_dt(R[i] + k3_R, P[i] + k3_P)
            k4_P = h * self.dP_dt(R[i] + k3_R, P[i] + k3_P)

            R[i+1] =R[i]+(1/6) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
            P[i+1] =P[i]+(1/6) * (k1_P + 2*k2_P + 2*k3_P + k4_P)
    
        return R, P
    
    def plot(self, h, T):
        R, P = self.solve(h, T)
        t = np.arange(0, T+h, h)
        
        plt.figure(figsize=(10, 5))
        plt.plot(t, R, label='Preys (R)')
        plt.plot(t, P, label='Predators (P)')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        plt.title('Lotka-Volterra Model by Runge-Kutta 4th Order')
        plt.grid()
        plt.show()


R0 = 40 
P0 = 9  
alpha = 0.1 
beta = 0.02 
delta = 0.01 
gamma = 0.1

h = 0.1
T = 50

if __name__ == '__main__':
    model = LotkaVolterra(R0, P0, alpha, beta, delta, gamma)
    model.plot(h, T)