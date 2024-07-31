import numpy as np
import matplotlib.pyplot as plt


class PopulationGrowth:
    
    def __init__(self, p0, r, K, h, T):
        self.p0 = p0
        self.r = r
        self.K = K
        self.h = h
        self.T = T
        
    def dp_dt(self, p):
        """
        Return the growth rate of population
        
        Parameters:
        p: population
        """
        return self.r * p * (1 - p / self.K)
    
    def solve(self):
        """
        Solve the population growth model using Runge-Kutta 4th order method
        """
        N = int(self.T / self.h)
        p = np.zeros(N + 1)
        t = np.zeros(N + 1)
        
        p[0] = self.p0
        t[0] = 0
        
        for i in range(N):
            k1 = self.h * self.dp_dt(p[i])
            k2 = self.h * self.dp_dt(p[i] + 0.5 * k1)
            k3 = self.h * self.dp_dt(p[i] + 0.5 * k2)
            k4 = self.h * self.dp_dt(p[i] + k3)
            
            p[i + 1] = p[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t[i + 1] = t[i] + self.h
            
        return t, p
    
    def plot(self):
        """
        Plot the population growth model
        """
        
        t, p = self.solve()
        
        plt.plot(t, p, 'b')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Population Growth Model')
        plt.show()
        
        print(f"Population: {np.round(p[-1])}")
        

if __name__ == "__main__":
    p0 = 10
    r = 0.1
    K = 1000
    h = 0.1
    T = 20
    
    model = PopulationGrowth(p0, r, K, h, T)
    model.plot()
    