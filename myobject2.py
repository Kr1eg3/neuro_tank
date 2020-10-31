from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, ode
from scipy.stats import norm


class SysSolv:
    def __init__(self):
        self.ts = []
        self.ys = []
        self.tmax = 20
        self.alpha = pi/4
        self.V_c = 800
        self.G = 11
        self.g = 9.81
        self.ro = 1
        self.P = 1
        self.Cx_a = 0.35
        self.Cy_a = 0.35
        self.S_m = 4
        
    def fout(self):
        self.ts.append(t)
        self.ys.append(list(y.copy))
    
    def Sys(self, t, y):
        dV, dTeta, dx, dy, dm, dx_c, dy_c = y
        return [self.P*cos(self.alpha)/dm - self.Cx_a*self.ro*dV**2*self.S_m/(2*dm) - self.g*sin(dTeta),
                self.Cy_a*self.ro*dV**2*self.S_m/(2*(dm*dV)) + self.P*sin(self.alpha)/(dm*dV) - self.g*cos(dTeta)/dV,
                dV*cos(dTeta),
                dV*sin(dTeta),
                -self.G,
                self.V_c,
                0]
    
    
tmax = 10    
    
    
if __name__ == "__main__":
    ODE = ode(SysSolv.Sys)
    y0, t0 = [1, 1, 1, 1, 1, 1, 1], 0
    r=ode(lambda t, y: SysSolv.Sys.set_integrator('dopri5', max_step = 0.01))
    r.set_solout(SysSolv.fout)
    r = ODE.set_initial_value(y0, t0)
    ret = r.integrate(tmax)
    Y = array(SysSolv.ys)
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    ax.plot(Y[:,0], Y[:,2], linewidth = 3)
    ax.grid(True)
