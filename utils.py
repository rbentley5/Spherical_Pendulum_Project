import numpy as np
import time



def rk4(fun, n, y0, t0, tfinal):
    start_time = time.time()
    h = (tfinal - t0) / n
    t = np.linspace(t0, tfinal, n+1)
    y0 = np.array(y0, dtype=float)
    y = np.empty((n+1, len(y0)), dtype=float)
    y[0] = y0
    
    for i in range(0, n):
        k1 = h * fun(t[i], y[i])
        k2 = h * fun(t[i] + h/2, y[i] + k1/2)
        k3 = h * fun(t[i] + h/2, y[i] + k2/2)
        k4 = h * fun(t[i] + h, y[i] + k3)

        y[i+1] = y[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    end_time = time.time()
    elasped = end_time-start_time
    print('RK4: '+str(elasped)+' secs')
    return t, y
def adams_bashforth(fun, n, y0, t0, tfinal):
    start_time = time.time()
    h = (tfinal - t0) / n
    t = np.linspace(t0, tfinal, n+1)
    y0 = np.array(y0, dtype=float)
    y = np.empty((n+1, len(y0)), dtype=float)
    f = np.empty((n, len(y0)), dtype=float)
    y[0] = y0

    f[0] = fun(t[0], y[0])
    y[1] = y[0] + h * f[0]

    f[1] = fun(t[1], y[1])
    y[2] = y[1] + h * (3 / 2 * f[1] - 1 / 2 * f[0])

    f[2] = fun(t[2], y[2])
    y[3] = y[2] + h * (23* f[2] - 16 * f[1] + 5 * f[0]) / 12

    f[3] = fun(t[3], y[3])
    y[4] = y[3] + h* (55 * f[3]- 59 * f[2] + 37 * f[1] - 9* f[0]) / 24

    for i in range(4, n):
        f[i] = fun(t[i], y[i])
        y[i+1] = y[i] + h* (1901 * f[i] - 2774 * f[i-1] + 2616 * f[i-2] - 1274* f[i-3] + 251*f[i-4]) / 720
    end_time = time.time()
    elasped = end_time-start_time
    print('Adams-Bashforth: '+str(elasped)+' secs')
    return t, y

def spherical_to_cartesian(l, theta, phi):
    x = l * np.sin(phi) * np.cos(theta)
    y = l*np.sin(phi) * np.sin(theta)
    z = l*(1 - np.cos(phi))
    return x, y, z