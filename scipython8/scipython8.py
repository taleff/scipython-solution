import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.constants as pc
from scipy.optimize import brentq, newton, minimize, minimize_scalar, curve_fit
import scipy.integrate

#P8.1.1
def pascal(rows):
    for n in range(rows):
        row = []
        for k in range(n+1):
            row.append(int(scipy.special.binom(n, k)))
        print(*row, sep = ' ')
        
#P8.1.2
def airy_pattern():
    x = np.linspace(-10, 10, 2000)
    I = (2*scipy.special.jn(1, x)/x) ** 2
    zeros = scipy.special.jn_zeros(1, 1)
    
    lam = 500*10**(-9)
    a = 1.5*10**(-3)
    k = 2*np.pi/lam
    theta = np.arcsin(x/(k*a)) * 3600  
    zeros = np.arcsin(zeros/(k*a)) * 3600 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(theta, I)
    ax.set_xlabel('Angle (arcsec)')
    
    plt.show()
    print('First Minimum: {:.3} arcseconds'.format(zeros[0]))

#P8.1.3
def get_wv(D0):
    bond_energy = D0 * 1000 / pc.N_A
    wv = (pc.c * pc.h / bond_energy) * 10**9
    return wv
    
#P8.1.4
def ellipsoid_surface(a, b, c):
    phi = np.arccos(c/a)
    k = (a*np.sqrt(b**2-c**2)) / (b*np.sqrt(a**2-c**2))
    
    S = 2*np.pi*c**2 + (2*np.pi*a*b/np.sin(phi))*( \
                        scipy.special.ellipkinc(phi, k**2)*np.cos(phi)**2 + \
                        scipy.special.ellipeinc(phi, k**2)*np.sin(phi)**2)
    
    r = phi/np.sin(phi)
    S_approx = 2*np.pi*c**2 + 2*np.pi*a*b*r*(1-(b**2-c**2)*r**2/(6*b**2)*
                                             (1-(3*b**2 + 10*c**2)*r**2/(56*b**2)))
    
    return S, S_approx
    
#P8.1.5
def drawdown():
    def theis(Q, T, S, r, t):
        u = r**2 * S / (4*T*t)
        s = Q / (2*np.pi*T) * scipy.special.exp1(u)
        return s
        
    Q, H0, S, T = 1000, 20, 0.0003, 1000
    r = np.linspace(-500, 500, 1000)
    s = theis(Q, T, S, r, 1)
    
    gamma = 0.577215664
    W_j = -gamma - np.log(r**2 * S / (4*T))
    j = Q / (2*np.pi*T) * W_j
    
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.axhline(H0, color = 'b', ls = 'solid', lw = 2, label = 'Water Table')
    ax.axvline(0, 0.1, 0.8, color = 'k', lw = 5)
    ax.plot(r, H0-s, 'b--', label = 'Theis')
    ax.plot(r, H0 - j, 'g-.', label = 'Jacob')
    
    ax.set_xlim(-500, 500)
    ax.set_ylim(15, 22)
    ax.set_yticklabels(range(-5, 3))
    ax.minorticks_on()
    ax.yaxis.grid(True)
    plt.legend(loc = 'lower left')
    plt.show()
    
#P8.1.6
def heatsink(w, r0, r1, T0 = 400, Te = 300):
    hc = 10 #W*m^-2*K^-1
    kappa = 200 #W*m^-1*K^-1
    beta = np.sqrt(hc / (kappa*(w/2)))
    u0, u1 = beta*r0, beta*r1
    coeff = 2*r0 / (beta*(r1**2-r0**2))
    
    numerator = scipy.special.kn(1, u0)*scipy.special.iv(1, u1) - \
                scipy.special.kn(1, u1)*scipy.special.iv(1, u0)
    denominator = scipy.special.kn(0, u0)*scipy.special.iv(1, u1) + \
                  scipy.special.kn(1, u1)*scipy.special.iv(0, u0)
                  
    eta = coeff * numerator / denominator
    
    area = np.pi * (r1**2 - r0**2)
    Qdot = eta*area*(T0-Te)
    
    return eta, Qdot
    
#Q8.2.1
def floor_func():
    f = lambda x: x//1 - 2*((x/2)//1)
    
    area = scipy.integrate.quad(f, 0, 6, points = [i for i in range(0, 7)])
    return area

#Q8.2.2
def definite_integrals(ev, p = 0, z = 0):
    funcs = [None]*5
    
    funcs[0] = lambda x: (x**4)*((1-x)**4) / (1+(x**2))
    funcs[1] = lambda x: (x**3) / ((np.e**x) - 1)
    funcs[2] = lambda x: x**(-x)
    funcs[3] = lambda x, p: (np.log(1/x))**p
    funcs[4] = lambda x, z: np.e**(z*np.cos(x))
    
    bounds = {0:(0, 1), 1:(0, np.inf), 2:(0, 1), 3:(0, 1), 4:(0, 2*np.pi)}
    args = {0:(), 1:(), 2:(), 3:(p), 4:(z)}
    
    ans = scipy.integrate.quad(funcs[ev], *bounds[ev], args = args[ev])
    return ans

#Q8.2.3
def pi_approx():
    f = lambda y, x: 4
    upper_bnd = lambda x: np.sqrt(1 - x**2)
    lower_bnd = lambda x: 0
    return scipy.integrate.dblquad(f, 0, 1, lower_bnd, upper_bnd)

#P8.2.1
def surface_area():
    def root(x):
        return np.sqrt(x), 0.5 / np.sqrt(x)
    
    def revolution(func, a, b):
        f = lambda x: func(x)[0]*np.sqrt(1+func(x)[1]**2)
        return 2 * np.pi * scipy.integrate.quad(f, a, b)[0]
    
    print(revolution(root, 0, 1))
    print(np.pi*(5**(3/2)-1)/6)
    
#P8.2.2
def secants():
    theta = np.linspace(-np.pi/2, np.pi/2, 101)
    f = lambda x: 1/np.cos(x)
    integ = [scipy.integrate.quad(f, 0, angle)[0] for angle in theta]
    
    guder = np.log(abs((1/np.cos(theta)) + np.tan(theta)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(theta, integ, 'b--', label = 'Scipy Integration')
    ax.plot(theta, guder, 'g-', label = 'Gudermannian function')
    
    ax.set_xticks(np.linspace(-3, 3, 7)*np.pi/6)
    ax.set_xticklabels([r'$\frac{-\pi}{2}$', r'$\frac{-\pi}{3}$', r'$\frac{-\pi}{6}$', 
                        r'$0$', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$'],
                        fontsize = 14)
    ax.grid()
    ax.minorticks_on()
    plt.legend(loc = 'upper left')
    plt.show()
    
#P8.2.3
def torus():
    R, r = 4, 1    
    
    f_v = lambda z, rho, theta: 2*rho
    upper_z = lambda theta, rho: np.sqrt(r**2 - (rho-R)**2)
    lower_z = lambda theta, rho: 0
    upper_rho = lambda theta: R+r
    lower_rho = lambda theta: R-r
    vol_approx = scipy.integrate.tplquad(f_v, 0, 2*np.pi, lower_rho, upper_rho,
                                         lower_z, upper_z)
                                         
    f_Iz = lambda z, rho, theta: (2/vol_approx[0])*(rho**3)
    Iz_approx = scipy.integrate.tplquad(f_Iz, 0, 2*np.pi, lower_rho, upper_rho,
                                         lower_z, upper_z)   
                                         
    f_Ix = lambda z, rho, theta: (2/vol_approx[0])*((rho**2)*(np.sin(theta)**2)+(z**2))*rho
    Ix_approx = scipy.integrate.tplquad(f_Ix, 0, 2*np.pi, lower_rho, upper_rho,
                                         lower_z, upper_z)
                                         
    vol = 2*(np.pi**2)*R*(r**2)
    I_z = R**2 + 0.75*(r**2)
    I_x = 0.5*(R**2) + (5/8)*(r**2)
    
    print(vol, vol_approx[0], I_z, Iz_approx[0], I_x, Ix_approx[0])
    
#8.2.4
def brusselator():
    a, b = 1, 1.8    
    
    def deriv(vals, t, a, b):
        x, y = vals
        dxdt = a - (1+b)*x + (x**2)*y
        dydt = b*x - (x**2)*y
        return dxdt, dydt
    
    init = (0, 0)
    t = np.linspace(0, 50, 1000)    
    x, y = scipy.integrate.odeint(deriv, init, t, args = (a, b)).T
    
    fig = plt.figure(figsize = (11.5, 8))
    ax1 = fig.add_subplot(121)
    ax1.plot(t, x, 'b-', label = 'Relative Conc. of X')
    ax1.plot(t, y, 'r-', label = 'Relative Conc. of Y')
    ax1.set_xlabel('Time', fontsize = 14)
    ax1.set_ylabel('Relative Concentration', fontsize = 14)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y, 'b-')
    ax2.set_xlabel('Relative Concentration of X', fontsize = 14)
    ax2.set_ylabel('Relative Concentration of Y', fontsize = 14)
    
    ax1.legend()
    plt.show()

#8.2.5
def pendulum():
    g = 9.81
    l = 1    
    
    def deriv(x, t, g, l):
        x, xdot = x
        dtheta = xdot
        dtheta2 = -(g/l)*np.sin(x)
        return dtheta, dtheta2
        
    init = (np.pi/3, 0)    
    t = np.linspace(0, 5, 100)
    
    theta, thetadot = scipy.integrate.odeint(deriv, init, t, args = (g, l)).T
    theta_approx = init[0]*np.cos(np.sqrt(g/l)*t)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, theta, label = 'Numerical Solution')
    ax.plot(t, theta_approx, label = 'Small Angle Approximation')
    
    plt.legend(fontsize = 10)
    plt.show()
    
#8.2.6
def ozone():
    k1 = 3 * 10**(-12) #s^-1
    k2 = 1.2 * 10**(-33) #cm^6 molec^-2 s^-1
    k3 = 5.5 * 10**(-4) #s^-1
    k4 = 6.9 * 10**(-16) #cm^3 molec^-1 s^-1
    M = 9 * 10**(17) # molec cm^-3
    conversion = (100)**3 / scipy.constants.N_A * 10**6
    
    def ozone_deriv(x, t, k1, k2, k3, k4, M):
        O2, O, O3 = x
        dO2dt = -k1*O2 - k2*O2*O*M + k3*O3 + 2*k4*O*O3
        dOdt = 2*k1*O2 - k2*O2*O*M + k3*O3 - k4*O*O3
        dO3dt = k2*O2*O*M - k3*O3 - k4*O*O3
        return dO2dt, dOdt, dO3dt
        
    t = np.linspace(0, 0.5 * 10**8, 10000)
    init = (0.21 * M, 0, 0)
    O2, O, O3 = scipy.integrate.odeint(ozone_deriv, init, t, 
                                       args = (k1, k2, k3, k4, M)).T
                                       
    O3_approx = np.sqrt(k1*k2/(k3*k4)) * init[0] * np.sqrt(M) * conversion
    O_approx = k3/(k2*init[0]*M) * O3_approx
                                       
    fig = plt.figure(figsize = (11.5, 8))
    ax1 = fig.add_subplot(121)
    ax1.plot(t, O * conversion, 'b-', label = 'O Concentration (Numerical)')
    ax1.axhline(O_approx, color = 'b', ls = '--', label = 'O Concentration (Steady State)')
    ax1.set_xlabel('Time /s', fontsize = 12)
    ax1.set_ylabel('Molar Concentration (millionths)', fontsize = 12)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(t, O3 * conversion, 'r-', label = 'O3 Concentration (Numerical)')
    ax2.axhline(O3_approx, color = 'r', ls = '--', label = 'O3 Concentration (Steady-State)')
    ax2.set_xlabel('Time /s', fontsize = 12)
    ax2.set_ylabel('Molar Concentration (millionths)', fontsize = 12)
    
    ax1.legend(loc = 'lower right', fontsize = 12)
    ax2.legend(loc = 'lower right', fontsize = 12)
    
    
    plt.show()
    
#P8.2.7ar = lambda phi: (1 + e*np.cos(phi)) / (1 - e**2)
def hyperion():
    e = 0.1
    BAC = 0.265
    
    def rotation(angle, phi, e, BAC):
        ar = lambda phi: (1 + e*np.cos(phi)) / (1 - e**2)
        
        omega, theta = angle
        domegadphi = -BAC * (3/(2*(1-e**2))) * ar(phi) * np.sin(2*(theta - phi))
        dthetadphi = omega * ((1 / ar(phi))**2)
        return domegadphi, dthetadphi
    
    init = (2, 0)
    phi = np.linspace(0, 10*np.pi, 1000)
    omega, theta = scipy.integrate.odeint(rotation, init, phi, args = (e, BAC)).T
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(phi, omega, 'b-', label = 'Rotation')
    ax.set_xlabel('Revolutions')
    ax.set_ylabel('Spin Rate')
    ax.set_xticks(np.arange(0, 10*np.pi + 1, 2*np.pi))
    ax.set_xticklabels(range(6))
    ax.set_xlim(0, 10 * np.pi)
    
    plt.show()

#P8.2.8
def lead():
    k1 = 1.816 * 10**(-5)
    k2 = 6.931 * 10**(-5)
    k3 = 1.232 * 10**(-4)
    k4 = 3.851 * 10**(-3)
    k5 = 2.310
    
    def lead_deriv(x, t, k1, k2, k3, k4, k5):
        Pb, Bi, Tl, Po, Stable = x
        dPbdt = -k1*Pb
        dBidt = k1*Pb - k2*Bi - k3*Bi
        dTldt = k2*Bi - k4*Tl
        dPodt = k3*Bi - k5*Po
        dStabledt = k4*Tl + k5*Po
        return dPbdt, dBidt, dTldt, dPodt, dStabledt
        
    t = np.linspace(0, 0.5*10**5, 1000)
    init = (1, 0, 0, 0, 0)
    
    Pb, Bi, Tl, Po, Stable = scipy.integrate.odeint(lead_deriv, init, t, args = (k1, k2, k3, k4, k5)).T
    elements = (Pb, Bi, Tl, Po, Stable)
    element_names = ('Lead 212', 'Bismuth 212', 'Thallium 208', 
                     'Polonium 212', 'Lead 208')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for element in range(5):
        ax.plot(t, elements[element], label = element_names[element])
        
    ax.set_xlabel('Time /s')
    ax.set_ylabel('Concentration')
        
    plt.legend(fontsize = 10)
    plt.show()
    
#Q8.4.1
def rational_root():
    func = lambda x: -(1/(x-3)**3) - x - 1
    
    fig = plt.figure()
    x = np.linspace(-5, 5, 1000)
    ax = fig.add_subplot(111)
    ax.plot(x, func(x))
    ax.axhline(0, color = 'k')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)    
    
    plt.show()
    
    roots = []
    roots.append(brentq(func, -2, 0))
    roots.append(brentq(func, 2, 2.5))
    roots.append(brentq(func, 2.5, 3.5))
    
    print(roots)
    
#Q8.4.2
def newtons(x, xnew = 0):
    func = [None] * 4
    deriv = [None] * 4
    
    func[0] = lambda x: x**3 - 5*x
    deriv[0] = lambda x: 3*x**2 - 5
    
    func[1] = lambda x: x**3 - 3*x + 1
    deriv[1] = lambda x: 3*x**2 - 3

    func[2] = lambda x: 2 - x**5
    deriv[2] = lambda x: -5*x**4

    func[3] = lambda x: x**4 - 4.29*x**2 - 5.29
    deriv[3] = lambda x: 4*x**3 - (2*4.29)*x
    
    fig = plt.figure()
    t = np.linspace(-5, 5, 1000)
    ax = fig.add_subplot(111)
    ax.plot(t, func[x](t))
    ax.axhline(0, color = 'k')
    ax.axvline(0, color = 'k')
    
    plt.show()
    
    return newton(func[x], fprime = deriv[x], x0 = xnew)
    
#Q8.4.3
def projectile():
    h, x, v0 = 5, 15, 25    
    
    func = lambda theta, h, x, v0: x*np.tan(theta) - (9.81 / (2*v0**2*np.cos(theta)))*(x**2) 
    
    fig = plt.figure()
    t = np.linspace(0, np.pi, 1000)
    ax = fig.add_subplot(111)
    ax.plot(t, func(t, h, x, v0))
    plt.xlim(0, np.pi)
    plt.ylim(-50, 50)
    ax.axhline(0, color = 'k')
    ax.axvline(0, color = 'k')
    
    return brentq(func, 0, 0.5, args = (h, x, v0))
    
#P8.4.1
def fencing():
    perimeter = lambda a: 2*a + (10000 / a)
    
    a = minimize(perimeter, 100)['x'][0]
    b = 10000/a
    return a, b

#P8.4.2a
def roots_brent():
    f = lambda x: 0.2 + x*np.cos(3/x)
    x = np.linspace(-4, 4, 801)
    roots = []
    
    for begin in x:
        end = begin + 0.01
        try:
            roots.append(brentq(f, begin, end))
        except ValueError:
            continue
        except ZeroDivisionError:
            continue
    
    return roots

#P8.4.2b
def roots_newton():
    f = lambda x: 0.2 + x*np.cos(3/x)
    x = np.linspace(-4, 4, 801)
    roots = set()
    
    for guess in x:
        try:
            new = newton(f, guess)
            for root in roots:
                if np.isclose(new, root):
                    break
            else:
                roots.add(new)
        except RuntimeError:
            continue
        
    return roots
    
#P8.4.3
def wien():
    def planck_neg(lam, T):
        c = 2.99792458 * 10**8
        h = 1.054571726 * 10**(-34) * 2 * np.pi
        k_B = 1.3806488 * 10**(-23)
        
        coeff = 2 * h * (c**2) / (lam**5)
        denom = np.exp(h*c/(lam*k_B*T)) - 1
        
        return -coeff * (1/denom)
        
    T = np.linspace(500, 6000, 56)
    max_lam = []
    bracket = (250*10**(-9), 500*10**(-9))
    
    for temp in T:
        max_lam.append(minimize_scalar(planck_neg, bracket = bracket, args = (temp))['x'])
    max_lam = np.array(max_lam)

    def wien_law(T, b):
        return b / T

    param, _ = curve_fit(wien_law, T, max_lam, 1)

    fig = plt.figure(figsize = (11.5, 8))
    ax = fig.add_subplot(111)
    ax.scatter(T, max_lam * 10**9, label = 'Maximum Emission Wavelength')
    ax.plot(T, param[0] / T * 10**9, '--', label = "Wien's Displacement Law")
    
    plt.legend()
    plt.show()
    
    return param[0]
    
#P8.4.4
def schrodinger(N):
    def estim_energy(psi):
        numer = scipy.integrate.quad(psi*psi.deriv(2), -1, 1)[0]
        denom = scipy.integrate.quad(psi*psi, -1, 1)[0]
        return -numer / denom

    def trial(x, N):
        Polynomial = np.polynomial.Polynomial
        trial = Polynomial([0])
        for n in range(N+1):
            phi = (Polynomial([1, -1])**(N-n+1)) * (Polynomial([1, 1])**(n+1))
            trial += x[n] * phi
        return trial
        
    def optimize_trial(x, N):
        func = trial(x, N)
        return estim_energy(func)
        
    energy = scipy.optimize.minimize(optimize_trial, [0.1]*(N+1), args = (N))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = np.linspace(-1, 1, 100)
    
    exact_func = lambda x: np.cos(np.pi*x/2)
    ax.plot(t, exact_func(t), label = 'Exact Solution')
    
    trial_func = trial(energy['x'], N)
    trial_func /= trial_func(0)
    ax.plot(t, trial_func(t), label = 'Trial Solution (N={})'.format(N))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.4)
    
    plt.legend()
    plt.show()
    
    print('Exact Energy: {:.5f}'.format(np.pi**2/4))
    print('Estimated Energy: {:.5f}'.format(energy['fun']))

    return energy
                