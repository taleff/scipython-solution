import math
import pylab
pylab.rc('text', usetex=True)

#P3.1.1
def functions_one():
    x = pylab.linspace(-20, 20, 1001)
    y1 = pylab.log(1 / (pylab.cos(x) ** 2))
    y2 = pylab.log(1 / (pylab.sin(x) ** 2))
    pylab.plot(x, y1)
    pylab.plot(x, y2)
    pylab.show()
    
#P3.1.2
def michaelis(v_max = 0.1, K_m = 0.04):
    x = pylab.linspace(0, 1, 1001)
    y1 = pylab.array(x)
    y1 = (v_max * y1) / (K_m + y1)
    pylab.plot(x, y1)
    pylab.show()
    
#P3.1.3
def gauss():
    x = pylab.linspace(-5, 5, 1001)
    y = pylab.array(x)
    y1 = (1 / (1 * math.sqrt(2 * math.pi))) * pylab.exp(-y ** 2 / (2 * 1 ** 2))
    y2 = (1 / (1.5 * math.sqrt(2 * math.pi))) * pylab.exp(-y ** 2 / (2 * 1.5 ** 2))
    y3 = (1 / (2 * math.sqrt(2 * math.pi))) * pylab.exp(-y ** 2 / (2 * 2 ** 2))
    pylab.plot(x, y1)
    pylab.plot(x, y2)
    pylab.plot(x, y3)
    pylab.show

#P3.2.1
def reaction(init = 2.0, k1 = 300, k2 = 100):
    t = pylab.linspace(0, 10, 1001)
    A = init * pylab.exp(-(k1 + k2) * (t / 1000))
    B = (k1 / (k1 + k2)) * init * (1 - pylab.exp(-(k1 + k2)* (t / 1000)))
    C = (k2 / (k1 + k2)) * init * (1 - pylab.exp(-(k1 + k2)* (t / 1000)))
    pylab.plot(t, A, 'b--')
    pylab.plot(t, B, 'r-')
    pylab.plot(t, C, 'g-')
    pylab.xlabel('Time (ms)')
    pylab.ylabel('Concentration (M)')
    pylab.show()
    
#P3.2.2
def prime(numb):
    for n in range(2, int(math.sqrt(numb) // 1) + 1):
        if not numb % n:
            return False
    return True
    
#P3.2.2
# Seeds: 12+28i (2956), 43+55i (2956), 232+277i (316268), 3+5i (1000), 5+23i (1000)
def gaussian(start, steps):
    x_list = []
    y_list = []
    dx, dy = 1, 0
    x = start.real - 1
    y = start.imag.real
    for n in range(steps):
        while True:
            x, y = x + dx, y + dy
            if ((x == 0 and y > 0 and (not (y - 3) % 4) and prime(y)) or 
                (x == 0 and y < 0 and (not (y + 3) % 4) and prime(y)) or
                (y == 0 and x > 0 and (not (x - 3) % 4) and prime(x)) or 
                (y == 0 and x < 0 and (not (x + 3) % 4) and prime(x)) or 
                (x != 0 and y != 0 and prime((x)**2 + (y)**2))):
                    dx, dy = -dy, dx
                    x_list.append(x)
                    y_list.append(y)                    
                    break
    pylab.plot(x_list,y_list)
    pylab.title('Gaussian Prime Spiral')
    pylab.xlabel('Real')
    pylab.ylabel('Imaginary')
    pylab.show
    
#P3.2.3
def UK_death():       
    x = [1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 85]
    female = [227, 5376, 10417, 4132, 2488, 1106, 421, 178, 65, 21, 7]
    male = [177, 4386, 8333, 1908, 1215, 663, 279, 112, 42, 15, 6]
    pylab.plot(x, male, 'b--', x, female, 'r-.')
    pylab.show()
    
#P3.3.1a
def archimedean(s, v, end):
    theta = pylab.linspace(0, end, 1001)
    r = s + v * pylab.array(theta)
    pylab.polar(theta, r, label = r'$r = {} + {}\theta$'.format(s, v))
    pylab.legend()
    pylab.show()
    
#P3.3.1b
def logarithmic(a, end):
    theta = pylab.linspace(0, end, 1001)
    r = a ** pylab.array(theta)
    pylab.polar(theta, r, label = r'$r = {}^\theta$'.format(a))
    pylab.legend()
    pylab.show()
    
#P3.3.2a
#Argon: (1.024 * 10 ** (-23), 1.582 * 10 ** (-26))
def atomic_potential(A, B):
    r = pylab.linspace(0.2, 0.6, 1001)
    pylab.xlabel('Distance (nm)')
    pylab.xlim(0.2, 0.6)
    
    U = ((B / pylab.array(r) ** 12) - (A / pylab.array(r) ** 6)) * 10 ** 18
    func1 = pylab.plot(r, U, 'b-', label = r'$U(r) = \frac{B}{r^{12}} - \frac{A}{r^6}$')
    pylab.ylabel('Energy (aJ)')
    pylab.ylim(-0.003, 0.004)
    pylab.legend()
    
    pylab.twinx()
    F = ((12* B / pylab.array(r) ** 13) - (6 * A / pylab.array(r) ** 7)) * 10 ** 18
    func2 = pylab.plot(r, F, 'r--', label = r'$F(r) = \frac{12B}{r^{13}} - \frac{6A}{r^7}$')
    pylab.ylabel('Force (aN)')
    pylab.ylim(-0.06, 0.08)
    
    funcs = func1 + func2
    labels = []
    for func in funcs:
        labels.append(func.get_label())
        
    pylab.legend(funcs, labels)
    pylab.show()
    
    return ('\u03b5: ' + '{:.5f}'.format(min(U[1:])) + ' nJ; ' + 'r_0: ' + 
            '{:.5f}'.format((2 * B / A) ** (1 / 6)))

#P3.3.2b
def atomic_harmonic(A, B):
    r = pylab.linspace(0, 1.0, 1001)
    pylab.xlabel('Distance (nm)')
    pylab.xlim(0.2, 0.6)
    
    U = ((B / pylab.array(r) ** 12) - (A / pylab.array(r) ** 6)) * 10 ** 18
    pylab.plot(r, U, 'b-', label = r'$U(r) = \frac{B}{r^{12}} - \frac{A}{r^6}$')
    pylab.ylabel('Energy (aJ)')
    pylab.ylim(-0.002, 0.001)
    
    r_0 = (2 * B / A) ** (1 / 6)
    k = ((156* B / r_0 ** 14) - (42 * A / r_0 ** 8)) * 10 ** 18
    V = 0.5 * k * (pylab.array(r) - r_0) ** 2 + min(U[1:])
    pylab.plot(r, V, 'r--', label = r'$V(r) = \frac{1}{2}k(r-r_0)+\epsilon$')
    
    pylab.legend()
    pylab.show()
    
#P3.3.3
def sunflower(n):
    phi = (1 + math.sqrt(5)) / 2
    theta = []
    r = []
    for n in range(1, n+1):
        theta.append(n * 2 * math.pi / phi)
        r.append(math.sqrt(n))
    pylab.polar(theta, r, 'o', linewidth = 5)
    pylab.show()
    
sunflower(500)