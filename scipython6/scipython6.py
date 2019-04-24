import numpy as np
Polynomial = np.polynomial.Polynomial
import pylab
import math
import random
from scipy.io import wavfile

#P6.1.1
def whales(name, pop, mass):
    dt = np.dtype([('whale', '|S50'), ('population', 'u8'), ('mass', 'f8')])
    whales = np.array([('Bowhead whale', 9000, 60), 
                       ('Blue whale', 20000, 120), 
                       ('Fin whale', 100000, 70), 
                       ('Humpback whale', 80000, 30), 
                       ('Gray whale', 26000, 35), 
                       ('Atlantic white-sided dolphin', 250000, 0.235), 
                       ('Pacific white-sided dolphin', 1000000, 0.15), 
                       ('Killer whale', 100000, 4.5), 
                       ('Narwhal', 25000, 1.5), 
                       ('Beluga', 100000, 1.5), 
                       ('Sperm whale', 2000000, 50), 
                       ('Baiji', 13, 0.13), 
                       ('North Atlantic right whale', 300, 75), 
                       ('North Pacific right whale', 200, 80), 
                       ('Southern right whale', 7000, 70)], dtype = dt)
    whales.sort(order = 'mass')
    print(np.searchsorted(whales['mass'], mass))
    whales.sort(order = 'population')
    print(np.searchsorted(whales['population'], pop))
    return whales

#P6.1.2   
def shoelace(vertices):
    vertices = np.vstack((vertices, vertices[-1, :]))
    s_1 = vertices[:-1, 0].ravel().dot(vertices[1:, 1].ravel())
    s_2 = vertices[:-1, 1].ravel().dot(vertices[1:, 0].ravel())
    return 0.5 * abs(s_1 - s_2)
    
#P6.1.3
def gauss():
    x = np.linspace(-10, 10, 1000)
    def normal(x, sigma):
        coeff =  1 / (sigma*math.sqrt(2*math.pi))
        return coeff * np.exp(-(x)**2/(2*sigma**2))
    y1, y2, y3 = normal(x, 0.5), normal(x, 1.0), normal(x, 1.5)
    print('Sum:', y1.dot(np.ones(1000)) * 20 / 1000, '\n',
          'Sum:', y2.dot(np.ones(1000)) * 20 / 1000, '\n', 
          'Sum:', y3.dot(np.ones(1000)) * 20 / 1000)
    pylab.plot(x, y1, x, y2, x, y3)
    
    def derivative(x, h, sigma):
        return ((normal(x+h, sigma) - normal(x-h, sigma)) / (2*h))
        
    d1, d2, d3 = derivative(x, 0.01, 0.5), derivative(x, 0.01, 1.0), \
                 derivative(x, 0.01, 1.5)
    pylab.plot(x, d1, '--', x, d2, '--', x, d3, '--')
    pylab.show()

#P6.2.1
def mountains():
    def parse_date(s):
        if s == 45:
            return '99/99/9999'
        return s
    def decimal_conv(d, m, s):
        return d + m * (1/60) + s * (1/3600)
    dt = np.dtype([('name', '|S14'), ('height', 'i4'), ('first ascent', '|S11'),  
                   ('winter ascent', '|S11'), ('latitude', '|S10'), 
                   ('longitude', '|S10')])
    mountains = np.loadtxt('ex6-2-b-mountain-data.txt', converters = {3:parse_date},
                           dtype = dt, comments = '--', skiprows = 10)
    low = mountains['name'][np.argmin(mountains['height'])]
    lats = [decimal_conv(float(lat[:2]), float(lat[3:5]), float(lat[6:8])) if lat[-1] == 78 
            else -1 * decimal_conv(float(lat[:2]), float(lat[3:5]), float(lat[6:8]))
            for lat in mountains['latitude']]
    lons = [decimal_conv(float(lat[:2]), float(lat[3:5]), float(lat[6:8])) if lat[-1] == 69 
            else -1 * decimal_conv(float(lat[:2]), float(lat[3:5]), float(lat[6:8]))
            for lat in mountains['longitude']]
    north = mountains['name'][np.argmax(np.array(lats))]
    east = mountains['name'][np.argmax(np.array(lons))]
    west = mountains['name'][np.argmin(np.array(lons))]
    south = mountains['name'][np.argmin(np.array(lats))]
    print('{} is the lowest mountain in the list\n'.format(low),
          '{} is the most northerly\n'.format(north),
          '{} is the most easterly\n'.format(east),
          '{} is the most westerly\n'.format(west),
          '{} is the most southerly\n'.format(south))
    dates = [str(date).split('/') for date in mountains['first ascent']]
    dates = [float(date[0][2:]) * 0.0001 + float(date[1]) * 0.01  + float(date[2][:-1]) 
             for date in dates]
    early = mountains['name'][np.argmin(np.array(dates))]
    print('{} was the earliest climbed moutain'.format(early))
    
    dates = [str(date).split('/') for date in mountains['winter ascent']]
    dates = [float(date[0][2:]) * 0.0001 + float(date[1]) * 0.01  + float(date[2][:-1]) 
             if date[0][2:] != "-'" else 9999 for date in dates]
    early = mountains['name'][np.argmin(np.array(dates))]
    print('{} was the earliest climbed moutain'.format(early))
    
    mountain_2 = np.hstack((mountains['name'].reshape(14,1), 
                     (mountains['height'] * 3.2808399).reshape(14,1), 
                     mountains['first ascent'].reshape(14,1)))

#P6.2.2
def airports(air_1, air_2):
    def haversine(lat1, lon1, lat2, lon2):
        r_e = 6378.1
        def h_sin(a):
            return (math.sin(a / 2)) ** 2
        return 2 * r_e * math.asin(math.sqrt(h_sin(lat2 - lat1) + 
               math.cos(lat1) * math.cos(lat2) * h_sin(lon2 - lon1)))
    dt = np.dtype([('code', '|S3'), ('lat', 'f8'), ('lon', 'f8')])
    airports = np.loadtxt('busiest_airports.txt', usecols = (0, -2, -1), 
                          dtype = dt)
    locs = np.array([air_1, air_2], dtype = '|S3')
    air_1, air_2 = airports['code'] == locs[0], \
                   airports['code'] == locs[1]
    dis = haversine(airports['lat'][air_1], airports['lon'][air_1],
                    airports['lat'][air_2], airports['lon'][air_2])
    return dis
    
#P6.2.3
def immunization():
    def parse_data(s):
        try:
            return int(s)
        except ValueError:
            return -2
    file = open('wb-data.dat', 'r').readline().split(';')
    titles = [element[:4] if element[0].isdigit() else element 
              for element in file]
    years = [int(element[:4]) for element in file if element[0].isdigit()]
    data = ['f4' if element[0].isdigit() else '|S30' for element in titles]
    dt = np.dtype({'names':titles, 'formats':data})
    immunization = np.genfromtxt('wb-data.dat', dtype = dt, skip_header = 1,
                                  skip_footer = 5, delimiter = ';', 
                                  filling_values = np.nan)

    for vaccine in ('Pol3', 'BCG', 'measles'):
        pylab.title('{} Vaccinations'.format(vaccine))
        for i in range(len(immunization)):
            if vaccine in str(immunization[i]['Series_Name']):
                y = [immunization[i]['{}'.format(year)] for year in years]
                pylab.plot(years, y, label = '{}'.format(str(immunization[i]['Country_Name'])[2:-1]))
        pylab.legend(loc = 'lower right')
        pylab.xlabel('Year')
        pylab.ylabel('Percent of One Year Olds Vaccinated')
        pylab.show()

#P6.3.1
def lotto():
    data = np.loadtxt('lottery-draws.txt', skiprows = 2, dtype = 'i8')
    comb = np.hstack((np.array([True]), (data[:, -1] > 0)[:-1]))
    comb = comb & (data[:, -1] > 0)
    numbs = (data[:, 0] < 13).astype('i8')
    for i in range(1, 6):
        numbs += (data[:, i] < 13).astype('i8')
    correlation = np.vstack((numbs[comb], data[:, -1][comb]))
    pylab.scatter(numbs[comb], data[:, -1][comb], marker = '.')
    pylab.show()
    return np.corrcoef(correlation)

#P6.3.2
def histogram():
    data = [int(random.normalvariate(50, 15)) for i in range(1000000)]
    hist, bins = np.histogram(data, bins=100, range=(0, 100), density = True)
    pylab.bar(bins[:-1], hist, np.array(1))
    pylab.show

#P6.3.3
def height():
    female = np.loadtxt('ex6-3-f-female-heights.txt')
    male = np.loadtxt('ex6-3-f-male-heights.txt')
    female, male = np.hstack((female[0], female[1], female[2], female[3], female[4])), \
                   np.hstack((male[0], male[1], male[2], male[3], male[4]))
    male_mean, male_std = np.mean(male), np.std(male)
    female_mean, female_std = np.mean(female), np.std(female)
    pylab.hist(male, bins = 10)
    pylab.xlabel('Height (cm)')
    pylab.show()
    pylab.hist(female, bins = 10, color = 'r')
    pylab.xlabel('Height (cm)')
    pylab.show()

#P6.4.1
def explosion():
    TNT_KILOTON = 4.184 * (10**9)
    P_AIR = 1.25
    data = np.loadtxt('new-mexico-blast-data.txt', skiprows = 1)
    times, radius = data[:, 0], data[:, 1]
    log_times, log_radius = np.log10(times), np.log10(radius)
    
    cmin, cmax = min(log_times), max(log_times)
    p = Polynomial.fit(log_times, log_radius, 1, window = (cmin, cmax), 
                       domain = (cmin, cmax))
    pylab.plot(log_times, log_radius, 'ob')
    pylab.plot(log_times, p(log_times), '-')
    pylab.show()
    E = (10**p(0.0) * P_AIR**0.2)**5
    print('The energy of the blast was {:.03} J'.format(E), 
          'or {:.03} kilotons of TNT'.format(E/TNT_KILOTON), sep='\n')
    
#P6.4.2
def x_and_y():
    data = np.loadtxt('ex6-4-a-anscombe.txt')
    for i in range(0, 8, 2):
        mean_x, variance_x = np.mean(data[:, i]), np.std(data[:, i])
        mean_y, variance_y = np.mean(data[:, i+1]), np.std(data[:, i+1])
        cmin, cmax = min(data[:, i]), max(data[:, i+1])
        p = Polynomial.fit(data[:, i], data[:, i+1], 1, window=(cmin, cmax), 
                           domain = (cmin, cmax))
        correlation = np.corrcoef(np.vstack((data[:, i], data[:, i+1])))
        A0, m = p
        pylab.plot(data[:, i], data[:, i+1], 'o')
        pylab.plot(data[:, i], p(data[:, i]))
        pylab.show()
        print('Mean of x: {}, Mean of y: {}'.format(mean_x, mean_y), 
              'Variance of x: {}, Variance of y: {}'.format(variance_x, variance_y), 
              'Correlation: {}'.format(correlation[1,0]),
              'Best Fit: {}x + {}'.format(m, A0), sep='\n')

#P6.4.3
def van_der_Waals():
    A = 4.225 # L2barmol-2
    A2 = 4.225 * 0.986923 #L2atmmol-2
    B = 0.03707 #Lmol-1
    R = 8.314 #JK-1mol-1
    R1 = 0.0821 #Latmmol-1K-1
    T_c, p_c, = 8*A*100 / (27*R*B), A*100000 / (27*(B**2))
    def molar_vol(t, p):
        gas = Polynomial([-A*B, A, -(p*B + R1*t), p])
        return gas
    print('The critical point of ammonia is at {:.3} K and {:.3} Pa'.format(T_c, p_c))
    print('The molar volume of ammonia at 298 K and 1 atm is {:.3} L/mol'.format(max(molar_vol(298, 1).roots())))
    print('The molar volume of ammonia at 500 K and 12 MPa is {:.3} L/mol'.format(float(max(molar_vol(500, 120*0.986923).roots()))))
    
    def isotherm(t, V):
        return (R1*t / (V-B)) - (A2 / (V**2))
        
    def ideal_isotherm(t, V):
        return R1*t / V
        
    V = np.linspace(0.1, 50, 100)
    p = isotherm(350, V)
    p_ideal = ideal_isotherm(350, V)
    
    pylab.plot(np.log10(p), np.log10(V), label = 'van der Waals')
    pylab.plot(np.log10(p_ideal), np.log10(V), label = 'Ideal')
    pylab.legend()
    pylab.xlabel('Log Pressure (atm)')
    pylab.ylabel('Log Molar Volume (L/mol)')
    pylab.show()
    
#P6.4.4
def saturn_V():
    accel = Polynomial([2.198, 2.842 * (10**(-2)), 1.061 * (10**(-3))])
    vel = accel.integ(k = 0)
    pos = vel.integ(k = 0)
    print('The rocket is {:.3} m high after 2m 15.2s'.format(pos(120+15.2)))
    
    gamma = 1.4
    R = 8.314
    M = 0.0288
    T = (pos * (-0.006)) + Polynomial([302])
    mach = (vel * vel) - (T * (gamma * R / M))
    time = mach.roots()[np.isreal(mach.roots())]
    time = float((time[time > 0])[0])
    print('The rocket reached Mach 1 at {:.3} m, {:.3} s after launch'.format(pos(time), time))
    
#Q6.5.2
def ticker():
    x = [1.3, 6.0, 20.2, 43.9, 77.0, 119.6, 171.7, 233.2, 304.2, 394.7,
         474.7, 574.1, 683.0, 801.3, 929.2, 1066.4, 1213.2, 1369.4, 1535.1,
         1710.3, 1894.9]
    x = np.array(x) / 100
    p = Polynomial.fit(np.linspace(0.1, len(x) * 0.1, len(x)), x, 2)
    g = p.deriv(2)(1)
    print('The acceleration of gravity is {:.3} m*s^-2'.format(g))
    
#P6.5.1
def planck():
    #L, M, T, Q, Temp are the columns
    #c, G, h, C, k_b are the rows
    units = np.array([( 1, 0,-1, 0, 0), 
                      ( 3,-1,-2, 0, 0), 
                      ( 2, 1,-1, 0, 0), 
                      ( 3, 1,-2,-2, 0), 
                      ( 2, 1,-2, 0,-1)])
    c = 2.99792458 * 10**8
    G = 6.67384 * 10**(-11)
    h_bar = 1.054571726 * 10**(-34)
    C = 8.9875517873681764 * 10**9
    k_B = 1.3806488 * 10**(-23)
    
    inv_units = np.linalg.inv(units)
    
    unit_names = (('length', 'm'), ('mass', 'kg'), ('time', 's'), 
                  ('charge', 'coulombs'), ('thermodynamic temperature', 'K'))
    for i in range(5):
        planck_unit = (c**inv_units[i, 0]) * (G**inv_units[i, 1]) * \
                      (h_bar**inv_units[i, 2]) * (C**inv_units[i, 3]) * \
                      (k_B**inv_units[i, 4])
        print('The planck unit for {name} is {val:.5e} {unit}'.format(name = 
              unit_names[i][0], val = planck_unit, unit = unit_names[i][1]))
    
#P6.5.2
def inertia(file):
    data = np.loadtxt('{}'.format(file))
    
    m = data[:, 0]
    pos = data[:, 1:4]
    
    c = (m*pos[:, 0]/sum(m), m*pos[:, 1]/sum(m), m*pos[:, 2]/sum(m))
    
    def inertia_tensor(m, pos, c):
        I = np.eye(3,3)
        for i in range(3):
            pos1, pos2 = (i+1) % 3, (i+2) % 3
            I[i, i] = sum(m * ((pos[:, pos1]-c[pos1])**2 + (pos[:, pos2]-c[pos2])**2))
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                I[i,j] = sum(m * (pos[:, i] - c[i]) * (pos[:, j] - c[j]))
        return I
        
    inertia = inertia_tensor(m, pos, c)
    vals, vecs = np.linalg.eig(inertia)
    
    vals.sort()
    if np.isclose(vals[0], vals[1]):
        if np.isclose(vals[1], vals[2]):
            classification = 'spherical top'
        else:
            classification = 'oblate symmetric top'
    elif np.isclose(vals[1], vals[2]):
        classification = 'prolate symmetric top'
    else:
        classification= 'asymmetric top'
    
    c = 2.99792458 * 10**8
    h_bar = 1.054571726 * 10**(-34)
    Q = 2*np.pi*h_bar / (8*np.pi*np.pi*c*vals)
    
    print('The principal moments of inertia are {}'.format(vals), \
          'The classification of this molecule is {}'.format(classification), \
          'The rotational constants are {}'.format(Q), sep='\n')
        
#P6.5.3
def decomp(matrix):
    u, sigma, v = np.linalg.svd(matrix)
    val_1, mat_1 = np.linalg.eig(matrix.dot(matrix.T))
    val_2, mat_2 = np.linalg.eig((matrix.T).dot(matrix))
    
    for i in range(3):
        u[i, :].sort()
        mat_1[i, :].sort()
        v[:, i].sort()
        mat_2[i, :].sort()
        for j in range(3):
            if not np.isclose(u[i, j], mat_1[i, j]):
                mat_1[i, :] = -mat_1[i, :]
                mat_1[i, :].sort()
                if not np.isclose(u[i, j], mat_1[i, j]):
                    return False
            if not np.isclose(v[j, i], mat_2[i, j]):
                mat_2[i, :] = -mat_2[i, :]
                mat_2[i, :].sort()
                if not np.isclose(v[j, i], mat_2[i, j]):
                    return False
    
    roots = np.hstack((np.sqrt(val_1), np.sqrt(val_2)))
    for n in roots:
        for r in sigma:
            if np.isclose(r, n):
                break
        else:
            return False
            
    return True
           
#P6.6.1
def fives(n):
    if n == 0:
        return np.matrix([[0], [1]])
    matrix = fives(n-1)
    return 9*matrix + np.matrix([[10**(n-1)], [0]])
    
def fives_check(numb):
    count = 0
    for n in range(numb):
        if (n % 10 == 5) or ((n//10) % 10 == 5) or ((n//100) % 10 == 5) or ((n//1000) % 10 == 5):
            count += 1
    return count
    
#6.6.2
def fibonacci(n):
    F = np.matrix([[1, 1], [1, 0]])
    vals, vecs = np.linalg.eig(F)
    C = np.vstack(vecs)
    D = np.linalg.inv(C)*F*C
    Dn = np.matrix([[D[0, 0]**(n-1),0], [0,D[1,1]**(n-1)]])
    Fn = C*Dn*np.linalg.inv(C)
    return Fn[0,0]

#6.6.3
def conics(A=0, B=0, C=0, D=0, E=0, F=0):
    Q = np.matrix([[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]])
    Q_det = np.linalg.det(Q)
    Q_sub_det = np.linalg.det(Q[0:2, 0:2])
    if np.isclose(0, Q_det):
        if np.isclose(0, Q_sub_det):
            return 'parallel lines'
        elif Q_sub_det < 0:
            return 'intersecting lines'
        else:
            return 'single point'
    elif np.isclose(Q_sub_det,0):
        return 'parabola'
    elif Q_sub_det < 0:
        return 'hyperbola'
    else:
        if np.isclose(A, C) and np.isclose(B, 0):
            return 'circle'
        else:
            return 'ellipse'

#P6.7.1
def experiments(ntrials, n):
    outcomes = np.array(['tails', 'heads'])
    trials = outcomes[np.random.randint(len(outcomes), size = (n, ntrials))]
    heads = [0]*n
    for i in range(n):
        heads[i] = sum(trials[i, :] == 'heads')
    bin_trials = np.random.binomial(100, 0.5, n)
    pylab.hist(heads, bins = 40)
    pylab.show()
    pylab.hist(bin_trials, bins = 40)
    pylab.show()
    
#P6.7.2a
def buffon(ntrials):
    starting_pos = np.random.random_sample(ntrials)
    thetas = np.random.random_sample(ntrials) * 2 * np.pi
    end_pos = starting_pos + np.cos(thetas)
    crossed = sum(end_pos > 1) + sum(end_pos < 0)
    return 2*ntrials/crossed

#P6.7.2b
def buffon_two(ntrials, a=0, d=1):
    starting_pos =  np.random.random_sample((2, ntrials)) * d
    crossed = sum(((starting_pos[0, :] + a) > d) | 
                  ((starting_pos[0, :] - a) < 0) | 
                  ((starting_pos[1, :] + a) > d) | 
                  ((starting_pos[1, :] - a) < 0))
    expected = 1 - (d-2*a)**2 / (d**2)
    return crossed/ntrials, expected

#P6.7.3
class Bacterium():
    
    def __init__(self, speed, tumble_size, 
                 toward_tumble_frequency, away_tumble_frequency, 
                 x=np.random.random_sample(), y=np.random.random_sample(),
                 theta=np.random.random_sample()*2*np.pi):
        #x,y are the starting positions of the bacterium
        #speed is the distance the bacterium moves in one flagella turn
        #theta is the angle the bacterium is pointing
        self.x_pos = x
        self.y_pos = y
        self.speed = speed
        self.theta = theta
        self.tumble_size = tumble_size
        self.toward_tumble_frequency = toward_tumble_frequency
        self.away_tumble_frequency = away_tumble_frequency
        
    def sense_nutrient(self, grad):
        #returns the value of the concentration gradient
        return grad(self.x_pos,self.y_pos)
    
    def move_forward(self):
        #moves the bacteria forward based on its direction and speed
        self.x_pos = self.x_pos + np.cos(self.theta)*self.speed
        self.y_pos = self.y_pos + np.sin(self.theta)*self.speed
        return (self.x_pos, self.y_pos)
        
    def tumble(self):
        #randomly changes direction of the bacterium
        self.theta = self.theta + np.random.normal()*self.tumble_size
        return self.theta
        
    def find_nutrient(self, grad, steps):
        #method that combines previous moves to find a nutrient
        pos1 = self.sense_nutrient(grad)
        prob = [self.toward_tumble_frequency, 1-self.toward_tumble_frequency]
        for n in range(steps):
            if np.random.choice([True, False], p=prob):
                self.tumble()
                continue
            else:
                yield self.move_forward()
                pos2 = self.sense_nutrient(grad)
                state = pos2 > pos1
                pos1, pos2 = pos2, pos1
                
            if state:
                prob = [self.toward_tumble_frequency, 1-self.toward_tumble_frequency]
            else:
                prob = [self.away_tumble_frequency, 1-self.away_tumble_frequency]
            
            
        

#P6.7.3
def bacteria():
    nutrient = (0.5, 0.5)
    grad = lambda x,y: -(x-nutrient[0])**2-(y-nutrient[1])**2

    b = Bacterium(0.01, math.pi/3, 0.3, 0.7, x=1, y=1)
    pos = np.array([x for x in b.find_nutrient(grad, 5000)])
    pylab.plot(pos[:, 0], pos[:, 1])
    pylab.show()

#P6.7.4    
class River:
    
    def __init__(self, phi, sigma, speed, start = (0,0)):
        self.phi = phi * (np.pi/180)
        self.sigma = sigma * (np.pi/180)
        self.start = start
        self.speed = speed
        self.path = []
        
    def walk(self, steps):
        self.path = np.zeros((steps, 2))
        self.path[0, 0], self.path[0, 1] = self.start[0], self.start[1]
        for n in range(1, steps):
            self.phi += np.random.normal(scale = self.sigma)
            self.path[n,0] = self.path[n-1, 0] + np.cos(self.phi)*self.speed
            self.path[n,1] = self.path[n-1, 1] + np.sin(self.phi)*self.speed
        
#P6.7.4
def meanders(trials):
    average = np.zeros((40, 2))
    count = 0
    for n in range(trials):
        r = River(110, 17, 1)
        r.walk(40)
        if (r.path[-1, 0]-10)**2 + (r.path[-1, 1])**2< 1:
            average = average + r.path
            count += 1
            pylab.plot(r.path[:, 0], r.path[:, 1], 'b--', lw = 1)
            
    average = average / count
    pylab.plot(average[:, 0], average[:, 1], 'g-', lw = 2)
    pylab.savefig('rivers.png')

#P6.8.1
def signal(end):
    nu = 250
    tau = 0.2
    end = int(end*1000)
    t = np.linspace(0, 1, 1000)[: end]
    f = np.cos(2*np.pi*nu*t) * np.exp(-t/tau)
    F = np.fft.fft(f)
    pylab.plot(F.real, 'k', label='real')
    pylab.plot(F.imag, 'gray', label='imag')
    pylab.show()

#P6.8.2
def square():
    t = np.linspace(0, 2, 2048)
    f = np.array([1 if n%1 < 0.5 else -1 for n in t])
    
    def expansion(terms, t):
        expansion = np.zeros(len(t)) 
        for k in range(1, terms+1):
            expansion += (1/(2*k-1))*np.sin(2*np.pi*(2*k-1)*t)
        return (4/np.pi)*expansion
    
    pylab.plot(t, f, 'b')
    pylab.plot(t, expansion(3, t), 'g--')
    pylab.plot(t, expansion(9, t), 'g--')
    pylab.plot(t, expansion(18, t), 'g--')
    pylab.show()
        
    F = np.fft.fft(f)
    freq = np.fft.fftfreq(len(t), 1/1024)
    spec = 2/len(t) * np.abs(F[:len(t)//2])
    pylab.plot(freq[:len(t)//2], spec, 'k')
    pylab.plot(freq[:len(t)//2], 2/len(t) * np.abs(np.fft.fft(expansion(3, t))[:len(t)//2]), 'g--')
    pylab.plot(freq[:len(t)//2], 2/len(t) * np.abs(np.fft.fft(expansion(9, t))[:len(t)//2]), 'g--')
    pylab.plot(freq[:len(t)//2], 2/len(t) * np.abs(np.fft.fft(expansion(18, t))[:len(t)//2]), 'g--')
    pylab.xlabel('Frequency /Hz')
    pylab.xlim(0, 50)
    pylab.show()
    
    
#P6.8.3
def sound():
    pass
