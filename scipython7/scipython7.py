import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import numpy as np
from matplotlib.patches import Ellipse

#P7.7.1
def big_mac():
    countries = ('argentina', 'australia', 'china', 'uk')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    dt = [('month', 'i4'), ('year', 'i8'), ('price', 'f8')]
    usa = np.loadtxt('us-bigmac.txt', dtype=dt, skiprows = 1)
    year = usa['month']*(1/12) + usa['year']
    for country in countries:
        data = np.loadtxt('{}-bigmac.txt'.format(country), skiprows = 1)
        valuation = (((data[:, 2]/data[:, 3])-usa['price'])/usa['price']) * 100
        ax.plot(year, valuation, label = country.upper())
    ax.axhline(0, color = 'k')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent Over/Undervaluation')
    ax.set_title('Big Max Index', fontsize = 16)
    ax.set_ylim(-100, 100)
    ax.set_xlim(min(year), max(year))
    ax.legend(loc = 'upper right', fontsize = 10)
    plt.show()
    
#P7.1.2
def west_nile_virus():
    years = np.linspace(1999, 2008, 10)
    neuro = np.array([59, 19, 64, 2946, 2866, 1148, 1309, 1495, 1227, 689])
    non_neuro = np.array([3, 2, 2, 1210, 6996, 1391, 1691, 2774, 117, 667])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.bar(left = years, height = neuro, label = 'Neuroinvasive', hatch = 'xxxx', 
           align = 'center', color = 'gray')
    ax.bar(years, height = non_neuro, bottom = neuro, label = 'Non-neuroinvasive', 
           align = 'center', color = 'gray')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Cases')
    ax.set_title('West Nile Virus Cases in the United States')
    ax.legend(loc = 'upper right')
    ax.set_xlim(1998, 2009)
    plt.show()
    
#P7.1.3
def population():
    dt = np.dtype([('country', 'S30'), ('value', 'f8')])
    
    values = {'gdp':0, 'bmi_men':0, 'population_total':0, 'continent':0}
    
    for file in ('gdp', 'bmi_men', 'population_total'):
        data = np.genfromtxt('{}.tsv'.format(file), delimiter = '\t', dtype = dt,
                             skip_footer = 1)
        values[file] = dict(data[(data['value'] == data['value'])])
        
    values['continent'] = dict(np.genfromtxt('continents.tsv', delimiter = '\t', 
                                        dtype = np.dtype([('country', 'S30'), ('continent', 'S20')]),
                                        skip_footer = 1))
        
    fig = plt.figure(figsize = (11.5, 8))
    ax = fig.add_subplot(111)
    
    colors = {b'Europe':'#ff0000', b'Africa':'#00ff00', b'North America':'#0000ff', 
              b'Oceania':'#000000', b'Asia':'#ff00ff', b'South America':'#00ffff'}
    
    legend = []
    for country in values['continent']:
        if (values['gdp'].get(country, -1) != -1) and \
           (values['bmi_men'].get(country, -1) != -1) and \
           (values['population_total'].get(country, -1) != -1):
            point = ax.scatter(values['gdp'][country]/10**3, values['bmi_men'][country],
                               s = (np.log10(values['population_total'][country]) - 3)**2,
                               color = colors[values['continent'][country]], marker = 'o')
            if not (str(values['continent'][country])[2:-1] in [i[1] for i in legend]):
                legend.append((point, str(values['continent'][country])[2:-1]))
    legend = np.array(legend)
    
    ax.set_xlabel('GDP per capita in thousands of dollars')
    ax.set_ylabel('Average Male BMI')               
    ax.set_title('Comparison of National BMI and GDP', fontsize = 18)
    ax.legend(legend[:, 0], legend[:, 1], loc = 'lower right')
    plt.savefig('countries.jpeg')
        
#P7.1.4
def carbon_dioxide():
    data = np.loadtxt('co2_mm_mlo.txt', usecols = (2,4,5))
    
    fig = plt.figure(figsize = (11.5, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(data[:, 0], data[:, 1], label = 'interpolated', lw = 0.5)
    ax.plot(data[:, 0], data[:, 2], label = 'trend', lw = 2)
    
    ax.set_xlabel('Year', fontsize = 16)
    ax.set_ylabel('Concentration of CO2 (ppm)', fontsize = 16)
    ax.minorticks_on()
    ax.grid()
    ax.legend(loc = 'upper left', fontsize = 12)
    plt.show()
    
#P7.1.5
def blackbody():
    def planck(lam, T):
        c = 2.99792458 * 10**8
        h = 1.054571726 * 10**(-34) * 2 * np.pi
        k_B = 1.3806488 * 10**(-23)
        
        lam = lam * 10**(-9)        
        
        coeff = 2 * h * (c**2) / (lam**5)
        denom = np.exp(h*c/(lam*k_B*T)) - 1
        
        return coeff * (1/denom)
    
    lam = np.linspace(10, 5000, 4991)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = {5778:'#0080FF', 5000:'#00FFFF', 4000:'#FFFF00', 3000:'#FF8000'}    
    
    for x in (5778, 5000, 4000, 3000):
        f = planck(lam, x)
        ax.plot(lam, f, label = '{} K'.format(x), color = colors[x])
    
    ax.set_xlim(0, 5000)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Spectral Radiance')
    ax.legend()    
    
    plt.show()
    
#P7.1.6
def circles(numb, r):
    theta = np.linspace(0, 2*np.pi, numb)
    
    x, y = np.cos(theta)*r + 0.5, np.sin(theta)*r + 0.5
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect = 'equal')
    
    for n in range(numb):
        circle = Circle((x[n],y[n]), r, color = 'gray', alpha = 0.4)
        ax.add_artist(circle)
        
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    plt.show()
    
#Q7.2.1
def sinc():
    x = np.linspace(-5, 5, 1000)
    y = x.copy()
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    cp = ax.contour(x[0, :], y[:, 0], np.sin(r)/r, colors = 'k')
    ax.contourf(x[0, :], y[:, 0], np.sin(r)/r, 100)
    ax.clabel(cp)
    
    plt.show()
    
#Q7.2.2
def birthday(month, day):
    days = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 
            11:30, 12:31}
    
    data = np.genfromtxt('birthday-data.csv', skip_header = 1, delimiter = ',', 
                         dtype = [('month', 'i4'), ('day', 'i4'), ('value', 'f8')])
    
    for x in range(len(data['value'])):
        if data['day'][x] > days[data['month'][x]]:
            data['value'][x] = np.nan
    
    start = 0
    rows= []    
    for x in range(12):
        rows.append(data['value'][start: start + 31]/1000)
        start += 31
    image = np.vstack(rows)
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    im = ax.imshow(image, interpolation = 'nearest')
    ax.set_yticks(range(12))
    ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 
                    'Sept', 'Oct', 'Nov', 'Dec'])
    ax.set_xticks(range(0, 31, 2))
    ax.set_xticklabels(range(1, 33, 2))
    
    ax.set_xlabel('Day of the Month')
    ax.set_title('Number of children born on a day')
    
    cbar = fig.colorbar(ax = ax, mappable = im, orientation = 'horizontal')
    cbar.set_label('Number of Births in thousands')
    
    months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'June':6, 'July':7, 
              'Aug':8, 'Sept':9, 'Oct':10, 'Nov':11, 'Dec':12}    
              
    try:
        int(month)
        if not (1 <= month <= 12):
            return 'Invalid Date'
    except ValueError:
        try:
            month = months[month]
        except KeyError:
            return 'Invalid Date'
    
    if not (1 <= day <= days[month]):
        return 'Invalid Date'
        
    total_births = sum([x for x in data['value'] if x == x])
    probability = ((data['value'][(month-1)*(31) + day - 1])/ total_births) * 100
    
    return 'Probability: {:.5f} percent'.format(probability)

#P7.2.1
def chaos(sides, r, iterations, x0, y0):
    if sides <= 2:
        return 'Not a polygon'
    
    def polygon_creator(sides, dist):
        theta = np.linspace(0, 2 * np.pi, sides + 1)[:-1]
        points = []
        for angle in theta:
            points.append((np.cos(angle)*dist, np.sin(angle)*dist))
        return points
    
    points = np.array(polygon_creator(sides, 1))
    plot = np.zeros((iterations + 1, 2))
    plot[0, :] = (x0, y0)
    
    for n in range(iterations):
        point = np.random.choice(sides)
        new_plot = (r * (plot[n, :]+points[point, :]))
        plot[n+1, :] = new_plot
        
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, aspect = 'equal')
    ax.scatter(plot[:, 0], plot[:, 1], s = 0.5)
    
    plt.show()
    
#P7.2.2    
def bmi():
    FEMALE, MALE = 0, 1
    dt = np.dtype([('mass', 'f8'), ('height', 'f8'), ('gender', 'i2')])
    data = np.loadtxt('body.dat.txt', usecols=(22,23,24), dtype=dt)
    mass = np.linspace(30, 120, 1000)
    height = np.linspace(1.4, 2.1, 1000)
    mass, height = np.meshgrid(mass, height)
    bmi = mass / (height**2)

    fig, ax = plt.subplots()

    def get_cov_ellipse(cov, centre, nstd, **kwargs):
        """
        Return a matplotlib Ellipse patch representing the covariance matrix
        cov centred at centre and scaled by the factor nstd.

        """

        # Find and sort eigenvalues and eigenvectors into descending order
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by 
        vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * nstd * np.sqrt(eigvals)
        return Ellipse(xy=centre, width=width, height=height,
                       angle=np.degrees(theta), **kwargs)

    labels, colours =['Female', 'Male'], ['magenta', 'blue']
    for gender in (FEMALE, MALE):
        sdata = data[data['gender']==gender]
        height_mean = np.mean(sdata['height'])
        mass_mean = np.mean(sdata['mass'])
        cov = np.cov(sdata['mass'], sdata['height'])
        ax.scatter(sdata['height'], sdata['mass'], s = 1, color=colours[gender],
                   label=labels[gender])
        e = get_cov_ellipse(cov, (height_mean, mass_mean), 3,
                            fc=colours[gender], alpha=0.2)
        ax.add_artist(e)
        
    cp = ax.contour(height*100, mass, bmi, levels = [18.5, 25, 30], colors = 'k')
    ax.clabel(cp, fmt = {18.5: 'Underweight', 25: 'Overweight', 30: 'Obese'},
              fontsize = 8)
    ax.set_xlabel('Height /cm')
    ax.set_ylabel('Mass /kg')
    ax.legend(loc='upper left', scatterpoints=1)
    plt.show()
    
#P7.2.3
def advection():
    pass
    

#P7.2.4
def julia_set(c = complex(-0.1, 0.65), nmax = 500, zmax = 10, res = 1000):
    re = np.linspace(-1.5, 1.5, res)
    im = np.zeros(res) * complex(0, 0)
    im.imag = np.linspace(-1.5, 1.5, res)
    
    re, im = np.meshgrid(re, im)
    grid = re + im
    
    image = np.zeros((res, res))

    for n in range(nmax + 1):
        comb = (abs(grid) < zmax)[:, :]
        image[comb] += np.ones((res, res))[comb]
        grid[comb] = grid[comb]**2 + c
        
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.xaxis.set_visible(False)    
    ax.yaxis.set_visible(False)
        
#P7.2.5
def island():
    square = np.load('gb-alt.npy')
    square[square == 0] = np.nan

    fig = plt.figure()
    images = ['']*4
    images[0] = fig.add_subplot(141)
    images[0].imshow(square)
    images[0].xaxis.set_visible(False)
    images[0].yaxis.set_visible(False)
    image = 2
    for sea_level_rise in (10, 50, 200):
        images[image-1] = fig.add_subplot(140 + image)
        square[square<sea_level_rise] = np.nan
        images[image-1].imshow(square)
        images[image-1].xaxis.set_visible(False)
        images[image-1].yaxis.set_visible(False)
        image += 1        
        