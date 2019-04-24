import pylab

#P4.6.4
class Experiment:
    
    def __init__(self, file):
        file = open('{}'.format(file), 'r')
        file = [line.split() for line in file.readlines()]
        self.x_vals, self.y_vals = [], []
        for line in file:
            try: 
                float(line[0])
            except ValueError:
                continue
            self.x_vals.append(float(line[0]))
            self.y_vals.append(float(line[1]))
            
    def multiply_x(self, a):
        self.x_vals = a * pylab.array(self.x_vals)
        
    def multiply_y(self, a):
        self.y_vals = a * pylab.array(self.y_vals)
    
    def transform_lnx(self):
        self.x_vals = pylab.log(self.x_vals)
    
    def transform_lny(self):
        self.y_vals = pylab.log(self.y_vals)
        
    def transform_inversex(self):
        self.x_vals = 1 / pylab.array(self.x_vals)
        
    def transform_inversey(self):
        self.y_vals = 1 / pylab.array(self.y_vals)
        
    def lin_reg(self):
        prod = pylab.array([x*y for x,y in zip(self.x_vals, self.y_vals)])
        sqd = pylab.array([x**2 for x in self.x_vals])
        self.x_vals = pylab.array(self.x_vals)
        self.y_vals = pylab.array(self.y_vals)
        m = ((pylab.mean(prod)-pylab.mean(self.x_vals)*pylab.mean(self.y_vals)) /
             (pylab.mean(sqd) - pylab.mean(self.x_vals)**2))
        c = pylab.mean(self.y_vals) - m*pylab.mean(self.x_vals)
        return (m, c)
    