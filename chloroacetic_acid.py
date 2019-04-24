from experiment import Experiment
import pylab

#P4.6.4
acetic_acid = [''] * 5 
k = [''] * 5
for n in (40, 50, 60, 70, 80):
    acetic_acid[(n//10)-4] = Experiment('caa-T/caa-{}.txt'.format(n))
    acetic_acid[(n//10)-4].transform_inversey()
    k[(n//10)-4] = (acetic_acid[(n//10)-4].lin_reg()[0], n)
    pylab.plot(acetic_acid[(n//10)-4].x_vals, acetic_acid[(n//10)-4].y_vals)
    pylab.show()
    

file = open('caa-T/caa-k.txt', 'w')
for data in k:
    file.write('{:2}  {:.10f}\n'.format(data[1], data[0]))
    
arrhenius = Experiment('caa-T/caa-k.txt')
arrhenius.transform_lny()
arrhenius.transform_inversex()
coeff = arrhenius.lin_reg()[1]
R = 8.314

print(coeff * (-R))
