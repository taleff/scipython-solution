import math
import sys

#P4.4.2
r_e = 6378.1

def h_sin(a):
    return (math.sin(a / 2)) ** 2
        
lat1, lon1, lat2, lon2 = float(sys.argv[1].split(',')[0]) * (math.pi / 180), \
                         float(sys.argv[1].split(',')[1]) * (math.pi / 180), \
                         float(sys.argv[2].split(',')[0]) * (math.pi / 180), \
                         float(sys.argv[2].split(',')[1]) * (math.pi / 180) 
                         
print(lat1, lon1, lat2, lon2)
                         
                         
d = 2 * r_e * math.asin(math.sqrt(h_sin(lat2 - lat1) + math.cos(lat1) * 
                        math.cos(lat2) * h_sin(lon2 - lon1)))
                        
print('{:.0f} km'.format(d))