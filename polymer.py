#P4.6.2
import math
import random

class Polymer:
    
    def __init__(self, N, a):
        self.N, self.a = N, a
        self.xyz = [(None, None, None)] * N
        self.R = None
        self.make_polymer()
        
    def make_polymer(self):
        self.xyz[0] = x, y, z = cx, cy, cz, = 0., 0., 0.
        for i in range(1, self.N):
            theta = math.acos(2 * random.random() - 1)
            phi = random.random() * 2.* math.pi
            x += self.a * math.sin(theta) * math.cos(phi)
            y += self.a * math.sin(theta) * math.sin(phi)
            z += self.a * math.cos(theta)
            self.xyz[i] = x, y, z
            cx, cy, cz = cx + x, cy + y, cz + z
        cx, cy, cz = cx / self.N, cy / self.N, cz / self.N
        self.R = x, y, z
        
        for i in range(self.N):
            self.xyz[i] = self.xyz[i][0] - cx, self.xyz[i][1] - cy, self.xyz[i][2] - cz
            
    def calc_Rg(self):
        self.Rg = 0.
        for x,y,z in self.xyz:
            self.Rg += x**2 + y**2 + z**2
        self.Rg = math.sqrt(self.Rg / self.N)
        return self.Rg
        
    def save_svg(self):
        file = open('polymer_image.svg', 'w')
        file.write('<?xml version="1.0" encoding="utf-8"?>')
        file.write('                 <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="500" height="500" style="background: #ffffff">')
        x1, y1 = 250, 250
        for x, y, z in self.xyz:
            cx = 250 + (math.cos(math.pi/6) * x - math.cos(math.pi/6) * z) * 10
            cy = 250 + (y - 0.5 * x - 0.5 * z) * 10
            file.write('<circle cx="{}" cy="{}" r="4" style="stroke: black; fill: red;"/>'.format(cx, cy))
        for x, y, z in self.xyz[1:]:
            x2 = 250 + (math.cos(math.pi/6) * x - math.cos(math.pi/6) * z) * 10
            y2 = 250 + (y - 0.5 * x - 0.5 * z) * 10
            file.write('<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="black" stroke-width="3"/>'.format(x1, x2, y1, y2))
            file.write('<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="red" stroke-width="3" stroke-linecap = "round"/>'.format(x1, x2, y1, y2))
            x1 = 250 + (math.cos(math.pi/6) * x - math.cos(math.pi/6) * z) * 10
            y1 = 250 + (y - 0.5 * x - 0.5 * z) * 10
        file.write('</svg>')
            
polymer = Polymer(1000, 1)
polymer.save_svg()