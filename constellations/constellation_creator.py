from star import Star
import math

#P4.6.3
#Function converts hrs, min, sec to radians
def convert(deg, mins, secs):
    degs = deg + mins/60 + secs/3600
    return degs * (math.pi/180)


file = open('bsc5.dat', 'r')
stars = file.readlines()

star_data = []

#Creating a star object for every star in bsc5.dat
for i, star_obj in enumerate(stars):
    #Skip star instantiation if coordinates to not exist        
    if star_obj[69:71] == '  ':
        continue
    
    r_a = convert(int(star_obj[75:77])*15, int(star_obj[77:79])*15, 
                      float(star_obj[79:83])*15)
    d = convert(int(star_obj[84:86]), int(star_obj[86:88]), 
                    float(star_obj[88:90]))
    #Check to see if coordinate is negative
    if star_obj[83] == '-':
        d *= (-1)
        
    star_data.append(Star(star_obj[4:14], star_obj[14:25], 
                          float(star_obj[102:107].strip(' ')), (r_a, d)))


constellation_list = ['And', 'Ant', 'Aps', 'Aql', 'Aqr', 'Ara', 'Ari', 'Aur', 
                      'Boo', 'CMa', 'CMi', 'CVn', 'Cae', 'Cam', 'Cap', 'Car', 
                      'Cas', 'Cen', 'Cep', 'Cet', 'Cha', 'Cir', 'Cnc', 'Col', 
                      'Com', 'CrA', 'CrB', 'Crt', 'Cru', 'Crv', 'Cyg', 'Del', 
                      'Dor', 'Dra', 'Equ', 'Eri', 'For', 'Gem', 'Gru', 'Her', 
                      'Hor', 'Hya', 'Hyi', 'Ind', 'LMi', 'Lac', 'Leo', 'Lep', 
                      'Lib', 'Lup', 'Lyn', 'Lyr', 'Men', 'Mic', 'Mon', 'Mus', 
                      'Nor', 'Oct', 'Oph', 'Ori', 'Pav', 'Peg', 'Per', 'Phe', 
                      'Pic', 'PsA', 'Psc', 'Pup', 'Pyx', 'Ret', 'Scl', 'Sco', 
                      'Sct', 'Ser', 'Sex', 'Sge', 'Sgr', 'Tau', 'Tel', 'TrA', 
                      'Tri', 'Tuc', 'UMa', 'UMi', 'Vel', 'Vir', 'Vol', 'Vul']

for constellation in constellation_list:
    cons_stars = []
    for star_obj in star_data:
        if constellation == star_obj.name[-3:] and \
           (star_obj.name.split()[0].isdigit() == False):
            cons_stars.append(star_obj)
    
    avg_a = sum([((star_obj.position[0])) for 
                 star_obj in cons_stars]) / len(cons_stars)
    avg_d = sum([star_obj.position[1] for star_obj in cons_stars]) / \
            len(cons_stars)

    for star_obj in cons_stars:
        star_obj.ortho_proj((avg_a, avg_d))
    
    file = open('{}.svg'.format(constellation), 'w')
    file.write('<?xml version="1.0" encoding="utf-8"?>')
    file.write('                 <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1000" height="1000" fill = "black">')
    file.write('<rect x ="0" y ="0" width ="1000" height="1000"/>')
    width = max([star_obj.position[1] for star_obj in cons_stars]) - min([star_obj.position[1] for star_obj in cons_stars])
    if width == 0:
        width = 1
    for star_obj in cons_stars:
        cx = 500 - star_obj.position_xy[0] * 600 / width
        cy = 500 - star_obj.position_xy[1] * 600 / width
        r = (10 - star_obj.magnitude) * 0.7
        if star_obj.name.split()[0].isdigit() == False:
            color = "blue"
        else:
            color = "black"
        file.write('<circle cx="{}" cy="{}" r="{}" style="stroke: {}; fill: white;"/>'.format(cx, cy, r, color))
    file.write('</svg>')