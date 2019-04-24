#P4.6.3
file = open('bsc5.dat', 'r')
stars = file.readlines()

star_data = []

#Creating a star object for every star in bsc5.dat
for i, star_obj in enumerate(stars):
    #Skip star instantiation if coordinates to not exist        
    star_data.append(star_obj[4:14])
    
constellations = set()    
for name in star_data:
    if name[-3:].isalpha():
        constellations.add(name[-3:])
                          
print(sorted(constellations))