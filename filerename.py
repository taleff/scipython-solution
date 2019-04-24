import sys
import os

#P4.4.4

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 
          'oct', 'nov', 'dec']
          
try:
    dir_name = sys.argv[1]
except (IndexError):
    sys.exit('Please enter a directory name on the command line. \nUsage: '
             'python {:s} dir_name'.format(sys.argv[0]))
             
if not os.path.exists(dir_name):
    sys.exit('Directory does not exist. \nUsage: '
             'python {:s} dir_name'.format(sys.argv[0]))

for filename in os.listdir(dir_name):
    try:
        d, month, y = int(filename[5:7]), filename[8:11], int(filename[12:14])
    except (ValueError, IndexError, TypeError):
        continue
    
    try:
        m = months.index(month.lower()) + 1
    except ValueError:
        continue
    
    newname = 'data-20{:02d}-{:02d}-{:02d}.txt'.format(y, m, d)
    newpath = os.path.join(dir_name, newname)
    oldpath = os.path.join(dir_name, filename)
    print(oldpath, '->', newpath)
    os.rename(oldpath, newpath)