from PIL import Image

def bounded(numb, tolerance = 1000):
    z = 0
    for n in range(tolerance):
        z = z ** 2 + numb
        if abs(z) >= 2:
            return n
    return True

def mandelbrot(size, tolerance = 1000):
    m_set = []
    for y in range(size):
        cy = -2 + (y / size) * 4
        line = []
        for x in range(size):
            cx = -3 + (x / size) * 4
            if bounded(complex(cx, cy), tolerance) is True:
                line.append(True)
            else:
                line.append(bounded(complex(cx, cy), tolerance))
        m_set.append(line)
    return m_set
    
def print_mandelbrot(size, tolerance, white = False, letter = ''):
    im = Image.new('RGB', (size, size))
    m_set = mandelbrot(size, tolerance)
    for y, line in enumerate(m_set):
        for x, point in enumerate(line):
            if point is True:
                im.putpixel((x, y), (0, 0, 0))
            elif not white:
                n = int(((abs(point - tolerance) / tolerance) * 255) // 1)
                im.putpixel((x, y), (n, n, n))
            else:
                im.putpixel((x, y), (255, 255, 255))
    im.save('mandelbrot{}.png'.format(letter), 'PNG')           
    
    
#for line in mandelbrot(90):
#    print(''.join(['*' if element is True else ' ' for element in line]))
    
for n in (10, 50, 100, 500, 1000, 2000):
    print_mandelbrot(n, 100, letter = '{}'.format(n))
           