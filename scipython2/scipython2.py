import math
from protein_lengths import naa
import random
from element_symbols import element_symbols

#P2.2.4 
def geoid(a,c):
    e=math.sqrt(1-(c**2/a**2))
    return 2*math.pi*(a**2)*(1+((1-e**2)/e)*math.atanh(e))

def sphere(r):
    return (4/3)*math.pi*r**2
  
#P2.3.2c
def cons_table():  
    con_one='{symb:4} = {val: .4e} {unit:4}' 
    print(con_one.format(symb='kB',val=1.3806504*10**(-23),unit='J/K'),
          con_one.format(symb='mu_e',val=-9.28476377*10**(-24),unit='J/T'),
          con_one.format(symb='N_A',val=6.02214179*10**(23),unit='mol-1'),
          con_one.format(symb='c',val=299792458,unit='m/s'),
          sep='\n')
      
#P2.3.2d
def cons_table_two():
    con_two='==={symb:>5} = {val:+.2E} [{unit:>9}]    ==='
    print(con_two.format(symb='G',val=6.67428*10**(-11),unit='Nm^2/kg^2'),
          con_two.format(symb='\u03BC'+'e',val=-9.28476377*10**(-24),unit='J/T'),
          sep='\n')
      
#P2.3.3a
def matrix_table_one():
    str_row_one='[ {a1: .1f} {a2: .1f} {a3: .1f} ]'
    print(str_row_one.format(a1=0, a2=3.4, a3=-1.2),
          str_row_one.format(a1=-1.1, a2=0.5, a3=-0.2),
          str_row_one.format(a1=2.3, a2=-1.4, a3=-0.7),
          sep='\n')
      
#P2.3.3b
def matrix_table_two():
    str_row_two='[ {a1:.0f} {a2:.0f} {a3:.0f} ]'
    print(str_row_two.format(a1=0, a2=0, a3=1),
          str_row_two.format(a1=0, a2=1, a3=0),
          str_row_two.format(a1=1, a2=0, a3=1),
          sep='\n')
      
#P2.3.4
def planet_list():
    planet='{name:<8}:{symb}'
    print(planet.format(name='Mercury', symb='\u263F'),
          planet.format(name='Venus', symb='\u2640'),
          planet.format(name='Earth', symb='\u2641'),
          planet.format(name='Mars', symb='\u2642'),
          planet.format(name='Jupiter', symb='\u2643'),
          planet.format(name='Saturn', symb='\u2644'),
          planet.format(name='Uranus', symb='\u2645'),
          planet.format(name='Neptune', symb='\u2646'),
          planet.format(name='Pluto', symb='\u2647'),
          sep='\n')
     
#Q2.4.3
def test_ranker(scores):
    ranks = []
    rank = 1
    for i, score in enumerate(scores):
        if i > 0 and scores[i-1] == score:
            ranks.append(ranks[i-1])
        else:
            ranks.append(rank)
        rank = i + 2
    return ranks

#Q2.4.4
def madhava(length = 20):
    sums = 0
    for n in range(length):
        sums = sums + ((-1)**n)/(3**n*(1+2*n))
    return sums*math.sqrt(12)

#P2.4.1
def omitted_multiply(l):
    new_l = []
    total_multiply = 1
    for n in range(len(l)):
        total_multiply *= l[n]
    for n in range(len(l)):
        new_l.append(total_multiply // l[n])
    return new_l

#P2.4.2
def hamming(s1, s2):
    distance = 0
    for sub1, sub2 in zip(s1, s2):
        if sub1 != sub2:
            distance += 1
    return distance

#P2.4.3
def pi_out_loud(digits=8):
    numbers = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    pi = math.pi
    for n in range(digits):
        print(numbers[int(pi // 1)], end = ' ')
        pi = (pi - pi//1) * 10
        if n == 0:
            print('point', end = ' ')

#P2.4.4
def pascal_tri(rows=8):
    pascal=[[1],[1,1]]
    for n in range(2,rows+1):
        curr_row=[1]
        for element in range(1,n):
            curr_row.append(pascal[n-1][element-1]+pascal[n-1][element])
        curr_row.append(1)
        pascal.append(curr_row)
    for x in range(rows+1):
        print('== {row:^40} =='.format(row=' '.join([str(z) for z in pascal[x]])))

#P2.4.5
def codon_unpack(code, frame):
    codons = []
    for n in range(int((len(code) - frame) // 3)):
        codons.append(code[3 * n + frame : 3 * (n + 1) + frame])
    return codons

#P2.4.6
def double_factorial(numb):
    product = 1
    for n in range(int(numb//2)):
        product *= numb - 2*n
    return product
    
#P2.4.7
def benford(numb):
    first_dig = [0 for x in range(9)]
    for x in range(len(numb)):
        first_dig[int(numb[x] // (10**(math.log10(numb[x]) // 1)))-1] += 1
    first_dig = ['{val:.3f}'.format(val = (x / len(numb))) for x in first_dig]
    return first_dig
    
#P2.4.7a
def fibonacci(length=10):
    fib = [1,1]
    for n in range(1, length - 1):
        fib.append(fib[n]+fib[n-1])
    return fib
    
#Q2.5.1
def normalization(seq):
    return [(element - min(seq)) / (max(seq) - min(seq)) for element in seq]

#Q2.5.2
def agm(a, b, pres=5):
    while True:
        a, b = 0.5*(a+b), math.sqrt(a*b)
        if '{:.{}f}'.format(a, pres) == '{:.{}f}'.format(b, pres):
            break
    return a

#Q2.5.3
def fizzbuzz(end):
    fizzbuzz = []
    for n in range(1, end+1):
        if not n % 3 and not n % 5:
            fizzbuzz.append('FizzBuzz')
        elif not n % 5:
            fizzbuzz.append('Buzz')
        elif not n % 3:
            fizzbuzz.append('Fizz')
        else:
            fizzbuzz.append(n)
    return fizzbuzz

#Q2.5.4
def alkanes(formula):
    carbons = ''
    hydrogens = ''
    n = formula.index('C') + 1
    while formula[n].isdigit():
        carbons += formula[n]
        n += 1
    h = formula.index('H') + 1
    while h < len(formula) and formula[h].isdigit():
        hydrogens += formula[h]
        h += 1
    if int(hydrogens) != 2 * int(carbons) + 2:
        return 'Not an unsaturated alkane'
    structure = 'H3C'
    for x in range(int(carbons)-2):
        structure += '-CH2'
    structure += '-CH3'
    return structure
    
#P2.5.2
def weak_acid(c, K_a, tol = 1 * 10 ** (-10)):
    a, b = 1, 0
    while abs(a - b) > tol:
        a, b = b, math.sqrt(K_a * (c - b))
    return b

#P2.5.3
def luhn(card):
    card_list = [int(x) for x in [*(str(card)[::-1])]]
    print(card_list)
    for digit in range(1, len(card_list), 2):
        card_list[digit] *= 2
        while card_list[digit] >= 10:
            card_list[digit] = [int(x) for x in [*(str(card_list[digit]))]]
            card_list[digit] = card_list[digit][0] + card_list[digit][1]
    total_sum = 0
    for digit in card_list:
        total_sum += digit
    return not bool(total_sum % 10)
    
#P2.5.4
def hero_root(numb, guess = 1, tol = 0.01):
    a, b = guess, numb
    while abs(a - b) > tol:
        a, b = b, 0.5*(a + numb / a)
    return b

#P2.5.5
def tomorrow(today, us_date_style = True):
    months = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    date = [int(x) for x in today.split('/')]
    if not us_date_style:
        date[0], date[1] = date[1], date[0]
    if date[1] < months[date[0]-1]:
        date[1] += 1
    else:
        if date[0] == 12:
            date[0], date[1], date[2] = 1, 1, date[2] + 1
        elif date[0] == 2:
            if not date[2] % 400:
                date[1] += 1
            elif not date[2] % 100:
                date[0], date[1] = 3, 1
            elif not date[2] % 4:
                date[1] += 1
            else:
                date[0], date[1] = 3, 1
            if date[1] == 30:
                date[0], date[1] = 3, 1
        else:
            date[0], date[1] = date[0] + 1, 1
    if us_date_style:
        return str(date[0]) + '/' + str(date[1]) + '/' + str(date[2])
    else:
        return str(date[1]) + '/' + str(date[0]) + '/' + str(date[2])         
            
#P2.5.6
def de_polignac(n):
    zeroes = 0
    for x in range(1, n+1):
        zeroes += (n / (5 ** x)) // 1
    return zeroes

#P2.5.7a
def hailstone(start):
    sequence = []
    while start != 1:
        sequence.append(start)
        if start % 2:
            start = 3 * start + 1
        else:
            start = start // 2
    return sequence

#P2.5.7b
def hailstone_stop(start):
    stop_time = 0
    while start != 1:
        if start % 2:
            start = 3 * start + 1
        else:
            start = start // 2
        stop_time += 1
    return stop_time

#P2.5.8
def eratosthenes(end):
    primes = [x for x in range(2, end + 1)]
    for prime in primes:
        for n in primes[prime:]:
            if not n % prime:
                primes.remove(n)
    return primes

#P2.5.9    
def totient(n):
    totient = 0
    for x in range(1, n):
        a, b = n, x
        while b:
            a, b = b, a % b
        if a == 1:
            totient += 1
    return totient
    
#P2.5.9
def pi_monte(points = 1000):
    pi_count = 0
    k = points
    while k > 0:
        x, y = random.random(), random.random()
        if y <= (math.sqrt(1 - x ** 2)):
            pi_count += 1
        k += -1
    return (pi_count * 4) / points
    
#P2.5.11
def jumble(string):
    string_list = [list(x) for x in [y for y in string.split()]]
    for word in string_list:
        start, last = 1, len(word) - 1
        for letter in word:
            if letter.isalpha():
                break
            start += 1
        for letter in word[::-1]:
            if letter.isalpha():
                break
            last += -1
        temp = word[start:last]
        random.shuffle(temp)
        word[start:last] = temp
    final_phrase = []
    for word in string_list:
        final_phrase.append(''.join(word))
    return ' '.join(final_phrase)
    
#P2.5.12
def electron_config(number, condensed = True):
    config = ['{symb}:'.format(symb = element_symbols[number - 1])]
    atomic_number = number
    counter = 1
    suborbitals = ('s', 'p', 'd', 'f')
    nobles = (2, 10, 18, 36, 54, 86, 118)
    noble = 0
    if condensed:
        while number - nobles[noble + 1] > 0:
            noble += 1
        config.append('[{}]'.format(element_symbols[nobles[noble] - 1]))
    while number > 0:
        for shell in range(1,8):
            suborbital = counter - shell
            if suborbital < 0 or number < 1 or suborbital > shell - 1:
                continue
            electrons = 0
            for electron in range(2 * (2 * suborbital + 1)):
                if number < 1:
                    break
                electrons += 1
                number += -1
            if not condensed or (number < (atomic_number - nobles[noble])):
                config.append('{}{}{}'.format(shell, suborbitals[suborbital], electrons))
        counter += 1
    return ' '.join(config)
    
#P2.6.1
def redwood(file = 'redwood-data.txt'):
    redwoods = open('{}'.format(file), 'r')
    trees = [tree.split() for tree in redwoods.readlines()]
    counter = 0
    park = ('Humboldt', 'Redwood', 'Montgomery')
    while counter < len(trees):
        if '#' in trees[counter]:
            trees.remove(trees[counter])
            counter += -1
        counter += 1
    diameters = [float(tree[-2]) for tree in trees]
    heights = [float(tree[-1]) for tree in trees]
    max_height = 0
    max_diameter = 0
    for tree in trees:
        if str(max(diameters)) == tree[-2]:
            max_diameter = tree
        if str(max(heights)) == tree[-1]:
            max_height = tree
    for i, word in enumerate(max_height):
        if word in park:
            max_height = ' '.join(max_height[0:i])
            break
    for i, word in enumerate(max_diameter):
        if word in park:
            max_diameter = ' '.join(max_diameter[0:i])
            break
    return 'Tallest: '+ str(max_height) + '; Widest: ' + str(max_diameter)

#P2.6.2a
def tough_censor(file, banned = ('c', 'fortran')):
    file = open('{}'.format(file), 'r')
    new_text = []
    for line in file:
        new_line = line
        for word in banned:
            while word.lower() in new_line.lower():
                new_line = (new_line[:new_line.lower().index(word.lower())] + 
                            '*' * len(word) + new_line[new_line.lower()
                            .index(word.lower()) + len(word):])
        new_text.append(new_line)
    return new_text

#P2.6.2b
def easy_censor(file, banned = ('c', 'fortran')):
    file = open('{}'.format(file), 'r')
    new_text = []
    words = [line.split() for line in file.readlines()]
    for line in words:
        new_line = []
        for word in line:
            start, last = 0, len(word)
            for letter in word:
                if letter.isalpha():
                    break
                start += 1
            for letter in word[::-1]:
                if letter.isalpha():
                    break
                last += -1
            if word[start:last].lower() in banned:
                word = word[:start] + '*' * len(word[start:last]) + word[last:]
            new_line.append(word)
        new_line = ' '.join(new_line)
        new_text.append(new_line)
    return '\n'.join(new_text)
    
#P2.6.3
def earth_similarity(file, line_skip = 3):
    planets = open('{}'.format(file), 'r')
    pos = (-8, -6, -4, -2)
    earth = (1.0, 1.0, 1.0, 288)
    weights = (0.57, 1.07, 0.7, 5.58)
    habitability = []
    for n in range(line_skip):
        line = planets.readline()
    while True:
        line = planets.readline().split()
        if line == []:
            break
        esi = 1
        for i in range(4):
            esi *= ((1 - abs((float(line[pos[i]]) - earth[i])/(float(line[pos[i]])
                     + earth[i]))) ** (weights[i] / 4))
        habitability.append((line[0], '{:.5f}'.format(esi)))
    return habitability
 
#P2.6.4a
def arrays(file):
    file = open('{}'.format(file), 'r')
    array = []
    for line in file:
        array.append(line.split())
    return array

#P2.6.4b
def array_col(file):
    matrix = arrays(file)
    return [[row[i] for row in matrix]for i in range(len(matrix[0]))]
    
#P2.6.4c
def array_diag(file, reverse = False):
    if reverse:
        matrix = array_col(file)
        for row in matrix:
            row.reverse()
    else:
        matrix = arrays(file)
    diags = []
    counter = 0
    while counter <= len(matrix[0]) + len(matrix) - 2:
        entry = []
        for y in range(len(matrix)):
            for x in range(len(matrix[0])):
                if x + y == counter:
                    entry.append(matrix[y][x])
        counter += 1
        diags.append(entry)
    if reverse:
        diags.reverse()
    return diags
    
#P2.7.1
def scrabble(start, word, direction):
    letters = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O')
    coords = list(start)
    if direction in ('across', 'left', 'right', 'a', 'acros', 'accros', 'accross'):
        return len(word) <= max([letters.index(coords[0]) + 1, 14 - letters.index(coords[0])])
    return len(word) <= max([int(coords[1]) + 1, 14 - int(coords[1])])
    
#P2.7.2
def factorial_divisor():
    n = 1
    while n<10:
        sum = 0
        for number in list(str(math.factorial(n))):
            sum += int(number)
        if math.factorial(n) % sum:
            return(n)
        n += 1
        
#P2.7.3a
def dot(a, b):
    dot_product = []
    for an, bn in zip(a,b):
        dot_product.append(an * bn)
    return dot_product

#P2.7.3b
def cross(a, b):
    cross_product = [0, 0, 0]
    cross_product[0] = a[1] * b[2] - a[2] * b[1]
    cross_product[1] = b[0] * a[2] - a[0] * b[2] 
    cross_product[2] = a[0] * b[1] - a[1] * b[0]
    return cross_product
    
#P2.7.3c
def scalar_trip(a, b, c):
    return dot(a, cross(b, c))
    
#P2.7.3d
def vector_trip(a, b, c):
    return cross(a, cross(b, c))
    
#P2.7.4
def pyramid_AV(n, s, h):
    a = (1 / 2) * s * (1 / math.tan(math.pi / n))
    volume = (1 / 3) * ((1 / 2) * n * s * a) * h
    s_area = ((1 / 2) * n * s) * (a + math.sqrt(h ** 2 + a ** 2))
    return volume, s_area

#P2.7.5
def projectile(v, a, radians = False):
    if not radians:
        a *= math.pi / 180
    R = ((v ** 2) * math.sin(2 * a)) / 9.81
    H = ((v ** 2) * (math.sin(a)) ** 2) / (2 * 9.81)
    return R, H
    
#P2.7.6
def sinm_cosn(m, n):
    if m <= 1 or n <= 1:
        return'Invalid values!'
    if not m % 2 and not n % 2:
        return ((math.pi / 2) * (double_factorial(m-1) * double_factorial(n-1))
                / double_factorial(m + n))
    return ((double_factorial(m-1) * double_factorial(n-1))
            / double_factorial(m + n))
            
#P2.7.7
def palindrome_two(word):
    print(word)
    print(word[0] == word[-1])
    if len(word) <= 1:
        return True
    if word[0] != word[-1]:
        return False
    if not palindrome_two(word[1:-1]):
        return False
    return True

#P2.7.8
def tetration(n, x):
    if n == 1:
        return x
    return x ** tetration(n-1, x)
    