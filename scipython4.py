import math
import pylab
from morse import morse
import copy
import sys

#P4.1.1
def swallow_speeds():
    file = open('swallow-speeds.txt', 'r')
    speeds = file.readlines()
    speed_sum = 0
    values = 0
    for speed in speeds:
        try:
            speed_sum += float(speed)
        except ValueError:
            continue
        values += 1
    return speed_sum / values
    
#P4.1.2
def str_vector(v):
    assert type(v) is list or type(v) is tuple, \
           'argument to str_vector must be a list or tuple'
    assert len(v) in (2,3), \
           'vector must be 2D or 3D in sstr_vector'
    for element in v:
        assert element.imag == 0, \
           'elements of vector must be real'
    unit_vectors = ['i', 'j', 'k']
    s = []
    for i, component, in enumerate(v):
        s.append('{}{}'.format(component.real, unit_vectors[i]))
    return '+'.join(s).replace('+-', '-')

#P4.1.3
def power(a, b):
    if a == 0 and b == 0:
        raise ValueError('0 ** 0 is undefined')
    return a ** b

#Q4.2.1
def pangram(word):
    return set() == set(list('abcdefghijklmnopqrstuvwxyz')) - \
                    set(list(word.lower()))
                    
#P4.2.1
def resistor(col1, col2, col3, col4):
    colors = {'bk' : (0, 0, -1), 'br' : (1, 1, 1), 'rd' : (2, 2, 2), 
              'or' : (3, 3, -1), 'yl' : (4, 4, 5), 'gr' : (5, 5, 0.5), 
              'bl' : (6, 6, 0.25), 'vi' : (7, 7, 0.1), 'gy' : (8, 8, 0.05), 
              'wh' : (9, 9, -1), 'au' : (-1, -1, 5), 'ag' : (-1, -1, 10), 
              '--' : (-1, -1, 20)}
    if colors[col1][0] == -1 or colors[col2][0] == -1 or colors[col3][1] == -1 \
       or colors[col4][2] == -1:
           return (-1, -1)
    value = (colors[col1][0] * 10 + colors[col2][0]) * 10 ** colors[col3][1]
    return (value, colors[col4][2])
    
#P4.2.2a
def word_count(file, length):
    file = open('{}'.format(file), 'r')
    text = file.readlines()
    punc = ('!', '?', ':', ';', ',', '(', ')', '\'', '.', '*', '[', ']', '"', '-', r'\n')
    split = ('\t')
    for i, line in enumerate(text):
        for symbol in punc:
            text[i] = text[i].replace(symbol, '')
        for symbol in split:
            text[i] = text[i].replace(split, ' ')
    for i, line in enumerate(text):
        text[i] = text[i].split()
    words = dict([])
    for line in text:
        for word in line:
            if words.get(word, -1) == -1:
                words[word] = 1
            else:
                words[word] += 1
    top_words = []
    for n in range(length):
        big_word = ''
        for key in words:
            if words[key] > words.get(big_word, -1):
                big_word = key
        top_words.append((big_word, words[big_word]))
        words[big_word] = 0
    return top_words
    
#P4.2.2b
def zipf(file, length):
    words = word_count(file, length)
    x = [x + 1 for x in range(length)]
    y1 = [math.log10(word[1]) for word in words]
    pylab.plot(x, y1, 'b-', label = 'Actual Frrequency')
    
    y2 = y1[0] - pylab.log10(x)
    pylab.plot(x, y2, 'r--', label = 'Theoretical Frequency')
    
    pylab.legend()
    pylab.show

print(zipf('moby-dick.txt', 2000))

#P4.2.3
def rpn():
    stack = []
    a, b = 1, 1
    while True:
        value = input()
        if value in ('+', '-', '*', '/', '**'):
            a = stack.pop()
            b = stack.pop()
            ops = {'+':b+a, '-':b-a, '*':b*a, '/':b/a, '**':b**a}
            print(ops[value])
            continue
        elif value == 'q':
            return None
        else:    
            stack.append(int(value))
        
#P4.2.4
def morse_t(phrase):
    phrase = phrase.split()
    words = []
    for word in phrase:
        translated = ''
        for letter in word:
            translated += ' ' + morse[letter]
        words.append(translated)
    return ' / '.join(words)
    
#P4.2.3
def morse_d(phrase):
    morse_d = dict([(val[1], val[0]) for val in morse.items()])
    phrase = [word.split() for word in phrase.split('/')]
    words = []
    for word in phrase:
        translated = ''
        for letter in word:
            translated += morse_d[letter]
        words.append(translated)
    return ' '.join(words)
    
#P4.2.5
def sharks(file):
    file = open('{}'.format(file), 'r')
    sharks = file.readlines()
    for i,line in enumerate(sharks):
        sharks[i] = sharks[i].strip('\n')
    species = dict([])
    i = 0
    while True:
        while sharks[i][0].isalpha():
            species[sharks[i]] = {}
            order = i
            i += 1
            while sharks[i][4].isalpha() and sharks[i][0:4] == '    ':
                species[sharks[order]][sharks[i].strip(' ')] = {}
                family = i
                i += 1
                while sharks[i][8].isalpha() and sharks[i][0:8] == '        ':
                    species[sharks[order]][sharks[family].strip(' ')][sharks[i].strip(' ')] = {}
                    genus = i
                    i += 1
                    while ':' in sharks[i]:
                        temp = [name.strip() for name in sharks[i].split(':')]
                        sci_name = temp[0].split()
                        sci_name = '{}. {}'.format(sci_name[0][0], sci_name[1])
                        species[sharks[order]][sharks[family].strip(' ')][sharks[genus].strip(' ')][sci_name] = temp[1]
                        if i < len(sharks) - 1:
                            i += 1
                        else:
                            break
        break
    return species        

#P4.3.1
def trace(matrix):
    return sum([matrix[x][x] for x in range(len(matrix))])
    
#P4.3.2a
def rot13_word(word, shift):
    return ''.join([chr((ord(x) - 97 + shift) % 26 + 97)for x in word])
    
#P4.3.2a
def rot13(sentence, shift):
    return ' '.join([rot13_word(word, shift) for word in sentence.split()])
    
#P4.3.3a
def automata(state, rule):
    rule = '{:08b}'.format(rule)
    outcomes = {'111':rule[0], '110':rule[1], '101':rule[2], '100':rule[3], 
                '011':rule[4], '010':rule[5], '001':rule[6], '000':rule[7]}
    new_state = ['0'] + [outcomes[state[n-1] + state[n] + state[n+1]] 
                for n in range(1, len(state) - 1)] + ['0']
    return new_state

#P4.3.3b
def automata_runner(rule, rows):
    state = ['0'] * 40 + ['1'] + ['0'] * 39
    for n in range(rows):
        yield ''.join(['*' if x == '1' else ' ' for x in state])
        state = automata(state, rule)

#P4.3.4a
def lambda_dict(file):
    temp = lambda data: (data.split()[0], data.split()[1])
    return dict([temp(data) for data in open('{}'.format(file), 'r')])

#P4.3.4b
def list_dict(file):
    return dict([(line.split()[0], line.split()[1]) 
                 for line in open('{}'.format(file), 'r')])
                     
#P4.3.5
def P(input_set):
    input_set = tuple(input_set)
    if len(input_set) == 1:
        return (set(), set(input_set))
    new_input = P(input_set[:-1])
    new_set = copy.deepcopy(new_input)
    [n.add(input_set[-1]) for n in new_set]
    return new_input + new_set
        
#P4.3.6
def corpus():
    alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                'y', 'z')
    combos = ['{}{}'.format(x,y) for x in alphabet for y in alphabet]
    words = set()
    for n in [title.split()[0] for title in open('cats.txt', 'r')]:
        file = open('{}'.format(n), 'r')
        words |= set([word.split('/')[0] for line in file.readlines() 
                      for word in line.split() if len(word.split('/')[0]) == 8])
    final = {}
    for n in combos:
        appearance = []
        for word in words:
            if n in word:
                appearance.append(word)
        if len(appearance) == 2:
            final[n] = (appearance[0], appearance[1])
    return final   


    