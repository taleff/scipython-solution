import random

def initialInput(prompt, error='Please enter a valid value.', lower=0, upper=10**5):
    initval=0
    while True:
        try:
            initval=int(input(prompt))
            if initval>lower and initval<=upper:
                break
            print(error)
        except ValueError:
            print(error)
    return initval

width = initialInput("Please enter grid width:")
height =  initialInput("Please enter grid height:")
mineNumber =  initialInput("Please enter number of mines:", upper=width*height)   
            
sweepGrid=[['']*width for h in range(height)]


mines=[n for n in range(height*width)]
for m in reversed(range(mineNumber)):
    mines.append(mines.pop(random.randint(0,height*width-1)))
    
for mine in mines[-mineNumber:]:
    sweepGrid[mine//width][mine%width]='mh'

for row in range(height):
    for col in range(width):
        if sweepGrid[row][col] == 'mh':
            continue
        count=0
        if row > 0:
            for check in [-1,0,1]:
                if col+check<0 or col+check>width-1:
                    continue
                if sweepGrid[row-1][col+check] == 'mh':
                    count=count+1
        for check in [-1,0,1]:
                if col+check<0 or col+check>width-1:
                    continue
                if sweepGrid[row][col+check] == 'mh':
                    count=count+1
        if row<height-1:
            for check in [-1,0,1]:
                if col+check<0 or col+check>width-1:
                    continue
                if sweepGrid[row+1][col+check] == 'mh':
                    count=count+1  
        sweepGrid[row][col]=str(count)+'h'
        
def boardReturn(inputX,inputY):
    clicked=0
    sweepGrid[inputY][inputX]= sweepGrid[inputY][inputX][0]+'c'
    clicked=clicked+1
    if sweepGrid[inputY][inputX][0]=='0':
        if inputY > 0:
            for check in [-1,0,1]:
                if inputX+check<0 or inputX+check>width-1 or 'c' in sweepGrid[inputY-1][inputX+check]:
                    continue
                clicked+=boardReturn(inputX+check, inputY-1)
        for check in [-1,0,1]:
            if inputX+check<0 or inputX+check>width-1 or check==0 or 'c' in sweepGrid[inputY][inputX+check]:
                continue
            clicked+=boardReturn(inputX+check, inputY)
        if inputY<height-1:
            for check in [-1,0,1]:
                if inputX+check<0 or inputX+check>width-1 or 'c' in sweepGrid[inputY+1][inputX+check]:
                    continue
                clicked+=boardReturn(inputX+check, inputY+1)
    return clicked
    
def printBoard():
    [print("".join(["." if i[1]=="h" else str(i[0]) for i in j])) for j in sweepGrid]
    
def printFinalBoard():
    [print("".join([str(i[0]) for i in j])) for j in sweepGrid]

clickedCount=0
while True:
    try:
        picked=[int(x) for x in input("Please enter x, y:").split(',')]
        if picked[0]<0 or picked[0]>=width or picked[1]<0 or picked[1]>=height or len(picked)!=2 or sweepGrid[picked[1]][picked[0]][1]=="c":
            print("Please enter valid coordinates.")
            continue
    except ValueError:
        print("Please enter valid coordinates.")
        continue
    if 'm' in sweepGrid[picked[1]][picked[0]]:
        print("Game Over")
        printFinalBoard()
        break
    clickedCount += boardReturn(picked[0],picked[1])
    if clickedCount==width*height-mineNumber:
        print("Congratulations")
        printFinalBoard()
        break
    printBoard()
    
    
    