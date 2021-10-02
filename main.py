import matplotlib.pyplot as plt
import math
import random
import numpy as np

# Global Variables and Data Structures
objectiveFunctionIndicator = 1 #default value
uptoDecimalPlaces = 3    # Number used in unidirectional search to define upto how many digits we want after decimal, for epsinol = 10**-4 its limited to 4-1 = 3
epsinol = 10**-4
epsilon1 = 10**-3
epsilon2 = 10**-3
epsilon3 = 10**-3
M=100                        # Max iterations in conjugate gradient method
deltaX = 10**-5
index = 0
grad = 0.0
noOfVariables = 2
delta = None
noOf_functionEval = None
intermediatepoints =100
queueSize = 5
myQ_1 = None  # Queue
xi = np.zeros(noOfVariables)
si = np.zeros(noOfVariables)

out = open(r"Phase_1_iterations.out", "w")

# Bounding Phase Method varibales for ploting
x_series = []
f_x_series = []
x_1_series = []
f_x_1_series = []
x_2_series = []
f_x_2_series = []

# Interval Halving Method Variables for ploting
x_m_series = []
f_x_m_series = []

# Let make a Queue(data structure) which will store the recent x values and corresponding function values
# we will only store limited values at this point


class CustomQueue:
    global queueSize

    def __init__(self) -> None:
        self.x_values = []
        self.function_values = []

    def push_x_and_f_value(self, x, f):
        # x is the point and f=function(x)
        self.x_values.append(x)
        self.function_values.append(f)

        # now pope the first item from both the que
        if(len(self.x_values) > queueSize):
            self.x_values.pop(0)
            self.function_values.pop(0)
        else:
            pass

class CustomList:
    global queueSize

    def __init__(self) -> None:
        self.index =[]
        self.x_values = []
        self.function_values = []

    def push_x_and_f_value(self, x, f):
        global index
        # x is the point and f=function(x)
        self.index.append(index)
        self.x_values.append(x)
        self.function_values.append(f)
        index = index + 1

# Initiate the Optimization Method Here
def start():
    # take input from the user and run the methods
    global out, noOf_functionEval, myQ_1, objectiveFunctionIndicator, delta

    # set no of function evaluation = 0
    noOf_functionEval = 0

    # reinitialise the queue
    myQ_1 = CustomQueue()
    
    objectiveFunctionIndicator = float(input("Input the serial number of desired objective function\t->\t"))
    # lowerLimit = float(input("Please enter the lower limit\t->\t"))
    # upperLimit = float(input("Please enter the upper limit\t->\t"))
    lowerLimit = -10.0
    upperLimit = 10.0
    
    
    # intermediatepoints = float(input("Please enter the number of intermediate points\t->\t"))
    print(conjugateGradientMethod(lowerLimit, upperLimit))

    # print(f"Bounding Phase Method -> {a} to {b}")
    # print(
    #     f"No of function evaluations at the end of Bracketing Method = {noOf_functionEval}")

    # print(f"Interval Halving Method -> {a} to {b}")
    # print(f"Total no of function evaluations = {noOf_functionEval}")

def gradientV(x):                              # A seperate function for gradient calculation of a function, input is a vector, output is a vector
    global grad, deltaX
    gradF = np.zeros(noOfVariables)
    gradF =x
    # print(grad)
    grad = 1
    # print(0)
    # print(x)
    # print(x+deltaX)
    # gradF = (objectiveFunction(x+deltaX) -objectiveFunction(x-deltaX))/2*deltaX
    grad = 0.0
    # print(gradF)
    # print(1123)
    return  gradF           

def magnitudeOfVector(x):                   # Function to calculate magnitude of a vector
    value = 0
    for i in range(noOfVariables):
        value = value + x[0,i]**2
    value = value**0.5
    return value

def conjugateGradientMethod(a, b):
    global xi, si, index
    k=0
    xStart = (b-a)*np.random.rand(1, noOfVariables) +a*np.ones(noOfVariables)           # A random vector x to start with between a and b limits, b is upperLimit an a is lowerLimit
    
    gradF = gradientV(xStart)
    # print(0000000)
    # list = np.empty([2, M])                                       # Storing x vector in 0th row and its gradient vector in 1st row
    # list[0,k] = xStart
    # list[1,k] = gradF

    list = CustomList()
    list.push_x_and_f_value(xStart, gradF)

    xi = xStart
    si = -gradF

    lmbda = unidirectionalSearch(a, b)                                                 # lmbda = lambda, output of unidirectional method will be lambda
    xStart = xi + lmbda*si
    k=1
    gradF = gradientV(xStart)

    # list[0,k] = xStart
    # list[1,k] = gradF
    list.push_x_and_f_value(xStart, gradF)
    # print(index)
    # print(11111111111111111111)
    while True:
        if(k-1 in list.index):
            ind = list.index.index(k-1)
            val = list.function_values[ind]
        sk = -gradF + (magnitudeOfVector(gradF)**2/magnitudeOfVector(val)**2)*si

        #Linear independence check
        cosine = (si*sk)/(magnitudeOfVector(si)*magnitudeOfVector(sk))
        # if(cosine>=0.99):
        #     pass


        xi = xStart
        si = sk

        lmbda = unidirectionalSearch(a,b)

        xStart = xi + lmbda*si
        gradF = gradientV(xStart)

        # list[0,k] = xStart
        # list[1,k] = gradF
        list.push_x_and_f_value(xStart, gradF)

        if(magnitudeOfVector(xStart-xi)/magnitudeOfVector(xi)<=epsilon2):
            break

        if(magnitudeOfVector(gradF)<=epsilon3):
            break



    return xStart


def unidirectionalSearch(a,b): # Returns a number that matches optimum point upto decimal places defines by "uptoDecimalPlaces"
    global delta
    myQ_1 = CustomQueue()
    delta = (b-a)/intermediatepoints                  #a= lowerlimit, b = upperlimit
    [a, b] = boundingPhaseMethod(a, b)

    myQ_1 = CustomQueue()
    out.write("\n\n")
    [a, b] = intervalHalving(a, b)

    step = 10.0**uptoDecimalPlaces
    return math.trunc(step*a)/step

# Bracketing Method
def boundingPhaseMethod(a, b):
    global out, x_series, f_x_series

    # Write in the output File
    out.write("Bounding Phase Method - Results\n")
    out.write("#Iteration\tx\tf(x)\n")

    k = 0
    deltaWithSign = None
    while True:
        # step 1
        x_0 = random.uniform(a, b)
        if (x_0 == a or x_0 == b):
            continue

        # step 2

        # In the below code there will be 3 function evaluations
        if objectiveFunction(x_0 - abs(delta)) >= objectiveFunction(x_0) and objectiveFunction(x_0 + abs(delta)) <= objectiveFunction(x_0):
            deltaWithSign = + abs(delta)
        elif objectiveFunction(x_0 - abs(delta)) <= objectiveFunction(x_0) and objectiveFunction(x_0 + abs(delta)) >= objectiveFunction(x_0):
            deltaWithSign = - abs(delta)
        else:
            continue

        # Write the initial value of x and f(x)
        out.write("{}\t{}\t{}\n".format(k, x_0, objectiveFunction(x_0)))

        # Fill the data in the List for ploting
        x_series.append(x_0)
        f_x_series.append(objectiveFunction(x_0))

        while True:
            # step 3
            x_new = x_0 + 2**k*deltaWithSign

            if objectiveFunction(x_new) < objectiveFunction(x_0):
                k += 1
                x_0 = x_new
                # write the new value of x and f(x)
                out.write("{}\t{}\t{}\n".format(
                    k, x_new, objectiveFunction(x_new)))

                # Fill the data in the List for ploting
                x_series.append(x_new)
                f_x_series.append(objectiveFunction(x_new))
                continue
            else:
                # return in [x_lower,x_upper] format
                temp1 = x_new-(2**k)*1.5*deltaWithSign
                temp2 = x_new
                if temp1 > temp2:
                    return [temp2, temp1]
                else:
                    return [temp1, temp2]

    '''
    # The total no of function evaluation should be equal to (No. of iteration + 2)
    for this function as k starts with 0 so the total no of function evaluation = (k+1)+2
    '''


# Bounding Phase method
def intervalHalving(a, b):
    global x_1_series, x_m_series, x_2_series, f_x_1_series, f_x_m_series, f_x_2_series

    # step 1
    x_m = (a+b)/2
    l = b-a
    no_of_iteration = 1

    # Write in the output File
    out.write("Interval Halving Method - Results\n")
    out.write(
        "#Iteration\tx_1\tx_m\tx_2\tf(x_1)\tf(x_m)\tf(x_2)\n")

    while True:
        # step2
        x_1 = a+l/4
        x_2 = b-l/4

        # Write to the output file
        out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            no_of_iteration, x_1, x_m, x_2, objectiveFunction(x_1), objectiveFunction(x_m), objectiveFunction(x_2)))

        # Fill the data in the store as well
        x_m_series.append(x_m)
        f_x_m_series.append(objectiveFunction(x_m))
        x_1_series.append(x_1)
        f_x_1_series.append(objectiveFunction(x_1))
        x_2_series.append(x_2)
        f_x_2_series.append(objectiveFunction(x_2))

        while True:
            # step3
            if objectiveFunction(x_1) < objectiveFunction(x_m):
                b = x_m
                x_m = x_1
                break

            # step4
            if objectiveFunction(x_2) < objectiveFunction(x_m):
                a = x_m
                x_m = x_2
                break

            else:
                a = x_1
                b = x_2
                break

        # step5
        l = b-a
        if abs(l) < epsinol:
            return [a, b]
        else:
            no_of_iteration += 1
            continue


# Objective functions' definition
def objectiveFunction(x):   # x is a scalar
    # check whther the x is already stored in the queue
    global noOf_functionEval, grad, xi, si
    # print(11111)
    # print(x)
    if(grad == 1.0):
        # print(22222)
        return objectiveFunctionPhase2(x)

    if (x in myQ_1.x_values):
        index = myQ_1.x_values.index(x)
        value = myQ_1.function_values[myQ_1.x_values.index(x)]
        
        # While dealing with Queue,
        # there might be a chance of one value remaining fixed (let say x_m) in corresponding itaraions.
        # Hence if you keep pushing other new value (let say x_1), it will soon eats up all the queue size
        # and you have to loose the the fixed value (in this case x_m).
        # So let update the the value as a last item if it appears again.

        if(index == 0):
            # When we are about to loose the value swap the value at the end again of the queue.
            myQ_1.x_values.pop(0)
            myQ_1.x_values.append(x)
            myQ_1.function_values.pop(0)
            myQ_1.function_values.append(value)

        return value
    # print(131231123)
    # print(xi)
    # print(si)
    value = objectiveFunctionPhase2(xi+x*si)

    myQ_1.push_x_and_f_value(x, value)
    return value

def objectiveFunctionPhase2(x): # input is a vector, output is a scalar value
    f = 0
    # print(123)
    # print(x)
    if(objectiveFunctionIndicator == 1):
        for i in range(noOfVariables):
            f = f + (i+1)*x[0,i]**2
    elif(objectiveFunctionIndicator == 2):
        for i in range(noOfVariables-1):
            f = f + 100*(x[0,i+1]-x[0,i]**2)**2 + (x[0,i]-1)**2        

    return f

# Run the code from here.
start()