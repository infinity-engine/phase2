import numpy as np
import math
import random
import matplotlib.pyplot as plt

objectiveFunctionIndicator = 1  # default value
dictForObjectiveFunction = {}
noOfFunctionEvaluations = 0
angleForDependencyInDegree = 1  # For Linear Dependancy Check
x_series = []  # Just for ploting, should not be used in other program


def objectiveFunction(*args):
    """This will provide the objective function values.

    Returns:
        number: Function values
    """
    global objectiveFunctionIndicator, dictForObjectiveFunction, noOfFunctionEvaluations

    # check whether values are already stored or not.
    if args in dictForObjectiveFunction:
        return dictForObjectiveFunction.get(args)

    # when values are not stored calculate fresh.
    result = 0
    # Sum Squares Function
    if objectiveFunctionIndicator == 1:
        sum = 0
        for i in range(len(args)):
            sum = sum + (i+1)*(args[i])**2
        result = sum

    # Rosenbrock Function
    elif objectiveFunctionIndicator == 2:
        sum = 0
        for i in range(len(args)-1):
            sum = sum + (100*(args[i+1]-args[i]**2)**2+(args[i]-1)**2)
        result = sum

    # Dixon-Price Function
    elif objectiveFunctionIndicator == 3:
        term_1 = (args[0]-1)**2
        sum = 0
        if len(args) > 1:
            for i in range(len(args)-1):
                sum = sum + (i+2)*(2*args[i+1]**2-args[i])**2
        result = term_1+sum

    # Trid Function
    elif objectiveFunctionIndicator == 4:
        sum_1 = sum_2 = 0
        for i in range(len(args)):
            sum_1 = sum_1 + (args[i]-1)**2
        for i in range(len(args)-1):
            sum_2 = sum_2 + args[i+1]*args[i]
        result = sum_1-sum_2

    # Zakharov Function
    elif objectiveFunctionIndicator == 5:
        sum_1 = sum_2 = sum_3 = 0
        for i in range(len(args)):
            sum_1 = sum_1 + args[i]**2
            sum_2 = sum_2 + 0.5*(i+1)*args[i]
        sum_3 = sum_2
        result = sum_1 + sum_2**2 + sum_3**4

    else:
        return None

    # store the new calculated value in the dictionary
    dictForObjectiveFunction[args] = result
    noOfFunctionEvaluations += 1
    return result


def partialDerivative(functionToOperate, variableIndicator, currentPoint):
    """
    This function will partially derive the a function
    with respect to variabel at a given point.
    It uses central difference method to implement the partial derivatives of first order.

    Args:
        functionToOperate (call back function): [function on which we will be differentiating]
        variableIndicator (int): [its an indicator for variable, starts from 1, with respect to which we will be partially differentiating]
        currentPoint (list): [current point at which we need to make the differentiation]

    Returns:
        [number]: [value]
    """
    deltaX = 10**-4
    pointOne = currentPoint.copy()
    pointTwo = currentPoint.copy()
    indicatorValue = currentPoint[variableIndicator-1]
    pointOne[variableIndicator-1] = indicatorValue + deltaX
    pointTwo[variableIndicator-1] = indicatorValue - deltaX
    return (functionToOperate(*pointOne)-functionToOperate(*pointTwo))/(2*deltaX)


def gradiantOfFunction(functionToOperate, currentPoint):
    """Generate gradiant of a vector at a particular point

    Args:
        functionToOperate (call back function): function on which gradiant to be operate on.
        currentPoint (list): current point at which gradiant to be calculated.

    Returns:
        numpy array: gradiant vector
    """

    # Create a Zero matrix with no. of rows = no of variable in currentpoint
    # and only one column
    A = np.zeros((len(currentPoint), 1))
    for i in range(len(currentPoint)):
        A[i][0] = partialDerivative(functionToOperate, i+1, currentPoint)

    return A


def boundingPhaseMethod(functionToOperate, delta, a, b):
    """This is a Bracketing method.
    Which will be used to optimize a single variable function.

    Args:
        functionToOperate (call back function): Objective Function
        delta (float): Separation between two points
        a (float): Lower limit
        b (float): Upper limit

    Returns:
        [a,b]: List containing the brackets [Lower,Upper].
    """

    deltaWithSign = None
    k = 0
    while True:
        # step 1
        x_0 = random.uniform(a, b)
        if (x_0 == a or x_0 == b):
            continue

        # step 2
        # In the below code there will be 3 function evaluations
        if functionToOperate(x_0 - abs(delta)) >= functionToOperate(x_0) and functionToOperate(x_0 + abs(delta)) <= functionToOperate(x_0):
            deltaWithSign = + abs(delta)
        elif functionToOperate(x_0 - abs(delta)) <= functionToOperate(x_0) and functionToOperate(x_0 + abs(delta)) >= functionToOperate(x_0):
            deltaWithSign = - abs(delta)
        else:
            continue

        while True:
            # step 3
            x_new = x_0 + 2**k*deltaWithSign
            if functionToOperate(x_new) < functionToOperate(x_0):
                k += 1
                x_0 = x_new
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


def intervalHalvingMethod(functionToOperate, epsinol, a, b):
    """This is a Region Elimination method.
    Which will be used to find the optimal solution for a single variable function.

    Args:
        functionToOperate (call back function): Objective Function
        epsinol (float): Very small value used to terminate the iteration
        a (float): Lower limit
        b (float): Upper limit

    Returns:
        List: A list contains the bracket [lower,upper]
    """

    # step 1
    x_m = (a+b)/2
    l = b-a
    no_of_iteration = 1
    while True:
        # step2
        x_1 = a+l/4
        x_2 = b-l/4
        while True:
            # step3
            if functionToOperate(x_1) < functionToOperate(x_m):
                b = x_m
                x_m = x_1
                break

            # step4
            if functionToOperate(x_2) < functionToOperate(x_m):
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


def changeToUniDirectionFunction(functionToOperate, x, s):
    """This is a function which will be used to scale the multivariable function
    into a single variable function using uni direction method.

    Args:
        functionToOperate (call back function): Multivariable objective function
        x (row vector): Position of initial point in [1,2,...] format
        s (row vector): Direction vector in [1,2,....] format

    Returns:
        function: Scaled functiion in terms of single variable -> a
    """
    return lambda a: functionToOperate(*(np.array(x)+np.multiply(s, a)))


def conjugateGradiantMethod(functionToOperate, limits, initialPoint):
    """This is an Gradiant Based Multi-Variable Optimisation Method.
    It is used to find the optimal solution of an objective function.

    Args:
        functionToOperate (call back function): Objective Function
        limits (list): in [lower,upper] format
        initialPoint (list): in [x0,x1,x2....] format
    """
    global x_series
    # step 1
    a, b = limits
    x_0 = list(initialPoint)
    epsinolOne = 10**-8
    epsinolTwo = 10**-8
    epsinolThree = 10**-3
    k = 0
    M = 1000
    x_series = []  # store the x vextors
    x_series.append(x_0)

    # step2
    s_series = []  # store the direction vectors
    gradiantAtX_0 = gradiantOfFunction(functionToOperate, x_0)
    s_series.append(-gradiantAtX_0)
    # print(x_series[-1],gradiantAtX_0)
    # Extra termination condition *****
    if (np.linalg.norm(gradiantAtX_0)) <= epsinolThree:
        print(f"CG: Termination Point 1. Iterations Count -> {k}")
        return (x_0)

    # step3
    # convert multivariable function into single variable function
    s_0 = s_series[0][:, 0]
    newObjectiveFunction = changeToUniDirectionFunction(
        functionToOperate, x_0, s_0)

    # search for unidirection optimal point
    m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
    m, n = intervalHalvingMethod(newObjectiveFunction, epsinolOne, m, n)
    optimumPoint = (m+n)/2
    # print("Optimum Point",optimumPoint)

    x_1 = (np.array(x_0)+np.multiply(s_0, optimumPoint))
    # print(x_1)
    x_series.append(x_1)
    k = 1

    while(True):

        # step 4
        part_1_s = -gradiantOfFunction(functionToOperate, x_series[k])
        p = (np.linalg.norm(-part_1_s))**2
        q = gradiantOfFunction(functionToOperate, x_series[k-1])
        r = (np.linalg.norm(q))**2
        t = p/r
        part_2_s = np.multiply(s_series[k-1], t)
        s = part_1_s + part_2_s
        s_series.append(s)  # s_series size will become to k+1

        # code to check linear independance
        s_k = s_series[k][:, 0]  # row vector
        s_k_1 = s_series[k-1][:, 0]  # row vector
        dotProduct = s_k @ s_k_1
        factor = np.linalg.norm(s_k)*np.linalg.norm(s_k_1)
        finalDotProduct = dotProduct/factor
        finalDotProduct = round(finalDotProduct, 3)
        dependencyCheck = math.acos(
            finalDotProduct)*(180/math.pi)  # in degrees
        # print(dependencyCheck)
        if abs(dependencyCheck) < angleForDependencyInDegree:
            # Restart
            print(f"Linear Dependency Found! Restarting with {x_series[k]}")
            return conjugateGradiantMethod(functionToOperate, limits, x_series[k])

        # step 5
        # convert multivariable function into single variable function
        x_k = x_series[k]  # 1-D list
        s_k = s_series[k][:, 0]  # 1-D list
        newObjectiveFunction = changeToUniDirectionFunction(
            functionToOperate, x_k, s_k)

        # search for unidirection optimal point
        m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
        m, n = intervalHalvingMethod(newObjectiveFunction, epsinolOne, m, n)
        optimumPoint = (m+n)/2
        x_new = (np.array(x_k)+np.multiply(s_k, optimumPoint))
        x_series.append(x_new)  # x_series size will be k+2

        # step 6
        # check the terminate condition
        norm_1 = np.linalg.norm(np.array(x_series[k+1])-np.array(x_series[k]))
        norm_2 = np.linalg.norm(x_series[k])
        factor = np.linalg.norm(gradiantOfFunction(
            functionToOperate, x_series[k+1]))

        if norm_2 != 0:
            if norm_1/norm_2 <= epsinolTwo:
                print(f"CG: Termination Point 2. Iterations Count -> {k}")
                return x_series[k+1]

        if factor <= epsinolThree or k+1 >= M:
            # terminate the function
            print(f"CG: Termination Point 3. Iterations Count -> {k+1}")
            return x_series[k+1]
        else:
            k += 1
            continue

        break


def start():
    #Function to initialise the program

    global objectiveFunctionIndicator
    out = open(r"CG_iterations.out", "w")
    objectiveFunctionIndicator = int(input("Enter function indicator : \t"))
    a = float(input("Enter the lower limit : \t"))
    b = float(input("Enter the upper limit : \t"))
    noOfVariables = int(input("Enter the number of variables : \t"))
    initialChoice = []
    for i in range(noOfVariables):
        initialChoice.append(random.uniform(a, b))
    print(f"\nInitial Point : \n{initialChoice}")
    print("\n")
    optimumPoint = conjugateGradiantMethod(
        objectiveFunction, [a, b], initialChoice)
    print(
        f"\nTotal no of function evaluations are \n{noOfFunctionEvaluations}\n")
    print(
        f"\nOptimal solutions for the current objective function, for the given range, is \n{optimumPoint}\n")
    print(
        f"\nOptimal value of the function is \n{objectiveFunction(*optimumPoint)}\n")

    # For ploting
    x_axis = range(len(x_series))
    y_axis = []
    for x in x_series:
        y_axis.append(objectiveFunction(*x))
        out.write(f"{x}\t{objectiveFunction(*x)}")
        out.write("\n")
    plt.plot(x_axis, y_axis, "r*-")
    plt.xlabel("Iteration Count")
    plt.ylabel("F(X)")
    plt.legend(["F(X)"])
    plt.show()


""" 
objectiveFunctionIndicator = 3
for i in range(1):
    print(conjugateGradiantMethod(objectiveFunction, [-10, 10], [
          9.274223534072267, 4.94298352297724, -6.49988330765493, -1.9615581061654161])) 
"""

start()


""" # For Observation
for i in range(5):
    objectiveFunctionIndicator = i+1
    print("\n\n\n\nObjective Function\t", objectiveFunctionIndicator, "\n")
    for j in range(5):
        initialChoice = []
        for k in range(5):
            initialChoice.append(random.uniform(-10, 10))
        print("Initial point\n", initialChoice)
        print(conjugateGradiantMethod(objectiveFunction, [-10, 10], initialChoice)) """
