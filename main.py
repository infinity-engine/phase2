import numpy as np
import math
import random

objectiveFunctionIndicator = 1  # default value
dictForObjectiveFunction = {}
noOfFunctionEvaluations = 0


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
    if objectiveFunctionIndicator == 1:
        sum = 0
        for i in range(len(args)):
            sum = sum + (i+1)*(args[i])**2

        # store the new calculated value in the dictionary
        dictForObjectiveFunction[args] = sum

        noOfFunctionEvaluations += 1
        return sum

    elif objectiveFunctionIndicator == 2:
        pass

    elif objectiveFunctionIndicator == 3:
        pass

    elif objectiveFunctionIndicator == 4:
        pass

    elif objectiveFunctionIndicator == 5:
        pass

    else:
        return None


def partialDerivative(functionToOperate, variableIndicator, currentPoint):
    """
    This function will partially derive the a function 
    with respect to variabel at a given point.
    It uses central difference method to implement the partial derivatives of first order.

    Args:
        functionToOperate (call back function): [function on which we will be differentiating]
        variableIndicator (int): [its an indicator for variable with respect to which we will be partially differentiating]
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
        nummpy array: gradiant vector
    """

    # Create a Zero matrix with no. of rows = no of variable in currentpoint
    # and only one column
    A = np.zeros((len(currentPoint), 1))
    for i in range(len(currentPoint)):
        A[i][0] = partialDerivative(functionToOperate, i+1, currentPoint)

    return A


def boundingPhaseMethod(functionToOperate, delta, a, b):
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


def intervalHalving(functionToOperate, epsinol, a, b):

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
    return lambda a: functionToOperate(*(np.array(x)+np.multiply(s, a)))


def conjugateGradiantMethod(functionToOperate, limits, initialPoint):
    # step 1
    a, b = limits
    x_0 = list(initialPoint)
    epsinolOne = epsinolTwo = epsinolThree = 10**-3
    k = 0
    M = 10
    x_series = []
    x_series.append(x_0)

    # step2
    s_series = []  # store the direction vector
    s_series.append(-gradiantOfFunction(functionToOperate, x_0))

    # step3
    # convert multivariable function into single variable function
    s_0 = s_series[0][:, 0]
    newObjectiveFunction = changeToUniDirectionFunction(
        functionToOperate, x_0, s_0)

    # search for unidirection optimal point
    m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
    m, n = intervalHalving(newObjectiveFunction, epsinolOne, m, n)
    optimumPoint = (m+n)/2
    #print("Optimum Point",optimumPoint)

    x = (np.array(x_0)+np.multiply(s_0, optimumPoint))
    # print(x)
    x_series.append(x)
    k = 1

    while(True):

        # step 4
        part_1_s = -gradiantOfFunction(objectiveFunction, x_series[k])
        p = (np.linalg.norm(-part_1_s))**2
        q = gradiantOfFunction(objectiveFunction, x_series[k-1])
        r = (np.linalg.norm(q))**2
        t = 0
        if p != 0 and r != 0:
            t = p/r
        part_2_s = np.multiply(s_series[k-1], t)
        s = part_1_s + part_2_s
        s_series.append(s)

        # write the code to check linear independance

        # step 5
        # convert multivariable function into single variable function
        x_k = x_series[k]  # 1-D list
        s_k = s_series[k][:, 0]  # 1-D list
        newObjectiveFunction = changeToUniDirectionFunction(
            functionToOperate, x_k, s_k)

        # search for unidirection optimal point
        m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
        m, n = intervalHalving(newObjectiveFunction, epsinolOne, m, n)
        optimumPoint = (m+n)/2
        x = (np.array(x_k)+np.multiply(s_k, optimumPoint))
        x_series.append(x)

        # step 6
        # check the terminate condition
        norm_1 = np.linalg.norm(np.array(x_series[k+1])-np.array(x_series[k]))
        norm_2 = np.linalg.norm(x_series[k])
        factor_1 = 0
        factor_2 = np.linalg.norm(gradiantOfFunction(
            functionToOperate, x_series[k+1]))

        if norm_1 != 0 and norm_2 != 0:
            factor_1 = norm_1/norm_2

        if factor_1 <= epsinolThree or factor_2 <= epsinolThree or k+1 >= M:
            # terminate the function
            return(x_series[k+1])
        else:
            k += 1
            continue

        break


print(conjugateGradiantMethod(objectiveFunction, [-10, 10], [-5, 5, 9]))
