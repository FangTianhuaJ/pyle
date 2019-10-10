import copy
import math
import numpy as np
import time
import matplotlib.pyplot as plt

def nelder_mead(f, x_start, step=[0.1,0.1,0.1], error=10e-6, max_attempts=20, 
    max_iter=0, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    # init
    dim = len(x_start)
    prev_best = f(x_start)
    attempts_num = 0
    response = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step[i]
        score = f(x)
        response.append([x, score])

    # simplex iter
    iters = 0
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('iterations')
    ax.set_ylabel('function value')
    while 1:
        # order
        response.sort(key=lambda x: x[1])
        best = response[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            print 'maximum number of iterations has been reached'
            return response[0]
        iters += 1
        ax.semilogy(iters,best,'o',color='k')
        plt.pause(0.001)
        # plt.show()
        # break after max_attempts iterations with no improvement
        time.sleep(0.001)
        print 'The best value so far is:', best

        if prev_best - best > error:
            attempts_num = 0
            prev_best = best
        else:
            attempts_num += 1
        if attempts_num >= max_attempts:
            print 'number of iterations:',iters
            print 'current optimal solution within max_attempts is {}'.format(response[0][1])
            print response[0]
            return response[0]

        # centroid
        x0 = [0.0] * dim
        for tup in response[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(response)-1)

        # reflection
        xr = x0 + alpha*(x0 - response[-1][0])
        rscore = f(xr)
        if response[0][1] <= rscore < response[-2][1]:
            print 'reflection is running ...'
            del response[-1]
            response.append([xr, rscore])
            continue

        # expansion
        if rscore < response[0][1]:
            print 'expansion is running ...'
            xe = x0 + gamma*(xr - x0)
            escore = f(xe)
            if escore < rscore:
                del response[-1]
                response.append([xe, escore])
                continue
            else:
                del response[-1]
                response.append([xr, rscore])
                continue

        # contraction
        if rscore >= response[-2][1]:
            print 'contraction is running ...'
            xc = x0 + rho*(response[-1][0]-x0)
            cscore = f(xc)
            if cscore < response[-1][1]:
                del response[-1]
                response.append([xc, cscore])
                continue

        # shrink
        print 'shrink is running ...'
        x1 = response[0][0]

        nresponse = [[x1, f(x1)]]
        for tup in response[1:]:
            xi = x1 + sigma*(tup[0] - x1)
            score = f(xi)
            nresponse.append([xi, score])
        response = nresponse
    # plt.pause(10)
    plt.ioff()
    plt.show()

def f_target(xparameter,a=2,b=None):
    x = xparameter[0]
    y = xparameter[1]
    z = xparameter[2]
    return (x-53.568)**2+(y-1.327)**2+(z+82.327)**2

if __name__ == "__main__":
    nelder_mead(f_target, np.array([0.0,0.0,0.0]), step=[1.0,1.0,1.0], error=10e-6, max_attempts=20,
    max_iter=0, alpha=1.0, gamma=2.0, rho= 0.5, sigma=0.5)