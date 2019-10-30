import numpy as np
import copy
import math
import pylesim.quantsim as sim
from pylesim import envelopes as env
import matplotlib.pyplot as plt
import random
from scipy.linalg import expm
import time
from pylesim.envelopes import Envelope
import pylesim.plotting as pyplt

# gloable parameters
T1, T2 = float('inf'), float('inf')  # coherence time
nonlin = -0.24  # GHz
T = np.arange(0.0,21,1)  # evolution time [ns]
length = (T[-1]-T[0])
uwx = 2*np.pi*env.cosine(t0=(T[0]+T[-1])/2.0, w=length, amp=1.0/length, phase=0.0)
uwy = 0.5/(nonlin*2*np.pi)*env.deriv(uwx)  # drag

def loss(x):
    return (x[0]-1.0)**2+(x[1]-2.0)**2

def drag_correct_3level(variable=[0.5,0.0], t0=10, w=20.0, step=0.1, phase=0, bloch=False):
    alpha, deltaf = variable[0], variable[1]
    q0 = sim.Qubit3(T1=T1, T2=T2, nonlin=nonlin)
    system = sim.QuantumSystem([q0])
    psi0 = np.array([1+0j,0,0])
    def cosinex(t0=t0, amp=1.0/w, w=w, phase=phase, df=deltaf):
        def timeFunc(t):
            x = t - t0
            return amp * 0.5 * (1+np.cos(2*np.pi*x/w)) * ((x+w/2.)>0) * ((-x+w/2.)>0) * \
                np.exp(1j*phase- 2j*np.pi*df*x)
        return Envelope(timeFunc, start=t0-w/2, end=t0+w/2)
    T0 = np.arange(0,w+step,step)
    cosine = cosinex(t0=t0, amp=1.0/w, w=w, phase=phase, df=deltaf)
    q0.uw = cosine-1j*(alpha/(2*np.pi*nonlin))*env.deriv(cosine)
    rhos0 = system.simulate(psi0, T0, method='fast')
    rhos = rhos0[:,0:2,0:2]
    P0, P1, P2 = rhos[:, 0][:, 0], rhos[:, 1][:, 1], rhos0[:, 2][:, 2]
    if bloch:
        pyplt.plotTrajectory(rhos, state=1, labels=True)
        plt.show()
    return 1 - np.real(P1[-1])

def tranfser_2state(variable=[1/100.0*np.sqrt(2),1/100.0], step=0.1, w=100, phase=0, deltaf=0, plot=False):
    t0 = w/2.0
    amp1, amp2 = variable[0], variable[1]
    q0 = sim.Qubit3(T1=T1, T2=T2, nonlin=nonlin)
    system = sim.QuantumSystem([q0])
    psi0 = np.array([1+0j,0,0])
    def cosinex(t0=t0, amp=None, w=w, phase=phase, df=deltaf):
        def timeFunc(t):
            x = t - t0
            return amp * 0.5 * (1+np.cos(2*np.pi*x/w)) * ((x+w/2.)>0) * ((-x+w/2.)>0) * \
                np.exp(1j*phase- 2j*np.pi*df*x)
        return Envelope(timeFunc, start=t0-w/2, end=t0+w/2)
    T0 = np.arange(0,w+step,step)
    q0.uw = cosinex(t0=t0, amp=amp1, w=w, phase=phase, df=deltaf)+cosinex(t0=t0, amp=amp2, w=w, phase=phase, df=nonlin)
    rhos = system.simulate(psi0, T0, method='fast')
    P0, P1, P2 = rhos[:, 0][:, 0], rhos[:, 1][:, 1], rhos[:, 2][:, 2]
    if plot:
        plt.plot(T0,P0,label='P0')
        plt.plot(T0,P1,label='P1')
        plt.plot(T0,P2,label='P2')
        plt.legend()
        plt.show()
    return 1 - np.real(P2[-1])

def phi_2level(x=None, t=T, U_target=np.array([[0,-1j],[-1j,0]]), t0=0, N=20, lamda=np.sqrt(2), delta=nonlin*2*np.pi, plot=False):
    dt = t[1]-t[0]
    sigmaI = np.diag([1,1])
    Hx = 0.5*np.array([[0,1],[1,0]])
    Hy = 0.5*np.array([[0,-1j],[1j,0]])
    Ht = []
    xt,yt = np.zeros(len(t)),np.zeros(len(t))
    scope1 = range(len(x))
    for i in scope1:
        xti = x[i]*(t0-dt/2.0<=t)*(t<t0+dt/2.0)
        t0 += dt
        xt = xt+xti
    Ut = sigmaI
    for loop_t in range(len(t)):
        Ht_loop_t = xt[loop_t]*Hx+yt[loop_t]*Hy
        Ut_loop_t = expm(-1j*Ht_loop_t*dt)
        Ut = np.dot(Ut, Ut_loop_t)
        Ht.append(Ht_loop_t)
    TrU = np.trace(np.dot(U_target.T.conj(),Ut))
    phi = np.real(0.25*TrU*TrU.conj())
    # print 'phi =', phi
    if plot:
        plt.plot(t,xt)
        plt.show()
    return phi 

def phi_cost_xy(x=None, t=T, U_target=np.array([[0,-1j,0],[-1j,0,0],[0,0,0]]), t0=0, N=20, lamda=np.sqrt(2), delta=nonlin*2*np.pi, plot=False):
    dt = t[1]-t[0]
    sigmaI = np.diag([1,1,1])
    projection = np.diag([1,1,0])
    Hd = np.diag([0,0,delta])
    Hx = 0.5*np.array([[0,1,0],[1,0,lamda],[0,lamda,0]])
    Hy = 0.5*np.array([[0,-1j,0],[1j,0,-1j*lamda],[0,1j*lamda,0]])
    Hz = np.diag([0,1,2])
    Ht = []
    xt,yt,zt = np.zeros(len(t)),np.zeros(len(t)),np.zeros(len(t))
    scope1, scope2 = range(len(x)/2), range(len(x)/2,len(x))
    for i,j in zip(scope1,scope2):
        xti = x[i]*(t0-dt/2.0<=t)*(t<t0+dt/2.0)
        ytj = x[j]*(t0-dt/2.0<=t)*(t<t0+dt/2.0)
        t0 += dt
        xt, yt = xt+xti, yt+ytj
    Ut = sigmaI
    for loop_t in range(len(t)):
        Ht_loop_t = Hd+xt[loop_t]*Hx+yt[loop_t]*Hy+zt[loop_t]*Hz
        Ut_loop_t = expm(-1j*Ht_loop_t*dt)
        Ut = np.dot(Ut, Ut_loop_t)
        Ht.append(Ht_loop_t)
    Ut_sub = np.dot(projection,Ut)
    U_target_new_sub = np.dot(projection,U_target.T.conj())
    TrU = np.trace(np.dot(U_target_new_sub,Ut_sub))
    phi = np.real(0.25*TrU*TrU.conj())
    # print 'phi =', phi
    if plot:
        plt.plot(t,xt,t,yt)
        plt.show()
    return 1-phi

def phi_cost_xyz(x=None, t=T, U_target=np.array([[0,-1j,0],[-1j,0,0],[0,0,0]]), t0=0, N=20, lamda=np.sqrt(2),delta=nonlin*2*np.pi, plot=False):
    dt = t[1]-t[0]
    sigmaI = np.diag([1,1,1])
    projection = np.diag([1,1,0])
    Hd = np.diag([0,0,delta])
    Hx = 0.5*np.array([[0,1,0],[1,0,lamda],[0,lamda,0]])
    Hy = 0.5*np.array([[0,-1j,0],[1j,0,-1j*lamda],[0,1j*lamda,0]])
    Hz = np.diag([0,1,2])
    Ht = []
    xt,yt,zt = np.zeros(len(t)),np.zeros(len(t)),np.zeros(len(t))
    scope1, scope2, scope3 = range(len(x)/3), range(len(x)/3,2*len(x)/3), range(2*len(x)/3,len(x))
    for i,j,k in zip(scope1,scope2,scope3):
        xti = x[i]*(t0<=t)*(t<t0+dt)
        ytj = x[j]*(t0<=t)*(t<t0+dt)
        ztk = x[k]*(t0<=t)*(t<t0+dt)
        t0 += dt
        xt, yt, zt = xt+xti, yt+ytj, zt+ztk
    Ut = sigmaI
    for loop_t in range(len(t)):
        Ht_loop_t = Hd+xt[loop_t]*Hx+yt[loop_t]*Hy+zt[loop_t]*Hz
        Ut_loop_t = expm(-1j*Ht_loop_t*dt)
        Ut = np.dot(Ut, Ut_loop_t)
        Ht.append(Ht_loop_t)
    Ut_sub = np.dot(projection,Ut)
    U_target_new_sub = np.dot(projection,U_target.T.conj())
    TrU = np.trace(np.dot(U_target_new_sub,Ut_sub))
    phi = np.real(0.25*TrU*TrU.conj())
    # print 'phi =', phi
    if plot:
        plt.plot(t,xt,t,yt,t,zt)
        plt.show()
    return 1-phi

def gradient_descense(f, x, step=0.01, num=10, adapt_step=False):
    alpha = 1e-5
    gradients = np.zeros(x.shape)
    gradient_new = []

    for loop_i in range(len(gradients)): 
        delta_vector = np.zeros(x.shape)
        delta_vector[loop_i] = alpha
        gradients[loop_i] = (f(x+delta_vector)-f(x-delta_vector)) / (2*alpha)
        if adapt_step:
            power_loop_i = []
            n = 1e3*np.random.rand(num)
            for loop_j in range(len(n)):
                xtest = 0.0001*np.ones(x.shape)
                xtest[loop_i] = random.sample(n, 1)[0]
                try:
                    if float(f(xtest)) != 0:
                        gradient_loop_j = (f(xtest+delta_vector)-f(xtest-delta_vector))/(2*alpha)
                        power_loop_j = (xtest[loop_i]*gradient_loop_j)/float(f(xtest))
                        power_loop_i.append(power_loop_j)
                except ValueError:
                    pass
            power_loop_i_new = float(np.mean(power_loop_i))
            gradient_loop_i_new = np.sign(gradients[loop_i])*np.sqrt(np.abs(gradients[loop_i]))**(1.0/power_loop_i_new)
            gradient_new.append(gradient_loop_i_new)

    if adapt_step:
        gradient_new = np.array(gradient_new)
        return x - gradient_new * step
    else:
        return x - gradients * step

def find_optimize(f, x=np.array([0.0,0.0]), maxloop=100, epsilon=1e-6, step=0.01, adapt_step=True, adjust_size=False, factor_up=2.0, factor_down=0.5, counts=0):
    for i in range(maxloop):
        counts += 1
        print '--------------------------------------'
        print 'counts =', counts
        print 'step =', step
        oldvalue = f(x)
        print 'oldvalue =', oldvalue
        x = gradient_descense(f, x, step=step, adapt_step=adapt_step)
        newvalue = f(x)
        print 'newvalue =', newvalue
        time.sleep(0.001) 
        if adjust_size:
            # step size is too large and needs to be reduced
            if newvalue - oldvalue > 0.0:
                print 'step size is large'
                step = step*factor_down
            # step size is too small, need to increase
            elif newvalue - oldvalue < 0.0 and np.abs(1-float(newvalue)/float(oldvalue)) < 0.1:
                print 'step size is small'
                step = step*factor_up
            else:
                print 'step size is normal'
        print ('x = {}'.format(x))
        print 'phi = {}, loss = {}'.format(1-newvalue,newvalue)
        if newvalue <= epsilon :
            break
    return x, newvalue

def nelder_mead(f, x_start, step=[0.1,0.1,0.1], error=10e-6, max_attempts=20, max_iter=0, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    # initial
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
    while 1:
        # order
        response.sort(key=lambda x: x[1])
        best = response[0][1]
        # break after max_iter
        if max_iter and iters >= max_iter:
            print 'maximum number of iterations has been reached'
            return response[0]
        iters += 1
        print 'counts =', iters
        # break after max_attempts iterations with no improvement
        time.sleep(0.001)
        print response[0]
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

if __name__ == '__main__':
    init = np.hstack([uwx(T),uwy(T)])
    #xnew = find_optimize(phi_cost_xy, x=init, maxloop=20, epsilon=1e-6, step=0.01, adapt_step=False, adjust_size=False)[0]
    xnew = nelder_mead(phi_cost_xy, init, step=0.1*np.ones(len(init)), error=10e-8, max_attempts=500, max_iter=1000)[0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(T,np.real(xnew[0:len(init)/2]),label='xt')
    ax.plot(T,np.real(xnew[len(init)/2:len(init)]),label='yt')
    ax.scatter(T,np.real(uwx(T)),label='initx')
    ax.scatter(T,np.real(uwy(T)),label='inity')
    ax.legend()
    plt.show()
