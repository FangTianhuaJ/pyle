import numpy as np
from numpy import cos, sin, pi, matrix

def rotationMatrix(theta):
    return np.array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])

def xy2cycles(x,y):
    if len(x)!= len(y):
        raise Exception('Lengths of x and y must be equal.')
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    data = np.vstack((x,y))
    #Make sure the first point has x>0
    if data[0,0]<0:
        data = -data
    #Rotate so that first point has angle=0
    angle0 = np.arctan(data[1,0]/data[0,0])
    data = np.dot(rotationMatrix(-angle0),data)
    phase = []
    prevAngle = 0.0
    for i in range(len(x)):
        rotated = np.dot(rotationMatrix(prevAngle),np.array([x[i],y[i]]))
        prevAngle += np.arctan(rotated[1]/rotated[0])
        phase.append(prevAngle)
    #Convert to cycles
    return np.array(phase)/(2*np.pi)

def weightedLinearFit(x,y,w):
    """Computes a linear fit with weights using Vandermonde matrix inversion"""
    #Compute sums
    y0 = np.sum(y*w)
    y1 = np.sum(x*y*w)
    M00 = np.sum(w)
    M10 = np.sum(w*x)
    M01 = M10
    M11 = np.sum(w*x*x)
    #build Vandermode matrix
    M = np.matrix(np.array([[M00,M01],[M10,M11]]))
    #Invert matrix and convert it to a numpy array
    Minv = np.asarray(M.I)
    Y = np.array([[y0],[y1]])
    #Multiply inversted matrix by 
    P = np.dot(Minv,Y)
    return P
    
def weightedLinearFitFixedSlope(x,y,p1,w):
    """Finds p0 in the equation y=p0+p1*x
    """
    p0 = np.sum((y-p1*x)*w)/np.sum(w)
    return p0