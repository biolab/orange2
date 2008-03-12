import orangeom
from orangeom import SOMLearner, SOMClassifier, SOMMap
import numpy

def SOMLearner(examples=None, weightID=0, parameters=None, **argkw):
    learner=orangeom.SOMLearner()
    learner.__dict__.update(argkw)
    if parameters:
        defparams={"alpha":1.0, "radius":5, "iterations":1000}
        learner.alpha=[]
        learner.radius=[]
        learner.iterations=[]
        learner.steps=len(parameters)
        for param in parameters:
            defparams.update(param)
            for k,v in defparams.items():
                learner.__dict__[k].append(v)
    if examples:
        return learner(examples, weightID)
    return learner

def getUMat(som):
    dim1=som.xDim*2-1
    dim2=som.yDim*2-1

    a=numpy.zeros((dim1, dim2))
    if som.topology==orangeom.SOMLearner.HexagonalTopology:
        return __fillHex(a, som)
    else:
        return __fillRect(a, som)

def __fillHex(array, som):
    d={}
    for n in som.nodes:
        d[(n.x,n.y)]=n
    check=lambda x,y:x>=0 and x<(som.xDim*2-1) and y>=0 and y<(som.yDim*2-1)
    dx=[1,0,-1]
    dy=[0,1, 1]
    for i in range(0,som.xDim*2,2):
        for j in range(0,som.yDim*2,2):
            for ddx,ddy in zip(dx,dy):
                if check(i+ddx, j+ddy):
                    array[i+ddx][j+ddy]=d[(i/2, j/2)].getDistance(d[(i/2+ddx, j/2+ddy)].referenceExample)
    dx=[1,-1,0,-1, 0, 1]
    dy=[0, 0,1, 1,-1,-1]
    for i in range(0,som.xDim*2,2):
        for j in range(0,som.yDim*2,2):
            l=[array[i+ddx,j+ddy] for ddx,ddy in zip(dx,dy) if check(i+ddx, j+ddy)]
            array[i][j]=sum(l)/len(l)
    return array

def __fillRect(array, som):
    d={}
    for n in som.nodes:
        d[(n.x,n.y)]=n
    check=lambda x,y:x>=0 and x<som.xDim*2-1 and y>=0 and y<som.yDim*2-1
    dx=[1,0,1]
    dy=[0,1,1]
    for i in range(0,som.xDim*2,2):
        for j in range(0,som.yDim*2,2):
            for ddx,ddy in zip(dx,dy):
                if check(i+ddx, j+ddy):
                    array[i+ddx][j+ddy]=d[(i/2, j/2)].getDistance(d[(i/2+ddx, j/2+ddy)].referenceExample)
    dx=[1,-1, 0,0,1,-1,-1, 1]
    dy=[0, 0,-1,1,1,-1, 1,-1]
    for i in range(0,som.xDim*2,2):
        for j in range(0,som.yDim*2,2):
            l=[array[i+ddx,j+ddy] for ddx,ddy in zip(dx,dy) if check(i+ddx, j+ddy)]
            array[i][j]=sum(l)/len(l)
    return array


