# -*- coding: utf-8 -*-

import numpy as np
import copy 

class Mesh:
    def __init__(self, nX, nY, length, width, fieldNames, value = 0):
        # grid divisions
        self.__xPoints = nX
        self.__yPoints = nY
    
        # geometry dimensions
        self.__length = length
        self.__width = width
        
        self.__fieldNames = np.unique(fieldNames)
        self.fieldMatrix = np.full(self.__matrixShape__(), value, dtype = float) 
        
    def addField(self, fieldNames):
        self.__fieldNames = np.append(self.__fieldNames,fieldNames)
        self.__fieldNames = np.unique(self.__fieldNames)
        
    def getFieldIndex(self, field):
        return list(self.__fieldNames).index(field)

    def __matrixShape__(self):
        return (len(self.__fieldNames), self.__xPoints , self.__yPoints)
    
    def setGridDivisions(self, nX, nY):
        self.__xPoints = nX
        self.__yPoints = nY
        
    def setPlateGeometry(self, Length, Width):
        self.__length = float(Length)
        self.__width = float(Width)

    def numberOfGridPoints(self):
        return self.__xPoints * self.__yPoints
    
    def gridShape(self):
        return (self.__xPoints, self.__yPoints)        
    
    def __calculateGridSize__(self):
        assert(self.__xPoints != 0 and self.__yPoints != 0)
        dx = self.__length / self.__xPoints
        dy= self.__width / self.__yPoints
        return (dx, dy)
            
    def generatePoints(self):
        assert(self.__xPoints != 0 and self.__yPoints != 0);
        gridSize = self.__calculateGridSize__()
        
        X = np.arange(0, self.__length, gridSize[0])
        Y = np.arange(0, self.__width, gridSize[1])
        
        return np.meshgrid(X, Y)
    
    def generateAnalyticalSolution(self):
        grid = self.generatePoints()
        analyticalSolution = np.zeros((self.__xPoints, self.__yPoints), dtype = float)
        for i in range(1, 100):
            nPI = np.pi*i
            nPIOverL = np.pi*i/(self.__length)
            analyticalSolution += (float(200)*(1 - np.power(-1, i))/(nPI*np.sinh(nPI)))*np.sin(nPIOverL*grid[1])*np.sinh(((grid[0])/self.__length)*nPI)
        return analyticalSolution
    
class BoundaryConditions:
    def __init__(self):
        self.boundaryDictionary = {"Boundary condition":[], "Value":[], "Variable":[], "Position index":[]}
        self.__boundaryVariablesValues = np.array([])
        self.__positionIndexArray = np.array([])
        
    def append(self, boundaryType, val, variable, posIndex):
        self.boundaryDictionary["Boundary condition"].append(boundaryType)
        self.boundaryDictionary["Value"].append(val)
        self.boundaryDictionary["Variable"].append(variable)
        self.boundaryDictionary["Position index"].append(posIndex)

    def __isHorizontalBoundary__(self, boundaryBegin, boundaryEnd):
        return (boundaryBegin[0] < boundaryEnd[0] and boundaryBegin[1] == boundaryEnd[1])

    def __isVerticalBoundary__(self, boundaryBegin, boundaryEnd):
        return (boundaryBegin[1] < boundaryEnd[1] and boundaryBegin[0] == boundaryEnd[0])

    def __smoothenCornerEffect__(self, mesh):
        xCorner = mesh.gridShape()[0] - 1
        yCorner = mesh.gridShape()[1] - 1
        
        mesh.fieldMatrix[:, 0, 0] = np.multiply(0.5, (mesh.fieldMatrix[:, 1, 0] + mesh.fieldMatrix[:, 0, 1]))
        mesh.fieldMatrix[:, 0, yCorner] = np.multiply(0.5, (mesh.fieldMatrix[:, 0, yCorner - 1] + mesh.fieldMatrix[:, 1, yCorner]))
        mesh.fieldMatrix[:, xCorner, 0] = np.multiply(0.5, (mesh.fieldMatrix[:, (xCorner - 1), 0] + mesh.fieldMatrix[:, xCorner, 1]))
        mesh.fieldMatrix[:, xCorner, yCorner] = np.multiply(0.5, (mesh.fieldMatrix[:, (xCorner - 1), yCorner] + mesh.fieldMatrix[:, xCorner, yCorner - 1]))
        
    def setBoundaryValues(self, mesh):
        boundaryIndexes = self.boundaryDictionary["Position index"]
        values = self.boundaryDictionary["Value"]

        for idx in range(len(boundaryIndexes)):
            index = boundaryIndexes[idx]
            val = values[idx]
            if self.__isHorizontalBoundary__(index[0], index[1]):
                verticalPos = index[0][1]
                for jdx in range(index[0][0], index[1][0]+1):
                    mesh.fieldMatrix[:,jdx, verticalPos] = val
            elif self.__isVerticalBoundary__(index[0], index[1]):
                horizontalPos = index[0][0]
                for jdx in range(index[0][1], index[1][1]+1):
                    mesh.fieldMatrix[:,horizontalPos, jdx] = val
        self.__smoothenCornerEffect__(mesh)
                                    
class Solver:
    def __init__(self, mesh, iterations, tolerance = float(1e-6)):
        self.__iterations = iterations
        self.__tolerance = tolerance
        
    def solve(self):
        unknownCount = mesh.__matrixShape__()[1] - 1
        iterationCount = 0
        error = float(1.0)
        lastIterationField = np.zeros_like(mesh.fieldMatrix)
        print("iteration", "   error")
        while(error > self.__tolerance):
            iterationCount += 1
            if iterationCount > self.__iterations: break
            for i in range(1, unknownCount):
                for j in range(1, unknownCount):
                    mesh.fieldMatrix[:, i, j] = np.multiply(0.25, (mesh.fieldMatrix[:, i-1 , j] + mesh.fieldMatrix[:, i+1, j] + mesh.fieldMatrix[:, i, j-1] + mesh.fieldMatrix[:, i, j+1]))
            error = np.max(abs(mesh.fieldMatrix[:, 1:unknownCount, 1:unknownCount] - lastIterationField[:, 1:unknownCount, 1:unknownCount]))
            lastIterationField = copy.deepcopy(mesh.fieldMatrix)
            print(iterationCount, error)

import matplotlib.pyplot as plt
class PostProcess:
    def __init__(self, mesh, colormap = 'jet', interpolation = 'none'):
        self.__cmap = colormap
        self.__interpolation = interpolation
        
    def plot(self, field):
        plt.imshow(mesh.fieldMatrix[mesh.getFieldIndex(field)], cmap = self.__cmap, interpolation = self.__interpolation)    

gridPtsX = 10
gridPtsY = 10

length = 10
width = 10

mesh = Mesh(gridPtsX, gridPtsY, length, width, "T")

boundary = BoundaryConditions()
boundary.append("Dirichlet", 0, "T", ((0,0),(0,gridPtsY - 1)))
boundary.append("Dirichlet", 100, "T", ((0,gridPtsY - 1),(gridPtsX - 1,gridPtsY - 1)))
boundary.append("Dirichlet", 0, "T", ((gridPtsX - 1,0),(gridPtsX - 1,gridPtsY - 1)))
boundary.append("Dirichlet", 0, "T", ((0,0),(gridPtsX - 1,0)))
boundary.setBoundaryValues(mesh)

laplacian = Solver(mesh, 1000, 1e-7)
laplacian.solve()

analyticalSolution = mesh.generateAnalyticalSolution()
plt.imshow(analyticalSolution, 'hot', interpolation = 'gaussian')
postProcess = PostProcess(mesh, 'hot', interpolation = 'gaussian')
postProcess.plot("T")