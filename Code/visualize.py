##
#Visualizing Point Cloud Data
##
import sys
import vtk
from numpy import random
filename = "data1OGD.pcd"
dataPath1 = "data/oakland_part3_am_rf.node_features"
dataPath2 = "data/oakland_part3_an_rf.node_features"

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        # Setup the colors array
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("colors")
        

    def addPoint(self, point, color):
        # Add the self.colors we created to the self.colors array
        self.colors.InsertNextTupleValue(color)
        # print self.colors
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
             
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        
    def readFile(self):
        self.points = []
        r = [255, 0, 0]
        g = [0, 255, 0]
        y = [0, 255, 255]
        w = [255, 255, 255]
        p = [255, 255, 0]

        colorCode = dict({1004: g, 1100: r, 1103: w, 1200: y, 1400: p})
        # 1004 - veg, 1100 - wire, 1103 - pole, ground - 1200, facade - 1400 
        for i, line in enumerate((open(filename, 'r'))):
            item = line.rstrip() # strip off newline and any other trailing whitespace
            if i>=4:
                pos = [float(x) for x in item.split(" ")[0:3]]
                color = int(item.split(" ")[4])
                # self.points.append(pos)
                # print colorCode[color]
                pointCloud.addPoint(pos, colorCode[color])
        print self.colors
        self.vtkPolyData.GetPointData().SetScalars(self.colors)
            

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = dataPath2


pointCloud = VtkPointCloud()
pointCloud.readFile()

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(pointCloud.vtkActor)
renderer.SetBackground(.1, .1, .1)
renderer.ResetCamera()

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Begin Interaction
renderWindow.Render()
renderWindowInteractor.Start()
