"""
<name>Simon</name>
<description>Simulator in Orange.</description>
<icon>icons/Simon.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>2100</priority>
"""

#import orange
from OWWidget import *
from OWGraph import *
from qt import *
from qttable import *
from qtcanvas import *
import OWGUI, OWGraphTools
from math import *


class Ball:
    def __init__(self):
        self.x = 800
        self.y = 500
        self.r = 15
        
def normalizeAngle(x):
    x = x % (2*pi)
    if x > pi:
        x = x - 2*pi
    return x

class Robot(object):
    def __init__(self):
        self.x = self.y = 200
        self.orientation = 0
        
        self.maxSpeed = 50
        self.speed_left = self.speed_right = 0
        self.cameraHalfAngle = 0.2

    def __setattr__(self, name, value):
        if name in ["speed_left", "speed_right"]:
            value = max(-self.maxSpeed, min(self.maxSpeed, value))
        if name == "orientation":
            value = normalizeAngle(value)
        object.__setattr__(self, name, value)#self.__dict__[name] = value

    def computeBallArea(self):
        r = self.ball.r
        balldist, ballangle = self.ball_distance, abs(self.ball_angle)
        half_phi = asin(r / balldist)
        # half_phi is the perceived angle between the center of the ball and its surface
        # ballangle is the (unoriented, positive) angle between the ball center and the robot's orientation
        # self.cameraHalfAngle is a half of the angle of the robots camera
        
        totarea = pi * r**2  *  balldist**-2
        # we see all of it
        if self.cameraHalfAngle > ballangle + half_phi:
            return totarea
        # we don't see it at all
        if self.cameraHalfAngle < ballangle - half_phi:
            return 0
        # we don't see the center
        elif self.cameraHalfAngle < ballangle:
            import random
            return random.random()
        # we see the center
        else:
            import random
            return random.random()
        
    def __getattr__(self, name):
        if name == "ball_distance":
            return hypot(self.y - self.ball.y, self.x - self.ball.x)
        elif name == "ball_angle":
            return normalizeAngle(atan2(self.ball.y - self.y, self.ball.x - self.x) - self.orientation)
        elif name == "ball_area":
            return self.computeBallArea()
            
        return object.__getattr__(self, name)#self.__dict__[name]
        
    def tick(self):
        delta_dist = (self.speed_left + self.speed_right) / (self.maxSpeed/2.)
        delta_angle = (self.speed_left - self.speed_right) / (self.maxSpeed*6.)

        self.orientation = (self.orientation + delta_angle) % (2*pi)
        self.x += delta_dist * cos(self.orientation)
        self.y += delta_dist * sin(self.orientation)

        
class OWSimon(OWWidget):
    settingsList=["logFrequency"]
    callbackDeposit=[]
    
    def __init__(self, parent=None, signalManager=None, name="Simon"):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs=[("Examples", ExampleTable, self.cdata)]
        self.outputs=[("Trace", ExampleTable)]
        
        self.loadSettings()
        self.maxDiff = 10
        self.logFrequency = 15

        self.robot = Robot()
        self.robot.ball = ball = Ball()
        self.clearLog()
        self.data = None

        box = OWGUI.widgetBox(self.controlArea, "Settings")
        OWGUI.spin(box, self, "logFrequency", 10, 100, 5, label="Sampling interval ", controlWidth = 40)
        OWGUI.spin(box, self, "robot.ball.r", 10, 100, 5, label="Ball size", controlWidth = 40, callback = self.updateBall)
        
        maxSpeed = self.robot.maxSpeed
        
        OWGUI.rubber(self.controlArea)

        vbox = self.controlArea #OWGUI.widgetBox(self.controlArea, "Simulation data")
        tableRows = ["Left wheel", "Right wheel", "Overall speed", "Rotation", "X", "Y", "orientation", "Ball distance", "Ball angle", "Ball area"]
        self.simulationData = QTable(len(tableRows), 1, vbox)
        self.simulationData.setTopMargin(0)
        self.simulationData.horizontalHeader().hide()
        vhead = self.simulationData.verticalHeader()
        for i, lab in enumerate(tableRows):
            vhead.setLabel(i, lab)
        vhead.setFixedWidth(90)
        
        OWGUI.separator(vbox)
        OWGUI.button(vbox, self, "Clear log", callback=self.clearLog)
        self.startButton = OWGUI.button(vbox, self, "Start", toggleButton = True, callback = self.startStop)
        OWGUI.button(vbox, self, "Send data", callback=self.sendData)
#        self.simulationData.adjustSize()
        
        OWGUI.separator(self.controlArea)
        self.playbackButton = OWGUI.button(vbox, self, "Playback", toggleButton = True, callback=self.playback)
        self.playbackButton.setEnabled(False)

        self.timer = QTimer(self)
        self.connect(self.timer, SIGNAL("timeout()"), self.moveRobot)

        self.circleX = [cos(2*pi*x/360.) for x in range(0, 361, 30)]
        self.circleY = [sin(2*pi*x/360.) for x in range(0, 361, 30)]
        
        self.box = QVBoxLayout(self.mainArea)
        self.graph = graph = OWGraph(self.mainArea)
        graph.setCanvasBackground(QColor(224, 224, 224))
        graph.enableXaxis(False)
        graph.enableYLaxis(False)
        graph.setAxisScale(QwtPlot.yLeft, 0, 1000, 100)
        graph.setAxisScale(QwtPlot.xBottom, 0, 1000, 100)
        
        curve = PolygonCurve(self.graph, QPen(QColor(0,0,0)), QBrush(QColor("white")))
        self.cameraRay = self.graph.insertCurve(curve)
        self.robotTrace = graph.addCurve("", Qt.black, QColor(160, 160, 160), 5, style = QwtCurve.Lines, symbol = QwtSymbol.Ellipse, xData = self.xTrace, yData = self.yTrace)
        self.robotKey = graph.addCurve("", Qt.black, Qt.black, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [], yData = [])
        curve = PolygonCurve(self.graph, QPen(QColor("darkGreen")), QBrush(QColor("darkGreen")))
        self.ballKey = self.graph.insertCurve(curve)
        
        self.box.addWidget(graph)
        self.updateBall()
        self.updateGraph()
        self.updateData()
        self.sendData()
        

    def startStop(self):
        if self.startButton.isOn():
            self.logCountdown = self.logFrequency
            self.timer.start(10)
            self.playbackButton.setEnabled(False)
        else:
            self.timer.stop()
            self.playbackButton.setEnabled(True)
            self.sendData()

    def playback(self):
        if self.playbackButton.isOn():
            if self.startButton.isOn():
                self.startButton.setDown(False)
                self.startStop()
            self.startButton.setEnabled(False)
            self.clearLog()
            self.logCountdown = self.logFrequency
            self.playPoint = 0
            ex = self.data[0]
            self.robot.x, self.robot.y, self.robot.orientation = ex["x"], ex["y"], ex["orientation"]
            self.robot.speed_left, self.robot.speed_right = ex["speed left"], ex["speed right"]
            self.timer.start(10)
        else:
            self.timer.stop()
            self.startButton.setEnabled(True)
            self.sendData()
        
    def stopPlayback(self):
        self.timer.stop()
        self.startButton.setEnabled(True)
        self.playbackButton.setDown(False)
        self.sendData()
        

    def logPoint(self):
        robot = self.robot
        self.log.append([len(self.log),
                         robot.speed_left, robot.speed_right,
                         robot.x, robot.y, robot.orientation/pi*180,
                         robot.ball_distance, robot.ball_angle/pi*180, robot.ball_area])
                      
    def clearLog(self):
        self.xTrace, self.yTrace = [], []
        self.log = []
        self.logPoint()
        
    def sendData(self):
        domain = orange.Domain([orange.FloatVariable(n) for n in ["time", "speed left", "speed right", "overall speed", "rotation", "x", "y", "orientation", "ball distance", "ball angle", "ball area", "ball distance t+1", "ball angle t+1", "delta ball distance", "delta ball angle"]], None)
        table = orange.ExampleTable(domain)
        for i, (time, left, right, x, y, orientation, ball_dist, ball_angle, ball_area) in enumerate(self.log[:-1]):
            table.append([time, left, right, (left+right)/2., left-right, x, y, orientation, ball_dist, ball_angle, ball_area, self.log[i+1][-3], self.log[i+1][-2], self.log[i+1][-3]-ball_dist, self.log[i+1][-2]-ball_angle])
        self.send("Trace", table)
        
    def updateBall(self):
        ball = self.robot.ball
        self.graph.curve(self.ballKey).setData([ball.x+ball.r*i for i in self.circleX], [ball.y+ball.r*i for i in self.circleY])
        self.graph.update()

    def updateGraph(self):
        robot = self.robot
        x, y, o, cha = robot.x, robot.y, robot.orientation, robot.cameraHalfAngle
        self.graph.curve(self.robotKey).setData([x+10*i for i in self.circleX], [y+10*i for i in self.circleY])
        self.graph.setCurveData(self.cameraRay, [x, x+1000*cos(o+cha), x+1000*cos(o-cha), x], [y, y+1000*sin(o+cha), y+1000*sin(o-cha), y])
        self.graph.curve(self.robotTrace).setData(self.xTrace, self.yTrace)
        self.graph.update()
        
    
    def updateData(self):
        robot = self.robot
        for i, (dec, d) in enumerate([(0, robot.speed_left), (0, robot.speed_right), (1, (robot.speed_left + robot.speed_right) / 2), (0, robot.speed_left - robot.speed_right),
                               (0, robot.x), (0, robot.y), (2, robot.orientation/pi*180), (2, robot.ball_distance), (2, robot.ball_angle/pi*180), (4, robot.ball_area)]):
            self.simulationData.setText(i, 0, ("  %%.%if" % dec) % d)
        #self.simulationData.setColumnWidth(1, 60)
        self.simulationData.adjustColumn(1)
        self.simulationData.adjustSize()
        
    def moveRobot(self):
        robot = self.robot
        
        robot.tick()

        self.updateGraph()
        self.updateData()
        
        self.logCountdown -= 1
        if not self.logCountdown:
            self.logCountdown = self.logFrequency
            self.xTrace.append(robot.x)
            self.yTrace.append(robot.y)
            if self.playbackButton.isOn():
                self.playPoint += 1
                if self.playPoint < len(self.data):
                    self.robot.speed_left, self.robot.speed_right = self.data[self.playPoint]["speed left"], self.data[self.playPoint]["speed right"]
                else:
                    self.stopPlayback()
            else:
                self.logPoint()


    def cdata(self, data):
        self.stopPlayback()
        self.data = data
        if self.data:
            if "speed left" in data.domain and "speed right" in data.domain:
                self.data = data
                self.error()
            else:
                self.data = None
                self.error("data does not contain the attributes with wheel speeds")
        else:
            self.data = None
            self.error()
            
        self.playbackButton.setEnabled(bool(self.data))
        
    def keyPressEvent(self, e):
        robot = self.robot
        # separate control for wheels
        if e.text() == "q":
            robot.speed_left += 1
        elif str(e.text()) in "zy":
            robot.speed_left -= 1
        elif e.text() == "o":
            robot.speed_right += 1
        elif str(e.text()) == "m":
            robot.speed_right -= 1

        # faster/slower
        elif e.text() == "t":
            robot.speed_left += 1
            robot.speed_right += 1
        elif e.text() == "b":
            robot.speed_left -= 1
            robot.speed_right -= 1
            
        # left/right
        elif e.text() == "k":
            if robot.speed_left - robot.speed_right == 1:
                robot.speed_left -= 1
            elif robot.speed_left - robot.speed_right > -self.maxDiff:
                robot.speed_left -= 1
                robot.speed_right += 1
        elif e.text() == "h":
            if robot.speed_left - robot.speed_right == -1:
                robot.speed_left += 1
            if robot.speed_left - robot.speed_right < self.maxDiff:
                robot.speed_left += 1
                robot.speed_right -= 1

        # faster/slower without turning
        elif e.text() == "r":
            robot.speed_left = robot.speed_right = (robot.speed_left + robot.speed_right + 2) / 2
        elif e.text() == "v":
            robot.speed_left = robot.speed_right = (robot.speed_left + robot.speed_right - 2) / 2

        # start/stop motion
        elif e.text() == " ":
            self.startButton.toggle()
            self.startStop()
        else:
            OWWidget.keyPressEvent(self, e)
            
        self.updateData()
            
import sys
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWSimon()
    app.setMainWidget(w)
    w.show()
    app.exec_loop()
