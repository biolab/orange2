"""
<name>Linear Results</name>
<description>Linear Results is an Orange Widget that shows results of logistic regression learner. </description>
<category>Classification</category>
<icon>icons/Unknown.png</icon>
<priority>210</priority>
"""
#
# OWLin_Results.py
#
# Lin_Results is an Orange Widget that
# shows regression coefficients (betas) 
# fitted by any fitter that returns a classifier with beta values
# 
# 
#

from qttable import *
from OWWidget import *
import math, orngLR

class OWLin_Results(OWWidget):
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self,
        parent, signalManager,
        "Linear Results",
        """Linear Results is an Orange Widget that
shows regression coeficients of attributes estimated in a regression learner
(logistic regression, linear regression)
""",
        TRUE,
        FALSE)
        
        self.notTargetClassIndex = 1
        self.TargetClassIndex = 0

        #load settings
        self.loadSettings()
        
        self.data=None

        #list inputs and outputs
        self.inputs=[("Classifier", orange.Classifier, self.classifier, 1), ("Examples", ExampleTable, self.cdata, 1), ("Target Class Value", int, self.ctarget, 1)]
        
        #GUI
        
        #give mainArea a layout
        self.layout=QVBoxLayout(self.mainArea)
        #add your components here
        self.table=QTable(self.mainArea)
        self.table.setSelectionMode(QTable.NoSelection) #hell, if i set this, swapRows() doesn't work - weird. and if i do not set this, the user can edit the fields
        self.layout.add(self.table)
        self.table.setNumCols(6)
        self.table.setNumRows(0)
        self.lresult_names = ["beta", "st. error", "wald Z", "P (chi^2)", "OR = exp(beta)"]
        self.topheader=self.table.horizontalHeader()
        self.topheader.setLabel(0,"Attribute")
        self.table.adjustColumn(0)
        self.connect(self.topheader,SIGNAL("pressed(int)"),self.sort)
        self.sortby=-1
        for i in range(5):
            self.topheader.setLabel(1+i,self.lresult_names[i])  
            self.table.adjustColumn(1+i)     
        self.resize(600,200) 
        
        #add controls to self.controlArea widget 
        #connect controls to appropriate functions
        

    def sort(self,col):
        "Sort the column col"
        if col==self.sortby:
            samecolumn=TRUE
        else:
            samecolumn=FALSE
        if not samecolumn:
            if col<2: 
                #table.sortColumn sort in alphabetical order, so it can only by used for the attribute and C/D columns
                self.table.sortColumn(col,TRUE,TRUE)
            else:            
            #we must do our own sorting
            #should take advantage of QTable's swapRows() method
            #so how about a simple bubblesort?
                for i in range(0,self.table.numRows()-1):
                    x=float(str(self.table.text(i,col)))
                    for j in range(i+1,self.table.numRows()):
                        y=float(str(self.table.text(j,col)))
                        if x>y:
#                            print "swap!"
                            self.swapRows(i,j)
                            x=y
        else:   #reversing selection is faster
            for i in range(0,self.table.numRows()/2):
                self.swapRows(i,self.table.numRows()-i-1)
        self.sortby=col
        self.repaint()
        
    def swapRows(self,i,j):
        for k in range (0, self.table.numCols()):
#            print "swapping ", self.table.text(i,k), "with ", self.table.text(j,k)
            temp=self.table.text(i,k)
            self.table.setText(i,k,self.table.text(j,k))
            self.table.setText(j,k,temp)
#            self.table.swapCells(i,k,j,k)  #why in heavens name doesn't this work for any other than the current column? 
            #Maybe sorting must be enabled? But how is one supposed to do his own sort?
 
    def classifier(self,classifier):
        if classifier==None:
            self.classifier=None
            self.table.setNumRows(0)
        else:
            self.classifier=classifier
        self.showResults()

    def cdata(self,data):
        self.data = data
        if data == None and self.classifier and isinstance(self.classifier, orange.BayesClassifier): 
            self.table.setNumRows(0)
        self.showResults()

    def showResults(self):
        # thisd subroutine computes standard error of estimated beta /for the time being use it only with discrete data
        def err(e, priorError, key, data):
            inf = 0.0
            sume = e[0]+e[1]
            for d in data:
                if d[at]==key:
                    inf += (e[0]*e[1]/sume/sume)
            inf = max(inf, 0.00000001)
            var = 1/inf - priorError*priorError
            return (math.sqrt(var))

        if self.classifier == None:
            return

        if isinstance(self.classifier, orange.LogRegClassifier):
            if self.notTargetClassIndex == 1 or self.notTargetClassIndex == cl.continuizedDomain.classVar[1]:
                mult = -1
            else:
                mult = 1

            self.table.setNumRows(len(self.classifier.continuizedDomain.attributes)+1)
            self.table.setText(0,0,"Constant")        
            self.table.setText(0,1,str(mult*round(self.classifier.beta[0],2)))        
            self.table.setText(0,2,str(round(self.classifier.beta_se[0],2)))        
            self.table.setText(0,3,str(round(self.classifier.wald_Z[0],2)))        
            self.table.setText(0,4,str(abs(round(self.classifier.P[0],2))))
                                  
            for i in range(len(self.classifier.continuizedDomain.attributes)):
                self.table.setText(i+1,0,str(self.classifier.continuizedDomain.attributes[i].name))        
                self.table.setText(i+1,1,str(mult*round(self.classifier.beta[i+1],2)))        
                self.table.setText(i+1,2,str(round(self.classifier.beta_se[i+1],2)))        
                self.table.setText(i+1,3,str(round(self.classifier.wald_Z[i+1],2)))        
                self.table.setText(i+1,4,str(abs(round(self.classifier.P[i+1],2))))
                self.table.setText(i+1,5,str(round(math.exp(self.classifier.beta[i+1]),2)))
        elif isinstance(self.classifier, orange.BayesClassifier) and self.data and self.data.domain == self.classifier.domain:
            numrows = 1
            for at in self.classifier.domain.attributes:
                if at.varType == orange.VarTypes.Discrete:
                    numrows += len(at.values)
                else:
                    numrows +=1
            cl = self.classifier
            classVal = self.classifier.domain.classVar.values
            
            self.table.setNumRows(numrows)
            
            dist1 = max(0.00000001, cl.distribution[classVal[self.notTargetClassIndex]])
            dist0 = max(0.00000001, cl.distribution[classVal[self.TargetClassIndex]])
            prior = dist0/dist1
            logprior = math.log(prior)
            sumd = dist1+dist0
            priorError = math.sqrt(1/((dist1*dist0/sumd/sumd)*len(self.data)))
            

            self.table.setText(0,0,"Constant")        
            self.table.setText(0,1,str(round(logprior,2)))        
            self.table.setText(0,2,str(round(priorError,2)))        
            self.table.setText(0,3,str(round(logprior/priorError,2)))        
            self.table.setText(0,4,str(abs(round(orngLR.lchisqprob(math.pow(logprior/priorError,2),1),2))))

            row = 1                                  
            for at in self.classifier.domain.attributes:
                if at.varType == orange.VarTypes.Discrete:
                    for cd in cl.conditionalDistributions[at].keys():
                        self.table.setText(row,0,str(at.name+"="+cd))
                        conditional0 = max(cl.conditionalDistributions[at][cd][classVal[self.TargetClassIndex]], 0.00000001)
                        conditional1 = max(cl.conditionalDistributions[at][cd][classVal[self.notTargetClassIndex]], 0.00000001)
                        beta = math.log(conditional0/conditional1/prior)
                        se = err(cl.conditionalDistributions[at][cd], priorError, cd, self.data) # standar error of beta 
                        
                        self.table.setText(row,1,str(round(beta,2)))        
                        self.table.setText(row,2,str(round(se,2)))        
                        self.table.setText(row,3,str(round(beta/se,2)))        
                        self.table.setText(row,4,str(abs(round(orngLR.lchisqprob(math.pow(beta/se,2),1),2))))
                        self.table.setText(row,5,str(round(math.exp(beta),2)))
                        row += 1
                else:
                    self.table.setText(row,0,str(at.name))
                    self.table.setText(row, 1, str("Not discrete attribute!"))
                    row+=1
        else:
            return

        self.table.adjustColumn(0)        

    def ctarget(self, target):
        self.TargetClassIndex = target
        if self.TargetClassIndex == 1:
            self.notTargetClassIndex = 0
        else:
            self.notTargetClassIndex = 1
        if self.classifier and self.classifier.domain and self.TargetClassIndex == self.self.classifier.classVar[1]:
            self.notTargetClassIndex = self.cl.domain.classVar[0]
        elif self.classifier and self.classifier.domain:
            self.notTargetClassIndex = self.cl.domain.classVar[1]
            
#test widget appearance        
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRank()
    a.setMainWidget(ow)
#here you can test setting some stuff
    ow.show()
    a.exec_loop()
    
    #save settings 
    ow.saveSettings()

