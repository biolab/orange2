#
# Module: Orange Dendrograms
# --------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
#
# Purpose: Dendrogram rendering for hierarchical clustering.
#
# Project initiated on 2003/05/09
#
# Notes: 
#   Classes GDHClustering and GHClustering are subclasses of the orngCluster's
#   DHClustering and HClustering, but they have an additional method dendrogram()
#   which outputs a piddle canvas with the image of the dendrogram.
#
# ChangeLog:
#   2003/05/13:
#       - line_size
#   2003/07/18:
#       - support for other canvases 
#   2003/09/12:
#       - cluster identification
#   2003/09/18:
#       - coloring, line width: red color indicates positive interactions,
#         blue color indicates negative interactions. Horizontal bars indicate
#         inter-cluster interactions, vertical bars indicate intra-cluster
#         interactions
# 


import orngCluster
import Tkinter, ImageTk
import piddle, piddlePIL, math


def _colorize(cc):
    bluefunc = lambda cc:1.0 / (1.0 + math.exp(-10*(cc-0.6)))
    redfunc = lambda cc:1.0 / (1.0 + math.exp(10*(cc-0.5)))
    cblu = bluefunc(cc)
    cred = redfunc(cc)
    cgre =  1 - pow(redfunc(cc+0.1),2) - bluefunc(cc-0.15)
    return piddle.Color(cred,cgre,cblu)

class DendrogramPlot:    
    def dendrogram(self,labels,width = 500, height = None, margin = 20, hook = 40, line_size = 2.0, cluster_colors = [], canvas = None, line_width = 1):
        # prevent divide-by-zero...
        if len(labels) < 2:
            return canvas

        ## ADJUST DIMENSIONS ###        

        if canvas == None:
            tcanvas = piddlePIL.PILCanvas()
        else:
            tcanvas = canvas

        if height==None:
            # compute the height
            lineskip = int(line_size*tcanvas.fontHeight()+1)
            height = int(2.0*margin + lineskip*(len(labels)-1) + tcanvas.fontHeight()+1)
        else:
            # compute lineskip
            lineskip = (height - 2.0*margin - tcanvas.fontHeight()) / (len(labels)-1)
        maxlabel = 0.0
        for s in labels:
            maxlabel = max(maxlabel,tcanvas.stringWidth(s))

        if canvas == None:
            canvas = piddlePIL.PILCanvas(size=(width,height))

        ### EXTRACT THE DENDROGRAM ###
        
        vlines = []               # vertical lines (cluster)
        hlines = []               # horizontal lines (clusters)
        origins = [0.0]*self.n    # text positions
        xpositions = [0.0]*self.n # cluster x positions (height)
        ypositions = [0]*self.n   # cluster y positions (average element)
        attcolors = [0]*self.n
        y = margin
        for i in range(len(labels)):
            # self.order identifies the label at a particular row
            ypositions[self.n-self.order[i]] = y
            y += lineskip
        xpositions.append("sentry")
        ypositions.append("sentry")
        p = self.n
        height = 0.0
        for i in range(self.n-1):
                height += self.height[i]
                if len(cluster_colors) == self.n-1:
                    coloV = _colorize(cluster_colors[i][0])
                    coloH = _colorize(cluster_colors[i][1])
                else:
                    # no color information
                    coloH = coloV = piddle.black
                vlines.append((height,ypositions[p+self.merging[i][0]],ypositions[p+self.merging[i][1]],coloV))
                avg = 0.0
                for j in self.merging[i]:
                    # record text origins
                    v = ypositions[p+j]
                    if j < 0:
                        origins[-1-j] = height
                        attcolors[-1-j] = coloH
                    else:
                        # create the cluster lines
                        hlines.append((v,xpositions[p+j],height,coloH))
                    avg += v             
                # recompute the average height of new cluster
                ypositions.append(0.5*avg)
                xpositions.append(height)

        ### DRAWING ###
                
        offset = width-maxlabel-hook-2*margin
        hs = (offset-margin)/height         # height scaling
        halfline = canvas.fontAscent()/2.0

        # print names
        y = margin
        for i in range(len(labels)):
            # self.order identifies the label at a particular row
            idx = self.order[i]-1
            x = offset-hs*origins[idx]
            canvas.drawString(labels[idx], hook+x, y+halfline)
            # draw the hook
            canvas.drawLine(x,y,x+hook*0.8,y,attcolors[idx],width=line_width)
            y += lineskip

        # print lines
        for (y,x1,x2,colo) in hlines:
            canvas.drawLine(offset-x1*hs,y,offset-x2*hs,y,colo,width=line_width)
        vlines.reverse() # smaller clusters are more interesting
        for (x,y1,y2,colo) in vlines:
            canvas.drawLine(offset-x*hs,y1,offset-x*hs,y2,colo,width=line_width)
            
        canvas.flush()
        return canvas

class GDHClustering(DendrogramPlot,orngCluster.DHClustering):
    pass

class GHClustering(DendrogramPlot,orngCluster.HClustering):
    pass


class ViewImage:
    class PicCanvas:
        def draw(self):
            self.canvas = Tkinter.Canvas(self.master,width=self.pic.width(),height=self.pic.height())
            self.canvas.pack()
            self.canvas.create_image(0,0,anchor=Tkinter.NW,image=self.pic)
        
        def __init__(self,master,pic):
            self.master = master
            self.pic = pic
            self.draw()

    def __init__(self,pic):
        self.root = Tkinter.Tk()
        self.root.title("ViewImage")
        self.canvas = self.PicCanvas(self.root,ImageTk.PhotoImage(pic))
        self.root.mainloop()

class ViewCanvas:
    def __init__(self,canvas):
        self.root = Tkinter.Tk()
        self.root.title("ViewCanvas")
        self.canvas = canvas
        self.root.mainloop()

if __name__== "__main__":
    import orange, orngInteract
    import warnings
    warnings.filterwarnings("ignore",module="orngInteract")

    tab = orange.ExampleTable('zoo.tab')
    im = orngInteract.InteractionMatrix(tab)
    (diss,labels) = im.exportDissimilarityMatrix()
    c = GDHClustering(diss)
    # canvas = c.dendrogram(labels)
    canvas = c.dendrogram(labels,line_size=1.8,cluster_colors=im.getClusterAverages(c),line_width=3)
    canvas.getImage().save("c_zoo.png")
    ViewImage(canvas.getImage())
