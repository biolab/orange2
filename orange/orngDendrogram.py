#
# Module: Orange Dendrograms
# --------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org) 
# (Copyright (C)2004 Aleks Jakulin)
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
#       - coloring, line width
#       - dissimilarity matrix visualization
#   2004/02/16:
#       - latent variable visualization
#       - black&white dissimilarity matrix
#   2004/03/17:
#       - general matrix visualization
#   2004/12/23:
#       - various improvements (matrix drawing, dendrograms)


import orngCluster
import Tkinter, ImageTk
import piddle, piddlePIL, math

_defaultfont = 'Courier'

def _colorize0(cc):
    #bluefunc = lambda cc:1.0 / (1.0 + math.exp(-10*(cc-0.6)))
    #redfunc = lambda cc:1.0 / (1.0 + math.exp(10*(cc-0.5)))
    cc = max(-50,cc)
    cc = min(50,cc)
    bluefunc = lambda cc:1.0 / (1.0 + math.exp(-10*(cc-0.6)))
    redfunc = lambda cc:1.0 / (1.0 + math.exp(10*(cc-0.5)))
    cblu = bluefunc(cc)
    cred = redfunc(cc)
    cgre =  1 - pow(redfunc(cc+0.1),2) - bluefunc(cc-0.15)
    #cgre =  1 - pow(redfunc(cc+0.2),2) - bluefunc(cc-0.3)
    return piddle.Color(cred,cgre,cblu)

def _colorize1(cc):
    bluefunc = lambda cc:1.0 / (1.0 + math.exp(-10*(cc-0.6)))
    redfunc = lambda cc:1.0 / (1.0 + math.exp(10*(cc-0.5)))
    cred = bluefunc(cc)
    cgre = redfunc(cc)
    cblu =  1 - pow(redfunc(cc+0.1),2) - bluefunc(cc-0.15)
    return piddle.Color(cred,cgre,cblu)

def _blackwhite(cc):
    v = 1.0-(1.0/max(1e-6,cc))
    return piddle.Color(v,v,v)

def _bw3(cc):
    v = cc
    return piddle.Color(v,v,v)

def _bw2(cc):
    v = abs(cc)*1.8
    return piddle.Color(v,v,v)

class DendrogramPlot:    
    def dendrogram(self,labels,width = 500, height = None, margin = 20, hook = 40, line_size = 2.0, cluster_colors = [], canvas = None, line_width = 1,color_mode=0, incremental_height=1, matr = [], g_lines=0, additional_labels = [], additional_matr=[], add_tags =[], adwidth=1.0, plot_gains = 0, gains = [], gain_width = 70, plot_ints = 0, right_align = 0, im = None):
        # prevent divide-by-zero...
        if len(labels) < 2:
            return canvas

        if plot_ints or (plot_gains and len(gains) == 0):
            # the gains have been proposed but not calculated
            if len(gains) > 0:
                warnings.warn('Ignoring the given gains: need to refer to the interaction matrix.')
            gains = [] # discretized gains
            mv = 1.0/max(1e-6,max(im.gains))
            for v in im.gains:
                gains.append(v*mv) # normalize with respect to the best attribute from correlation analysis.
                
        if len(gains) > 0:
            # fix the widths
            gain_l = []
            for i in xrange(self.n):
                gain_l.append(gains[i])
    
        max_intlen = 0.0
        if plot_ints:
            # include the interactions between all pairs
            intlist = []
            for i in xrange(self.n-1):
                idx1 = self.order[i]-1
                idx2 = self.order[i+1]-1
                ig = im.way3[(min(idx1,idx2),max(idx1,idx2),-1)].InteractionInformation()
                if gains[idx1] < gains[idx2]:
                    idx1,idx2 = idx2,idx1
                if ig > 0:
                    max_intlen = max(max_intlen,gain_width*ig*mv)
                    gain_l[idx1] += ig*mv # possibly new maximum width
                    intlist.append((idx1,ig*mv,0.0))
                else:
                    intlist.append((idx2,ig*mv,1.0))
        else:
            intlist = []

        ## ADJUST DIMENSIONS ###        
        if canvas == None:
            tcanvas = piddlePIL.PILCanvas()
        else:
            tcanvas = canvas

        normal = piddle.Font(face=_defaultfont)
        bold = piddle.Font(face=_defaultfont,bold=1)
        if height==None:
            # compute the height
            lineskip = int(line_size*tcanvas.fontHeight(normal)+1)
            lines = len(labels)-1
            if len(additional_labels) > 0:
                lines += 1+len(additional_labels)
            height = int(2.0*margin + lineskip*(lines) + tcanvas.fontHeight(normal)+1)
        else:
            # compute lineskip
            lineskip = (height - 2.0*margin - tcanvas.fontHeight(normal)) / (len(labels)-1)
        maxlabel = 0.0
        spacew = tcanvas.stringWidth("   ",font=normal)
        swids = []
        for i in xrange(len(labels)):
            swid = tcanvas.stringWidth(labels[i],font=normal)
            swids.append(swid + spacew)
            if len(gains) > 0:
                assert(len(gains)==len(labels))
                swid += spacew + gain_l[i]*gain_width
            maxlabel = max(maxlabel,swid)
        maxswid = max(swids)

        if canvas == None:
            canvas = piddlePIL.PILCanvas(size=(width,height))

        if len(matr)>0:
            block = lineskip/2-1
        else:
            block = 0

        _colorize = _color_picker(color_mode)

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
        # bottom-up construction
        height = 0.0
        displacement = 0.0
        for i in range(self.n-1):
            if len(cluster_colors) == self.n-1:
                coloV = _colorize(cluster_colors[i][0])
                coloH = _colorize(cluster_colors[i][1])
            else:
                # no color information
                coloH = coloV = piddle.black
            # obtain the height
            if incremental_height:
                nheight = min(xpositions[p+self.merging[i][0]],xpositions[p+self.merging[i][1]])
                nheight -= self.height[i]
                displacement = min(displacement,nheight)
            else:
                height -= self.height[i]
                nheight = height

            vlines.append((nheight,ypositions[p+self.merging[i][0]],ypositions[p+self.merging[i][1]],coloV))
            avg = 0.0
            for j in self.merging[i]:
                # record text origins
                v = ypositions[p+j]
                if j < 0:
                    origins[-1-j] = nheight
                    attcolors[-1-j] = coloH
                else:
                    # create the cluster lines
                    hlines.append((v,xpositions[p+j],nheight,coloH))
                avg += v             
            # recompute the average height of new cluster
            ypositions.append(0.5*avg)
            xpositions.append(nheight)            
        #print displacement
        ### DRAWING ###            
                
        offset = width-maxlabel-hook-2*margin
        if len(matr)>0:
            offset -= 2*(len(matr[0]))*adwidth*block + 2*block # correct the right-hand side
        hs = (offset-margin)/(height-displacement)               # height scaling
        if incremental_height:
            hs = -hs
        halfline = canvas.fontAscent(font=normal)/2.0

        # print line-guides
        if g_lines and len(matr)==len(labels):
            colo = piddle.Color(0.9,0.9,0.9)
            y = margin
            s = len(matr[0])
            sx1 = width-margin-block
            sx2 = width-margin-2*(len(matr[0]))*adwidth*block-block
            canvas.drawLine(sx1,y-block-1,sx2,y-block-1,colo,width=1)
            x2 = width-margin-block
            for i in range(len(labels)):
                idx = self.order[i]-1
                x1 = offset+hook-hs*(origins[idx])+tcanvas.stringWidth(labels[idx],font=normal)+4
                canvas.drawLine(x1,y,sx2,y,colo,width=1)
                canvas.drawLine(sx1,y+block+1,sx2,y+block+1,colo,width=1)
                y += lineskip
            if len(additional_labels)>0:
                y += lineskip
                canvas.drawLine(sx1,y-block-1,sx2,y-block-1,colo,width=1)
                for i in range(len(additional_labels)):
                    x1 = offset+hook+5+tcanvas.stringWidth(additional_labels[i],font=normal)+4
                    canvas.drawLine(x1,y,sx2,y,colo,width=1)
                    canvas.drawLine(sx1,y+block+1,sx2,y+block+1,colo,width=1)
                    y += lineskip
            # vertical guides
            for i in range(len(matr[0])+1):
                x = width-margin-(2*(i)*adwidth*block)-block
                canvas.drawLine(x,margin-block,x,y-lineskip+block+1,colo,width=1)

        # print names
        y = margin
        for i in range(len(labels)):
            # self.order identifies the label at a particular row
            idx = self.order[i]-1
            x = offset-hs*(origins[idx])
            if labels[idx][0] == '*':
                canvas.drawString(labels[idx][1:], hook+x, y+halfline,font=bold)
            else:
                canvas.drawString(labels[idx], hook+x, y+halfline,font=normal)
            # draw the hook
            canvas.drawLine(x,y,x+hook*0.8,y,attcolors[idx],width=line_width)
            GSERIF = 1.2*line_width
            MULT = 1.2*line_width
            XMULT = 0.6*line_width
            XSERIF = 0.6*line_width
            SWIDTH = 1
            if len(gains) > 0 :
                # draw the gain line
                if right_align:
                    orig = width-margin-gain_width*gains[idx]-max_intlen
                else:
                    orig = hook+x+swids[idx]
                if gain_width*gains[idx] >= 2.0*MULT:
                    canvas.drawLine(orig+MULT,y,orig+gain_width*gains[idx]-MULT,y,piddle.black,width=MULT) # actual line
                canvas.drawLine(orig,y,orig+gain_width*gains[idx],y,piddle.black,width=SWIDTH) # thin line
                canvas.drawLine(orig,y-GSERIF,orig,y+GSERIF,piddle.black,width=SWIDTH) #serif 1
                canvas.drawLine(orig+gain_width*gains[idx],y-GSERIF,orig+gain_width*gains[idx],y+GSERIF,piddle.black,width=SWIDTH) #serif2
            if len(intlist) > 0 and i > 0:
                (qidx,widt,cc) = intlist[i-1]
                if right_align:
                    nx = width-margin-max_intlen
                else:
                    nx = offset-hs*(origins[qidx])+hook+swids[qidx]+gain_width*gains[qidx]
                ny = y-0.5*lineskip
                colo = _colorize(cc)
                if widt > 0:
                    disp = XMULT
                    seri = XSERIF
                else:
                    disp = -XMULT
                    seri = -XSERIF
                if abs(gain_width*widt) >= 2.0*XMULT:
                    canvas.drawLine(nx+disp,ny,nx+gain_width*widt-disp,ny,colo,width=XMULT) # actual line
                    canvas.drawLine(nx+gain_width*widt-seri,ny-seri,nx+gain_width*widt,ny,colo,width=SWIDTH) # arrowpoint 1
                    canvas.drawLine(nx+gain_width*widt-seri,ny+seri,nx+gain_width*widt,ny,colo,width=SWIDTH) # arrowpoint 2
                canvas.drawLine(nx,ny,nx+gain_width*widt,ny,colo,width=SWIDTH) # thin line
                canvas.drawLine(nx,ny-XSERIF,nx,ny+XSERIF,colo,width=SWIDTH) # serif 1            
            y += lineskip

        for i in range(len(additional_labels)):
            y += lineskip
            if additional_labels[i][0] == '*':
                canvas.drawString(additional_labels[i][1:], offset+hook+5, y+halfline,font=bold)
            else:
                canvas.drawString(additional_labels[i], offset+hook+5, y+halfline,font=normal)

        y += lineskip*1.5
        for i in range(len(add_tags)):
            wi = tcanvas.stringWidth(add_tags[i],font=normal)
            x = width-margin-2*(len(add_tags)-i-0.5)*adwidth*block - block - wi/2
            canvas.drawString(add_tags[i], x, y+halfline,font=bold)

        # print lines
        for (y,x1,x2,colo) in hlines:
            canvas.drawLine(offset-(x1)*hs,y,offset-(x2)*hs,y,colo,width=line_width)
        vlines.reverse() # smaller clusters are more interesting
        for (x,y1,y2,colo) in vlines:
            canvas.drawLine(offset-(x)*hs,y1,offset-(x)*hs,y2,colo,width=line_width)

        ### MATRIX RENDERING ###
        if len(matr)==len(labels):
            y = margin
            for i in range(len(labels)):
#                print labels[i],matr[i]
                idx = self.order[i]-1
                mm = matr[idx]
                for j in range(len(mm)):
                    # self.order identifies the label at a particular row
                    x = width-margin-2*(len(mm)-j-0.5)*adwidth*block - block 
                    v = 1-mm[j]
                    #if v < 254.0/255.0:
                    colo = piddle.Color(v,v,v)
                    canvas.drawRect(x-adwidth*block+1,y-block,x+adwidth*block-1,y+block,edgeColor=colo,fillColor=colo)
                y += lineskip
            for i in range(len(additional_matr)):
                y += lineskip
                mm = additional_matr[i]
                for j in range(len(mm)):
                    x = width-margin-2*(len(mm)-j-0.5)*adwidth*block - block 
                    v = 1-mm[j]
                    colo = piddle.Color(v,v,v)
                    canvas.drawRect(x-adwidth*block+1,y-block,x+adwidth*block-1,y+block,edgeColor=colo,fillColor=colo)
            
        canvas.flush()
        return canvas

    def matrix(self,labels, diss, margin = 10, hook = 10, block = None, line_size = 2.0, att_colors = [], canvas = None,color_mode=0,diagonal=0):
        # prevent divide-by-zero...
        if len(labels) < 2:
            return canvas

        ## ADJUST DIMENSIONS ###        

        if canvas == None:
            tcanvas = piddlePIL.PILCanvas()
        else:
            tcanvas = canvas

        normal = piddle.Font(face=_defaultfont)
        bold = piddle.Font(face=_defaultfont,bold=1)

        def pickfont(st):
            if st[0]=='*':
                return (bold,'%s'%st[1:])
            else:
                return (normal,'%s'%st)

        # compute the height
        lineskip = int(line_size*tcanvas.fontHeight(normal)+1)
        labellen = []
        for s in labels:
            (myfont,st) = pickfont(s)
            labellen.append(tcanvas.stringWidth(st,font=myfont))
        maxlabel = max(labellen)
        width = height = int(1 + 2.0*margin + hook + maxlabel + lineskip*(len(labels)) + tcanvas.fontHeight(normal))

        if block == None:
            block = lineskip/2-1

        if canvas == None:
            canvas = piddlePIL.PILCanvas(size=(width,height))

        _colorize = _color_picker(color_mode)

        ### DRAWING ###
        font = normal
                
        offset = maxlabel+margin
        halfline = canvas.fontAscent(normal)/2.0
        # print names
        for i in range(len(labels)):
            # self.order identifies the label at a particular row
            idx = self.order[i]-1
            x = offset - labellen[idx]
            y = offset + lineskip*(i+1)
            # horizontal
            (myfont,xst) = pickfont(labels[idx])
            if not diagonal or i > 0 or len(att_colors)>0:
                canvas.drawString(xst, x, y+halfline,font = myfont)
            y2 = offset + lineskip*(i+1)
            # vertical
            if diagonal:
                if len(att_colors)>0:
                    canvas.drawString(xst, y+block+halfline-lineskip+hook, y2+block-hook-lineskip, angle=90,font=myfont)
                elif i < len(labels)-1:
                    canvas.drawString(xst, y+block+halfline-lineskip+hook, y2+block-hook, angle=90,font=myfont)
            elif not diagonal:
                canvas.drawString(xst, y+block+halfline-lineskip+hook, offset+lineskip-block-hook, angle=90,font=myfont)
            for j in range(i):
                idx2 = self.order[j]-1
                colo = _colorize(diss[max(idx,idx2)-1][min(idx,idx2)])
                x = offset+hook+lineskip*(j)+block
                y = offset+lineskip*(i+1)
                canvas.drawRect(x-block,y-block,x+block,y+block,edgeColor=colo,fillColor=colo)
                if not diagonal:
                    x = offset+hook+lineskip*(i)+block
                    y = offset+lineskip*(j+1)
                    canvas.drawRect(x-block,y-block,x+block,y+block,edgeColor=colo,fillColor=colo)
            if len(att_colors) > 0:
                # render the gain
                x = offset+hook+lineskip*(i)+block
                y = offset+lineskip*(i+1)
                colo = _colorize(att_colors[idx])
                canvas.drawRect(x-block,y-block,x+block,y+block,edgeColor=colo,fillColor=colo)
                
        canvas.flush()
        return canvas

def Matrix(diss = [], hlabels=[], vlabels=[], sizing = [], margin = 10, hook = 10, block = None, line_size = 2.0, color_mode=0, sizing2 = [], canvas = None, multiplier = 1.0):
    # prevent divide-by-zero...
    if len(hlabels) < 2:
        return canvas

    ## ADJUST DIMENSIONS ###        

    if canvas == None:
        tcanvas = piddlePIL.PILCanvas()
    else:
        tcanvas = canvas

    normal = piddle.Font(face=_defaultfont)
    bold = piddle.Font(face=_defaultfont,bold=1)

    square = 0
    if len(diss) > 0:
        yd = len(diss)
        if len(diss) == len(diss[0]):
            square = 1
            xd = yd
        else:
            xd = len(diss[0])
    else:
        yd = len(sizing)
        if len(sizing) == len(sizing[0]):
            square = 1
            xd = yd
        else:
            xd = len(sizing[0])

    if len(hlabels) == 0:
        hlabels = ["" for i in xrange(yd)]

    if len(vlabels) == 0:
        if square:
            vlabels = hlabels # vertical labels...
        else:
            vlabels = ["" for i in xrange(xd)]

    # compute the height
    lineskip = int(line_size*tcanvas.fontHeight(normal)+1)
    labellen = [tcanvas.stringWidth(s,font=normal) for s in hlabels]
    vlabellen = [tcanvas.stringWidth(s,font=normal) for s in vlabels]
    maxlabelx = max(labellen)
    maxlabely = max(vlabellen)
    width = int(1 + 2.0*margin + hook + maxlabelx + lineskip*(len(vlabels)) + tcanvas.fontHeight(normal))
    height = int(1 + 2.0*margin + hook + maxlabely + lineskip*(len(hlabels)) + tcanvas.fontHeight(normal))

    if block == None:
        block = lineskip/2-1

    if canvas == None:
        canvas = piddlePIL.PILCanvas(size=(width,height))

    _colorize = _color_picker(color_mode)

    ### DRAWING ###
            
    offsetx = maxlabelx+margin
    offsety = maxlabely+margin
    halfline = canvas.fontAscent(normal)/2.0

    for i in range(len(vlabels)):
        x2 = offsetx + lineskip*(i) + hook
        y2 = offsety + halfline - hook
        # vertical
        if vlabels[i][0] == '*':
            canvas.drawString(vlabels[i][1:], x2+block+halfline, y2+block, angle=90,font=bold)
        else:
            canvas.drawString(vlabels[i], x2+block+halfline, y2+block, angle=90,font=normal)
        
    # print names
    for i in range(len(hlabels)):
        # self.order identifies the label at a particular row
        x = offsetx - labellen[i]
        y = offsety + lineskip*(i+1)+halfline
        canvas.drawString(hlabels[i], x, y,font=normal)
        for j in range(len(vlabels)):
            x = offsetx+hook+lineskip*(j)+block
            y = offsety+lineskip*(i+1)
            if len(sizing) == 0:
                ss = 1.0
            else:
                ss = min(1,sizing[i][j])
            ss *= multiplier
            if len(diss) > 0:
                colo = _colorize(diss[i][j])
                canvas.drawRect(x-ss*block,y-ss*block,x+ss*block,y+ss*block,edgeColor=colo,fillColor=colo,edgeWidth=0.5)
            if len(sizing2) > 0:
                ss = sizing2[i][j]
                ss *= multiplier
                canvas.drawRect(x-ss*block,y-ss*block,x+ss*block,y+ss*block,edgeColor=piddle.black,fillColor=None,edgeWidth=0.5)
    canvas.flush()
    return canvas

# test colors
##    y = 100
##    for x in xrange(256):
##        colo = _colorize(x/255.0)
##        canvas.drawRect(x*2,y-block,x*2+1,y+block,edgeColor=colo,fillColor=colo)
##    y = 150
##    for x in xrange(125,130):
##        colo = _colorize(x/255.0)
##        canvas.drawRect(x*2,y-block,x*2+1,y+block,edgeColor=colo,fillColor=colo)

# a more vertically oriented matrix, with vertical columns of density
def YDensityMatrix(diss = [], hlabels=[], vlabels=[], margin = 10, hook = 10, block = None,
                  line_size = 2.0, ysize = 100, ticklen = 3, color_mode=0, canvas = None):
    # prevent divide-by-zero...
    if len(hlabels) < 2:
        return canvas

    ## ADJUST DIMENSIONS ###        

    if canvas == None:
        tcanvas = piddlePIL.PILCanvas()
    else:
        tcanvas = canvas

    normal = piddle.Font(face=_defaultfont)
    bold = piddle.Font(face=_defaultfont,bold=1)

    if len(diss) > 0:
        yd = len(diss)
        xd = len(diss[0])

    if len(vlabels) == 0:
        vlabels = ["" for i in xrange(xd)]
    else:
        assert(xd==len(vlabels))

    if len(hlabels) < 2:
        hlabels = ["0.0","1.0"]

    dh = ysize/float(len(hlabels)-1)
    
    # compute the height
    lineskip = int(line_size*tcanvas.fontHeight(normal)+1)
    labellen = [tcanvas.stringWidth(s,font=normal) for s in hlabels]
    vlabellen = [tcanvas.stringWidth(s,font=normal) for s in vlabels]
    maxlabelx = max(labellen)
    maxlabely = max(vlabellen)
    width = int(1 + 2.0*margin + hook + maxlabelx + lineskip*(len(vlabels)) + tcanvas.fontHeight(normal))
    height = int(1 + 2.0*margin + hook + maxlabely +  ysize)

    if block == None:
        block = lineskip/2-1

    if canvas == None:
        canvas = piddlePIL.PILCanvas(size=(width,height))

    _colorize = _color_picker(color_mode)

    ### DRAWING ###
            
    offsetx = maxlabelx+margin
    offsety = maxlabely+margin
    halfline = canvas.fontAscent(normal)/2.0

    for i in xrange(len(vlabels)):
        x2 = offsetx + lineskip*(i) + hook
        y2 = offsety 
        # vertical
        if vlabels[i][0] == '*':
            canvas.drawString(vlabels[i][1:], x2+block+halfline, y2, angle=90,font=bold)
        else:
            canvas.drawString(vlabels[i], x2+block+halfline, y2, angle=90,font=normal)

    for i in xrange(len(hlabels)):
        x = offsetx - labellen[i]
        y = offsety + hook + i*dh
        # horizontal
        if hlabels[-i-1][0] == '*':
            canvas.drawString(hlabels[-i-1][1:], x, y + halfline ,font=bold)
        else:
            canvas.drawString(hlabels[-i-1], x,y + halfline, font=normal)
        # tick
        canvas.drawLine(offsetx-ticklen+hook,y,offsetx+hook,y,width=1)

    dy = ysize/float(len(diss))
    for i in xrange(len(diss)):
        for j in xrange(len(vlabels)):
            x = offsetx+hook+lineskip*(j)+block
            y = offsety+hook+i*dy
            colo = _colorize(diss[i][j])
            canvas.drawRect(x-block,y,x+block,y+dy,edgeColor=colo,fillColor=colo)
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


#
# Some color schemes follow...
#
##Apache-Style Software License for ColorBrewer software and ColorBrewer Color Schemes
##Version 1.1
##
##Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State University. All rights reserved.
##Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
##1. Redistributions as source code must retain the above copyright notice, this list of conditions and the following disclaimer.
##2. The end-user documentation included with the redistribution, if any, must include the following acknowledgment:
##This product includes color specifications and designs developed by Cynthia Brewer (http://colorbrewer.org/).
##Alternately, this acknowledgment may appear in the software itself, if and wherever such third-party acknowledgments normally appear.
##4. The name "ColorBrewer" must not be used to endorse or promote products derived from this software without prior written permission. For written permission, please 
##contact Cynthia Brewer at cbrewer@psu.edu.
##5. Products derived from this software may not be called "ColorBrewer", nor may "ColorBrewer" appear in their name, without prior written permission of Cynthia Brewer.
##
##THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
##MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CYNTHIA BREWER, MARK HARROWER, OR THE 
##PENNSYLVANIA STATE UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
##BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
##CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY 
##WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# this class interpolates between the 11 steps
class RdBu:
    def __init__(self,darken=0):
        profile = [[103,0,31],[178,24,43],[214,96,77],[244,165,130],[253,219,199],[255,255,255],[209,229,240],[146,197,222],[67,147,195],[33,102,172],[5,48,97]]
        profile.append(profile[-1]) # terminator...
        profile.append(profile[-1]) # terminator...
        self.LUT = []
        for i in xrange(255):
            a = i/254.0 # keep it even so that it's white in the middle
            b = a*(len(profile)-3)
            bi = int(b) # round down
            db = b-bi   # difference
            assert(db >= 0.0)
            if darken:
                idb = (1.0-db)/270.0
                db /= 270.0
            else:
                idb = (1.0-db)/254.0
                db /= 254.0
            rgb = [profile[bi][x]*idb+profile[bi+1][x]*db for x in xrange(3)]
            self.LUT.append(piddle.Color(rgb[0],rgb[1],rgb[2]))
                            
    def __call__(self, x):
        return self.LUT[int(round(max(0.0,min(1.0,x))*254.0))]

def _color_picker(color_mode):
    if color_mode==1:
        _colorize = _colorize1 # gregor
    elif color_mode==0:
        _colorize = _colorize0 # aleks
    elif color_mode==2:
        _colorize = _blackwhite # interaction matrices, etc., 1/color
    elif color_mode==3:
        _colorize = _bw2        # grayscale for dendrograms
    elif color_mode==4:
        _colorize = RdBu()
    elif color_mode==5:
        _colorize = RdBu(darken=1)
    elif color_mode==6:
        _colorize = _bw3        # plain grayscale
    return _colorize



if __name__== "__main__":
    import orange, orngInteract
    import warnings
    warnings.filterwarnings("ignore",module="orngInteract")

    tab = orange.ExampleTable('zoo.tab')
    im = orngInteract.InteractionMatrix(tab)
    (diss,labels) = im.exportDissimilarityMatrix()
    c = GDHClustering(diss)
    canvas = c.dendrogram(labels)
    canvas.getImage().save("c_zoo.png")
    ViewImage(canvas.getImage())

    (dissx,labels,gains) = im.exportDissimilarityMatrix(show_gains = 0, color_coding = 1, color_gains = 1)
    canvas = c.matrix(labels,dissx,att_colors = gains)
    canvas.getImage().save("m_zoo.png")
    ViewImage(canvas.getImage())
