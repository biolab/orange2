from __future__ import with_statement
import random, types

def getobjectname(x, default=""):
    if type(x)==types.StringType:
        return x
      
    for i in ["name", "shortDescription", "description", "func_doc", "func_name"]:
        if getattr(x, i, ""):
            return getattr(x, i)

    if hasattr(x, "__class__"):
        r = repr(x.__class__)
        if r[1:5]=="type":
            return str(x.__class__)[7:-2]
        elif r[1:6]=="class":
            return str(x.__class__)[8:-2]

    return default


def demangleExamples(x):
    if type(x)==types.TupleType:
        return x
    else:
        return x, 0
    

class BooleanCounter:
  def __init__(self, bits):
    self.bits = bits
    self.state = None

  def __iter__(self):
    if self.state:
        return self
    else:
        return BooleanCounter(self.bits)
    
  def next(self):
    if self.state:
      for bit in range(self.bits-1, -1, -1):
        self.state[bit] = (self.state[bit]+1) % 2
        if self.state[bit]:
          break
      else:
        self.state = None
    else:
      self.state = [0]*self.bits

    if not self.state:
        raise StopIteration, "BooleanCounter: counting finished"

    return self.state


class LimitedCounter:
  def __init__(self, limits):
    self.limits = limits
    self.state = None
    
  def __iter__(self):
    if self.state:
        return self
    else:
        return LimitedCounter(self.limits)

  def next(self):
    if self.state:
      i = len(self.limits)-1
      while (i>=0) and (self.state[i]==self.limits[i]-1):
        self.state[i] = 0
        i -= 1
      if i==-1:
        self.state = None
      else:
        self.state[i] += 1
    else:
      self.state = [0]*len(self.limits)
  
    if not self.state:
      raise StopIteration, "LimitedCounter: counting finished"

    return self.state


class MofNCounter:
    def __init__(self, m, n):
        if m > n:
            raise TypeError, "Number of selected items exceeds the number of items"
        
        self.state = None
        self.m = m
        self.n = n
        
    def __iter__(self):
        if self.state:
            return self
        else:
            return MofNCounter(self.m, self.n)
        
    def next(self):
        if self.state:
            m, n, state = self.m, self.n, self.state
            for place in range(m-1, -1, -1):
                if state[place] + m-1-place < n-1:
                    state[place] += 1
                    for place in range(place+1, m):
                        state[place] = state[place-1] + 1
                    break
            else:
                self.state = None
                raise StopIteration, "MofNCounter: counting finished"
        else:
            self.state = range(self.m)
            
        return self.state[:]
             
class NondecreasingCounter:
  def __init__(self, places):
    self.state=None
    self.subcounter=None
    self.places=places

  def __iter__(self):
    if self.state:
        return self
    else:
        return NondecreasingCounter(self.places)

  def next(self):
    if not self.subcounter:
      self.subcounter=BooleanCounter(self.places-1)
    if self.subcounter.next():
      self.state=[0]
      for add_one in self.subcounter.state:
        self.state.append(self.state[-1]+add_one)
    else:
      self.state=None
  
    if not self.state:
      raise StopIteration, "NondecreasingCounter: counting finished"

    return self.state


class CanonicFuncCounter:
  def __init__(self, places):
    self.places = places
    self.state = None

  def __iter__(self):
    if self.state:
        return self
    else:
        return CanonicFuncCounter(self.places)

  def next(self):
    if self.state:
      i = self.places-1
      while (i>0) and (self.state[i]==max(self.state[:i])+1):
        self.state[i] = 0
        i -= 1
      if i:
        self.state[i] += 1
      else:
        self.state=None
    else:
      self.state = [0]*self.places

    if not self.state:
      raise StopIteration, "CanonicFuncCounter: counting finished"
    
    return self.state



import random

class BestOnTheFly:
    def __init__(self, compare=cmp, seed = 0, callCompareOn1st = False):
        self.randomGenerator = random.Random(seed)
        self.compare=compare
        self.wins=0
        self.bestIndex, self.index = -1, -1
        self.best = None
        self.callCompareOn1st = callCompareOn1st

    def candidate(self, x):
        self.index += 1
        if not self.wins:
            self.best=x
            self.wins=1
            self.bestIndex=self.index
            return 1
        else:
            if self.callCompareOn1st:
                cmpr=self.compare(x[0], self.best[0])
            else:
                cmpr=self.compare(x, self.best)
            if cmpr>0:
                self.best=x
                self.wins=1
                self.bestIndex=self.index
                return 1
            elif cmpr==0:
                self.wins=self.wins+1
                if not self.randomGenerator.randint(0, self.wins-1):
                    self.best=x
                    self.bestIndex=self.index
                    return 1
        return 0

    def winner(self):
        return self.best

    def winnerIndex(self):
        if self.best is not None:
            return self.bestIndex
        else:
            return None

def selectBest(x, compare=cmp, seed = 0, callCompareOn1st = False):
    bs=BestOnTheFly(compare, seed, callCompareOn1st)
    for i in x:
        bs.candidate(i)
    return bs.winner()

def selectBestIndex(x, compare=cmp, seed = 0, callCompareOn1st = False):
    bs=BestOnTheFly(compare, seed, callCompareOn1st)
    for i in x:
        bs.candidate(i)
    return bs.winnerIndex()

def compare2_firstBigger(x, y):
    return cmp(x[0], y[0])

def compare2_firstSmaller(x, y):
    return -cmp(x[0], y[0])

def compare2_lastBigger(x, y):
    return cmp(x[-1], y[-1])

def compare2_lastSmaller(x, y):
    return -cmp(x[-1], y[-1])

def compare2_bigger(x, y):
    return cmp(x, y)

def compare2_smaller(x, y):
    return -cmp(x, y)


def frange(*argw):
    start, stop, step = 0.0, 1.0, 0.1
    if len(argw)==1:
        start=step=argw[0]
    elif len(argw)==2:
        stop, step = argw
    elif len(argw)==3:
        start, stop, step = argw
    elif len(argw)>3:
        raise AttributeError, "1-3 arguments expected"

    stop+=1e-10
    i=0
    res=[]
    while 1:
        f=start+i*step
        if f>stop:
            break
        res.append(f)
        i+=1
    return res


verbose = 0

def printVerbose(text, *verb):
    if len(verb) and verb[0] or verbose:
        print text

import sys

class ConsoleProgressBar(object):
    def __init__(self, title="", charwidth=40, step=1, output=sys.stderr):
        self.title = title + " "
        self.charwidth = charwidth
        self.step = step
        self.currstring = ""
        self.state = 0
        self.output = output

    def clear(self, i=-1):
        try:
            if hasattr(self.output, "isatty") and self.output.isatty():
                self.output.write("\b" * (i if i != -1 else len(self.currstring)))
            else:
                self.output.seek(-i if i != -1 else -len(self.currstring), 2)
        except Exception: ## If for some reason we failed 
            self.output.write("\n")

    def getstring(self):
        progchar = int(round(float(self.state) * (self.charwidth - 5) / 100.0))
        return self.title + "=" * (progchar) + ">" + " " * (self.charwidth - 5 - progchar) + "%3i" % int(round(self.state)) + "%"

    def printline(self, string):
        try:
            self.clear()
            self.output.write(string)
            self.output.flush()
        except Exception:
            pass
        self.currstring = string

    def __call__(self, newstate=None):
        if newstate == None:
            newstate = self.state + self.step
        if int(newstate) != int(self.state):
            self.state = newstate
            self.printline(self.getstring())
        else:
            self.state = newstate

    def finish(self):
        self.__call__(100)
        self.output.write("\n")
        
def progressBarMilestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])
        
class ColorPalette(object):
    def __init__(self, colors, gamma=None, overflow=(255, 255, 255), underflow=(255, 255, 255), unknown=(0, 0, 0)):
        self.colors = colors
        self.gammaFunc = lambda x, gamma:((math.exp(gamma*math.log(2*x-1)) if x > 0.5 else -math.exp(gamma*math.log(-2*x+1)) if x!=0.5 else 0.0)+1)/2.0
        self.gamma = gamma
        self.overflow = overflow
        self.underflow = underflow
        self.unknown = unknown

    def get_rgb(self, val, gamma=None):
        if val is None:
            return self.unknown
        gamma = self.gamma if gamma is None else gamma
        index = int(val * (len(self.colors) - 1))
        if val < 0.0:
            return self.underflow
        elif val > 1.0:
            return self.overflow
        elif index == len(self.colors) - 1:
            return tuple(self.colors[-1][i] for i in range(3)) # self.colors[-1].green(), self.colors[-1].blue())
        else:
            red1, green1, blue1 = [self.colors[index][i] for i in range(3)] #, self.colors[index].green(), self.colors[index].blue()
            red2, green2, blue2 = [self.colors[index + 1][i] for i in range(3)] #, self.colors[index + 1].green(), self.colors[index + 1].blue()
            x = val * (len(self.colors) - 1) - index
            if gamma is not None:
                x = self.gammaFunc(x, gamma)
            return [(c2 - c1) * x + c1 for c1, c2 in [(red1, red2), (green1, green2), (blue1, blue2)]]
        
    def __call__(self, val, gamma=None):
        return self.get_rgb(val, gamma)
    
import math

#from ColorPalette import ColorPalette
    
class GeneratorContextManager(object):
   def __init__(self, gen):
       self.gen = gen
   def __enter__(self):
       try:
           return self.gen.next()
       except StopIteration:
           raise RuntimeError("generator didn't yield")
   def __exit__(self, type, value, traceback):
       if type is None:
           try:
               self.gen.next()
           except StopIteration:
               return
           else:
               raise RuntimeError("generator didn't stop")
       else:
           try:
               self.gen.throw(type, value, traceback)
               raise RuntimeError("generator didn't stop after throw()")
           except StopIteration:
               return True
           except:
               # only re-raise if it's *not* the exception that was
               # passed to throw(), because __exit__() must not raise
               # an exception unless __exit__() itself failed.  But
               # throw() has to raise the exception to signal
               # propagation, so this fixes the impedance mismatch 
               # between the throw() protocol and the __exit__()
               # protocol.
               #
               if sys.exc_info()[1] is not value:
                   raise

def contextmanager(func):
    def helper(*args, **kwds):
        return GeneratorContextManager(func(*args, **kwds))
    return helper

from functools import wraps
def with_state(func):
    @wraps(func)
    def wrap(self, *args, **kwargs):
        with self.state(**kwargs):
            r = func(self, *args)
        return r
    return wrap

def with_gc_disabled(func):
    import gc
    def disabler():
        gc.disable()
        try:
            yield
        finally:
            gc.enable()
    @wraps(func)
    def wrapper(*args, **kwargs):
        with contextmanager(disabler)():
            return func(*args, **kwargs)
    return wrapper  

import numpy

class Renderer(object):
    render_state_attributes = ["font", "stroke_color", "fill_color", "render_hints", "transform", "gradient", "text_alignment"]
    
    ALIGN_LEFT, ALIGN_RIGHT, ALIGN_CENTER = range(3)
      
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.render_state = {}
        self.render_state["font"] = ("Times-Roman", 10)
        self.render_state["fill_color"] = (0, 0, 0)
        self.render_state["gradient"] = {}
        self.render_state["stroke_color"] = (0, 0, 0)
        self.render_state["stroke_width"] = 1
        self.render_state["text_alignment"] = self.ALIGN_LEFT
        self.render_state["transform"] = numpy.matrix(numpy.eye(3))
        self.render_state["view_transform"] = numpy.matrix(numpy.eye(3))
        self.render_state["render_hints"] = {}
        self.render_state_stack = []
        
    def font(self):
        return self.render_state["font"]
    
    def set_font(self, family, size):
        self.render_state["font"] = family, size

    def fill_color(self):
        return self.render_state["fill_color"]
    
    def set_fill_color(self, color):
        self.render_state["fill_color"] = color
        
    def set_gradient(self, gradient):
        self.render_state["gradient"] = gradient
        
    def gradient(self):
        return self.render_state["gradient"]
        
    def stroke_color(self):
        return self.render_state["stroke_color"]
    
    def set_stroke_color(self, color):
        self.render_state["stroke_color"] = color
    
    def stroke_width(self):
        return self.render_state["stroke_width"]
    
    def set_stroke_width(self, width):
        self.render_state["stroke_width"] = width
        
    def set_text_alignment(self, align):
        self.render_state["text_alignment"] = align
        
    def text_alignment(self):
        return self.render_state["text_alignment"]
        
    def transform(self):
        return self.render_state["transform"]
    
    def set_transform(self, transform):
        self.render_state["transform"] = transform
        
    def render_hints(self):
        return self.render_state["render_hints"]
    
    def set_render_hints(self, hints):
        self.render_state["render_hints"].update(hints)
    
    def save_render_state(self):
        import copy
        self.render_state_stack.append(copy.deepcopy(self.render_state))
    
    def restore_render_state(self):
        self.render_state = self.render_state_stack.pop(-1)
        
    def apply_transform(self, transform):
        self.render_state["transform"] = self.render_state["transform"] * transform
         
    def translate(self, x, y):
        transform = numpy.eye(3)
        transform[:, 2] = x, y, 1
        self.apply_transform(transform)
    
    def rotate(self, angle):
        angle *= 2 * math.pi / 360.0
        transform = numpy.eye(3)
        transform[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        self.apply_transform(transform)
    
    def scale(self, sx, sy):
        transform = numpy.eye(3)
        transform[(0, 1), (0, 1)] = sx, sy 
        self.apply_transform(transform)
        
    def skew(self, sx, sy):
        transform = numpy.eye(3)
        transform[(1, 0), (0, 1)] = numpy.array([sx, sy]) * 2 * math.pi / 360.0
        self.apply_transform(transform)
    
    def draw_line(self, sx, sy, ex, ey, **kwargs):
        raise NotImplemented
    
    def draw_lines(self, points, **kwargs):
        raise NotImplemented
    
    def draw_rect(self, x, y, w, h, **kwargs):
        raise NotImplemented
    
    def draw_polygon(self, vertices, **kwargs):
        raise NotImplemented

    def draw_arch(self, something, **kwargs):
        raise NotImplemented
    
    def draw_text(self, x, y, text, **kwargs):
        raise NotImplemented
    
    def string_size_hint(self, text, **kwargs):
        raise NotImpemented
    
    @contextmanager
    def state(self, **kwargs):
        self.save_render_state()
        for key, value in kwargs.items():
            if key in ["translate", "rotate", "scale", "skew"]:
                getattr(self, key)(*value)
            else:
                getattr(self, "set_" + key)(value)
        try:
            yield
        finally:
            self.restore_render_state()
            
    def save(self):
        raise NotImplemented
    
    def close(self, file):
        pass
    
class EPSRenderer(Renderer):
    EPS_DRAW_RECT = """/draw_rect 
{/h exch def /w exch def
 /y exch def /x exch def
 newpath
 x y moveto
 w 0 rlineto
 0 h neg rlineto
 w neg 0 rlineto
 closepath
} def"""
    
    EPS_SET_GRADIENT = """<< /PatternType 2
 /Shading
   << /ShadingType 2
      /ColorSpace /DeviceRGB
      /Coords [%f %f %f %f]
      /Function
      << /FunctionType 0
         /Domain [0 1]
         /Range [0 1 0 1 0 1]
         /BitsPerSample 8
         /Size [%i]
         /DataSource <%s>
      >>
   >>
>>
matrix
makepattern
/mypattern exch def
/Pattern setcolorspace
mypattern setcolor

"""

    EPS_SHOW_FUNCTIONS = """/center_align_show
{ dup stringwidth pop
  2 div
  neg
  0 rmoveto
  show } def
  
/right_align_show
{ dup stringwidth pop
  neg
  0 rmoveto
  show } def
"""
    def __init__(self, width, height):
        Renderer.__init__(self, width, height)
        from StringIO import StringIO
        self._eps = StringIO()
        self._eps.write("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: 0 0 %i %i\n" % (width, height))
        self._eps.write(self.EPS_SHOW_FUNCTIONS)
        self._eps.write("%f %f translate\n" % (0, self.height))
        self.set_font(*self.render_state["font"])
        self._inline_func = dict(stroke_color=lambda color: "%f %f %f setrgbcolor" % tuple(255.0 / c for c in color),
                                 fill_color=lambda color:"%f %f %f setrgbcolor" % tuple(255.0 / c for c in color),
                                 stroke_width=lambda w: "%f setlinewidth" % w)
        
    def set_font(self, family, size):
        Renderer.set_font(self, family, size)
        self._eps.write("/%s findfont %f scalefont setfont\n" % self.font())
        
    def set_fill_color(self, color):
        Renderer.set_fill_color(self, color)
        self._eps.write("%f %f %f setrgbcolor\n" % tuple(c/255.0 for c in color))
        
    def set_gradient(self, gradient):
        Renderer.set_gradient(self, gradient)
        (x1, y1, x2, y2), samples = gradient
        binary = "".join([chr(int(c)) for p, s in samples for c in s])
        import binascii
        self._eps.write(self.EPS_SET_GRADIENT % (x1, y1, x2, y2, len(samples), binascii.hexlify(binary))) 
        
    def set_stroke_color(self, color):
        Renderer.set_stroke_color(self, color)
        self._eps.write("%f %f %f setrgbcolor\n" % tuple(c/255.0 for c in color))
        
    def set_stroke_width(self, width):
        Renderer.set_stroke_width(self, width)
        self._eps.write("%f setlinewidth\n" % width)
        
    def set_render_hints(self, hints):
        Renderer.set_render_hints(self, hints)
        if hints.get("linecap", None):
            map = {"butt":0, "round":1, "rect":2}
            self._eps.write("%i setlinecap\n" % (map.get(hints.get("linecap"), 0)))
       
    @with_state 
    def draw_line(self, sx, sy, ex, ey, **kwargs):
        self._eps.write("newpath\n%f %f moveto %f %f lineto\nstroke\n" % (sx, -sy, ex, -ey))
        
    @with_state
    def draw_rect(self, x, y, w, h, **kwargs):
        self._eps.write("newpath\n%(x)f %(y)f moveto %(w)f 0 rlineto\n0 %(h)f rlineto %(w)f neg 0 rlineto\nclosepath\n" % dict(x=x,y=-y, w=w, h=-h))
        self._eps.write("gsave\n")
        if self.gradient():
            self.set_gradient(self.gradient())
        else:
            self.set_fill_color(self.fill_color())
        self._eps.write("fill\ngrestore\n")
        self.set_stroke_color(self.stroke_color())
        self._eps.write("stroke\n")
        
    @with_state
    def draw_polygon(self, vertices, **kwargs):
        self._eps.write("newpath\n%f %f moveto\n" % vertices[0])
        for x, y in vertices[1:]:
            self._eps.write("%f %f lineto\n" % (x, y))
        self._eps.write("closepath\n")
        self._eps.write("gsave\n")
        self.set_fill_color(self.fill_color())
        self._eps.write("fill\ngrestore\n")
        self.set_stroke_color(self.stroke_color())
        self._eps.write("stroke\n")
        
    @with_state
    def draw_text(self, x, y, text, **kwargs):
        show = ["show", "right_align_show", "center_align_show"][self.text_alignment()]
        self._eps.write("%f %f moveto (%s) %s\n" % (x, -y, text, show))
        
    def save_render_state(self):
        Renderer.save_render_state(self)
        self._eps.write("gsave\n")
        
    def restore_render_state(self):
        Renderer.restore_render_state(self)
        self._eps.write("grestore\n")
        
    def translate(self, dx, dy):
        Renderer.translate(self, dx, dy)
        self._eps.write("%f %f translate\n" % (dx, -dy))
        
    def rotate(self, angle):
        Renderer.rotate(self, angle)
        self._eps.write("%f rotate\n" % -angle)
        
    def scale(self, sx, sy):
        Renderer.scale(self, sx, sy)
        self._eps.write("%f %f scale\n" % (sx, sy))
    
    def skew(self, sx, sy):
        Renderer.skew(self, sx, sy)
        self._eps.write("%f %f skew\n" % (sx, sy))
        
    def save(self, filename):
#        self._eps.close()
        open(filename, "wb").write(self._eps.getvalue())
        
    def string_size_hint(self, text, **kwargs):
        import warnings
        warnings.warn("EpsRenderer class does not suport exact string width estimation", stacklevel=2)
        return len(text) * self.font()[1]
        
        
class PILRenderer(Renderer):
    def __init__(self, width, height):
        Renderer.__init__(self, width, height)
        import Image, ImageDraw, ImageFont
        self._pil_image = Image.new("RGB", (width, height), (255, 255, 255))
        self._draw =  ImageDraw.Draw(self._pil_image, "RGB")
        self._pil_font = ImageFont.load_default()

    def _transform(self, x, y):
        p = self.transform() * [[x], [y], [1]]
        return p[0, 0], p[1, 0]
    
    def set_font(self, family, size):
        Renderer.set_font(self, family, size)
        import ImageFont
        try:
            self._pil_font = ImageFont.load(family + ".ttf", size)
        except Exception:
            import warnings
            warnings.warn("Could not load %s.ttf font!", stacklevel=2)
            try:
                self._pil_font = ImageFont.load("cour.ttf", size)
            except Exception:
                warnings.warn("Could not load the cour.ttf font!! Loading the default", stacklevel=2)
                self._pil_font = ImageFont.load_default()
        
    @with_state
    def draw_line(self, sx, sy, ex, ey, **kwargs):
        sx, sy = self._transform(sx, sy)
        ex, ey = self._transform(ex, ey)
        self._draw.line((sx, sy, ex, ey), fill=self.stroke_color(), width=int(self.stroke_width()))

    @with_state
    def draw_rect(self, x, y, w, h, **kwargs):
        x1, y1 = self._transform(x, y)
        x2, y2 = self._transform(x + w, y + h)
        self._draw.rectangle((x1, y1, x2 ,y2), fill=self.fill_color(), outline=self.stroke_color())
        
    @with_state
    def draw_text(self, x, y, text, **kwargs):
        x, y = self._transform(x, y - self.font()[1])
        self._draw.text((x, y), text, font=self._pil_font, fill=self.stroke_color())
        
    def save(self, file):
        self._pil_image.save(file)
        
    def string_size_hint(self, text, **kwargs):
        return self._pil_font.getsize(text)[1]

class SVGRenderer(Renderer):
    SVG_HEADER = """<?xml version="1.0" ?>
<svg height="%f" version="1.0" width="%f" xmlns="http://www.w3.org/2000/svg">
<defs>
    %s
</defs>
    %s
</svg>
"""
    def __init__(self, width, height):
        Renderer.__init__(self, width, height)
        self.transform_count_stack = [0]
        import StringIO
        self._svg = StringIO.StringIO()
        self._defs = StringIO.StringIO()
        self._gradients = {}
        
    def set_gradient(self, gradient):
        Renderer.set_gradient(self, gradient)
        if gradient not in self._gradients.items():
            id = "grad%i" % len(self._gradients)
            self._gradients[id] = gradient
            (x1, y1, x2, y2), stops = gradient
            (x1, y1, x2, y2) = (0, 0, 100, 0)
            
            self._defs.write('<linearGradient id="%s" x1="%f%%" y1="%f%%" x2="%f%%" y2="%f%%">\n' % (id, x1, y1, x2, y2))
            for offset, color in stops:
                self._defs.write('<stop offset="%f" style="stop-color:rgb(%i, %i, %i); stop-opacity:1"/>\n' % ((offset,) + color))
            self._defs.write('</linearGradient>\n')
        
    def get_fill(self):
        if self.render_state["gradient"]:
            return 'style="fill:url(#%s)"' % ([key for key, gr in self._gradients.items() if gr == self.render_state["gradient"]][0])
        else:
            return 'fill="rgb(%i %i %i)"' % self.fill_color()
        
    def get_stroke(self):
#        if self.render_state["gradient"]:
#            return ""
#        else:
            return 'stroke="rgb(%i, %i, %i)"' % self.stroke_color() + ' stroke-width="%f"' % self.stroke_width()
        
    def get_text_alignment(self):
        return 'text-anchor="%s"' % (["start", "end", "middle"][self.text_alignment()])
    
    def get_linecap(self):
        return 'stroke-linecap="%s"' % self.render_hints().get("linecap", "butt")
        
    @with_state
    def draw_line(self, sx, sy, ex, ey):
        self._svg.write('<line x1="%f" y1="%f" x2="%f" y2="%f" %s %s/>\n' % ((sx, sy, ex, ey) + (self.get_stroke(), self.get_linecap())))
        
#    @with_state
#    def draw_lines(self):
        
    @with_state
    def draw_rect(self, x, y, w, h):
        self._svg.write('<rect x="%f" y="%f" width="%f" height="%f" %s %s/>\n' % ((x, y, w, h) + (self.get_fill(),) + (self.get_stroke(),)))
            
    @with_state
    def draw_polygon(self, vertices, **kwargs):
        path = "M %f %f L " % vertices[0]
        path += " L ".join("%f %f" % vert for vert in vertices[1:])
        path += " z"
        self._svg.write('<path d="%s" %s/>' % ((path,) + (self.get_stroke(),)))
        
    @with_state
    def draw_text(self, x, y, text):
        self._svg.write('<text x="%f" y="%f" font-family="%s" font-size="%f" %s>%s</text>\n' % ((x, y) + self.font() +(self.get_text_alignment(), text)))
        
    def translate(self, x, y):
        self._svg.write('<g transform="translate(%f,%f)">\n' % (x, y))
        self.transform_count_stack[-1] = self.transform_count_stack[-1] + 1
        
    def rotate(self, angle):
        self._svg.write('<g transform="rotate(%f)">\n' % angle)
        self.transform_count_stack[-1] = self.transform_count_stack[-1] + 1
        
    def scale(self, sx, sy):
        self._svg.write('<g transform="scale(%f,%f)">\n' % (sx, sy))
        self.transform_count_stack[-1] = self.transform_count_stack[-1] + 1
        
    def skew(self, sx, sy):
        self._svg.write('<g transform="skewX(%f)">' % sx)
        self._svg.write('<g transform="skewY(%f)">' % sy)
        self.transform_count_stack[-1] = self.transform_count_stack[-1] + 2

    def save_render_state(self):
        Renderer.save_render_state(self)
        self.transform_count_stack.append(0)
        
    def restore_render_state(self):
        Renderer.restore_render_state(self)
        count = self.transform_count_stack.pop(-1)
        self._svg.write('</g>\n' * count)
        
    def save(self, filename):
        open(filename, "wb").write(self.SVG_HEADER % (self.height, self.width, self._defs.getvalue(), self._svg.getvalue()))
        
class CairoRenderer(Renderer):
    def __init__(self, width, height):
        Renderer.__init__(self, width, height)
        
