import orange, math

def perfectSquares(x):
    return filter(lambda x:math.floor(math.sqrt(x)) == math.sqrt(x), range(x+1))

class A:
    def __init__(self, x):
        self.x = x
    def __str__(self):
        return "value: %s" % self.x

a = 12

data = orange.ExampleTable("pythonvariable.tab")
for i in data:
    print i
