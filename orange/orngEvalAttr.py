### Janez 03-02-14: Added weights
### Inform Blaz and remove this comment


class OrderAttributesByMeasure:
  def __init__(self, measure=None):
    self.measure=measure

  def __call__(self, data, weight):
    if self.measure:
      measure=self.measure
    else:
      measure=orange.MeasureAttribute_relief(m=5,k=10)
      
    measured=[(attr, measure(attr, data, None, weight)) for attr in data.domain.attributes]
    measured.sort(lambda x, y: cmp(x[1], y[1]))
    return [x[0] for x in measured]


