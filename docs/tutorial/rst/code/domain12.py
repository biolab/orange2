# Description: Add a new attribute price to a car data set, compute it from two existing attributes (buying, maint)
# Category:    preprocessing
# Uses:        car
# Classes:     Domain, Value, getValueFrom, EnumVariable
# Referenced:  domain.htm

import orange
data = orange.ExampleTable('car.tab')

# add attribute price = f(buying, maint)
# see also http://www.ailab.si/hint/car_dataset.asp

priceTable = {}
priceTable['v-high:v-high'] = 'v-high'
priceTable['high:v-high'] = 'v-high'
priceTable['med:v-high'] = 'high'
priceTable['low:v-high'] = 'high'
priceTable['v-high:high'] = 'v-high'
priceTable['high:high'] = 'high'
priceTable['med:high'] = 'high'
priceTable['low:high'] = 'med'
priceTable['v-high:med'] = 'high'
priceTable['high:med'] = 'high'
priceTable['med:med'] = 'med'
priceTable['low:med'] = 'low'
priceTable['v-high:low'] = 'high'
priceTable['high:low'] = 'high'
priceTable['med:low'] = 'low'
priceTable['low:low'] = 'low'

def f(price, buying, maint):
  return orange.Value(price, priceTable['%s:%s' % (buying, maint)])

price = orange.EnumVariable("price", values=["v-high", "high", "med", "low"])
price.getValueFrom = lambda e, getWhat: f(price, e['buying'], e['maint'])
newdomain = orange.Domain(data.domain.attributes + [price, data.domain.classVar])
newdata = data.select(newdomain)

print
for a in newdata.domain.attributes:
  print "%10s" % a.name,
print "%10s" % newdata.domain.classVar.name
for i in [1, 200, 300, 1200, 1700]:
  for a in newdata.domain.attributes:
    print "%10s" % newdata[i][a],
  print "%10s" % newdata[i].getclass()
