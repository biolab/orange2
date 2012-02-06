# Description: Shows different uses of orange.Domain
# Category:    preprocessing
# Uses:        glass
# Classes:     Domain
# Referenced:  domain.htm

import orange

domain = orange.ExampleTable("glass").domain

tests = ( '(["Na", "Mg"], domain)',
          '(["Na", "Mg"], 1, domain)',
          '(["Na", "Mg"], 0, domain)',
          '(["Na", "Mg"], domain.variables)',
          '(["Na", "Mg"], 1, domain.variables)',
          '(["Na", "Mg"], 0, domain.variables)',
          '([domain["Na"], "Mg"], 0, domain.variables)',
          '([domain["Na"], "Mg"], 0, domain)',
          '([domain["Na"], "Mg"], 0, domain.variables)',
          '([domain["Na"], domain["Mg"]], 0)',
          '([domain["Na"], domain["Mg"]], 1)',
          '([domain["Na"], domain["Mg"]], None)',
          '([domain["Na"], domain["Mg"]], orange.EnumVariable("something completely different"))',
          '(domain)',
          '(domain, 0)',
          '(domain, 1)',
          '(domain, "Mg")',
          '(domain, domain[0])',
          '(domain, None)',
          '(domain, orange.FloatVariable("nothing completely different"))')
          
for args in tests:
  line = "orange.Domain%s" % args
  d = eval(line)
  print line
  print "  classVar: %s" % d.classVar
  print "  attributes: %s" % d.attributes
  print
