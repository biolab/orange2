.. index:: association rules

Association rules
=================

Association rules are fun to do in Orange. One reason for this is
Python, and particular implementation that allows a list of
association rules to behave just like any list in Python. That is, you
can select parts of the list, you can remove rules, even add them
(yes, ``append()`` works on Orange association rules!).

For association rules, Orange straightforwardly implements APRIORI
algorithm (see Agrawal et al., Fast discovery of association rules, a
chapter in Advances in knowledge discovery and data mining, 1996),
Orange includes an optimized version of the algorithm that works on
tabular data).  For number of reasons (but mostly for convenience)
association rules should be constructed and managed through the
interface provided by :py:mod:`Orange.associate`.  As implemented in Orange,
association rules construction procedure does not handle continuous
attributes, so make sure that your data is categorized. Also, class
variables are treated just like attributes.  For examples in this
tutorial, we will use data from the data set :download:`imports-85.tab <code/imports-85.tab>`, which
surveys different types of cars and lists their characteristics. We
will use only first ten attributes from this data set and categorize
them so three equally populated intervals will be created for each
continuous variable.  This will be done through the following part of
the code::

   data = orange.ExampleTable("imports-85")
   data = orange.Preprocessor_discretize(data, \
     method=orange.EquiNDiscretization(numberOfIntervals=3))
   data = data.select(range(10))

Now, to our examples. First one uses the data set constructed with
above script and shows how to build a list of association rules which
will have support of at least 0.4. Next, we select a subset of first
five rules, print them out, delete first three rules and repeat the
printout. The script that does this is (part of :download:`assoc1.py <code/assoc1.py>`, uses
:download:`imports-85.tab <code/imports-85.tab>`)::

   rules = orange.AssociationRulesInducer(data, support=0.4)
   
   print "%i rules with support higher than or equal to %5.3f found." % (len(rules), minSupport)
   
   orngAssoc.sort(rules, ["support", "confidence"])
   
   orngAssoc.printRules(rules[:5], ["support", "confidence"])
   print
   
   del rules[:3]
   orngAssoc.printRules(rules[:5], ["support", "confidence"])
   print

The output of this script is::

   87 rules with support higher than or equal to 0.400 found.
   
   supp    conf    rule
   0.888   0.984   engine-location=front -> fuel-type=gas
   0.888   0.901   fuel-type=gas -> engine-location=front
   0.805   0.982   engine-location=front -> aspiration=std
   0.805   0.817   aspiration=std -> engine-location=front
   0.785   0.958   fuel-type=gas -> aspiration=std
   
   supp    conf    rule
   0.805   0.982   engine-location=front -> aspiration=std
   0.805   0.817   aspiration=std -> engine-location=front
   0.785   0.958   fuel-type=gas -> aspiration=std
   0.771   0.981   fuel-type=gas aspiration=std -> engine-location=front
   0.771   0.958   aspiration=std engine-location=front -> fuel-type=gas
   
Notice that the when printing out the rules, user can specify which
rule evaluation measures are to be printed. Choose anything from
``['support', 'confidence', 'lift', 'leverage', 'strength',
'coverage']``.

The second example uses the same data set, but first prints out five
most confident rules. Then, it shows a rather advanced type of
filtering: every rule has parameters that record its support,
confidence, etc... These may be used when constructing your own filter
functions. The one in our example uses ``support`` and ``lift``.

.. note:: 
   If you have just started with Python: lambda is a compact way to
   specify a simple function without using def statement. As a
   function, it uses its own name space, so minimal lift and support
   requested in our example should be passed as function
   arguments. 

Here goes the code (part of :download:`assoc2.py <code/assoc2.py>`)::

   rules = orange.AssociationRulesInducer(data, support = 0.4)
   
   n = 5
   print "%i most confident rules:" % (n)
   orngAssoc.sort(rules, ["confidence"])
   orngAssoc.printRules(rules[0:n], ['confidence','support','lift'])
   
   conf = 0.8; lift = 1.1
   print "\nRules with support>%5.3f and lift>%5.3f" % (conf, lift)
   rulesC=rules.filter(lambda x: x.confidence>conf and x.lift>lift)
   orngAssoc.sort(rulesC, ['confidence'])
   orngAssoc.printRules(rulesC, ['confidence','support','lift'])
   
Just one rule with requested support and lift is found in our rule set::

   5 most confident rules:
   conf    supp    lift    rule
   1.000   0.478   1.015   fuel-type=gas aspiration=std drive-wheels=fwd -> engine-location=front
   1.000   0.429   1.015   fuel-type=gas aspiration=std num-of-doors=four -> engine-location=front
   1.000   0.507   1.015   aspiration=std drive-wheels=fwd -> engine-location=front
   1.000   0.449   1.015   aspiration=std num-of-doors=four -> engine-location=front
   1.000   0.541   1.015   fuel-type=gas drive-wheels=fwd -> engine-location=front
   
   Rules with confidence>0.800 and lift>1.100
   conf    supp    lift    rule
   0.898   0.429   1.116   fuel-type=gas num-of-doors=four -> aspiration=std engine-location=front
   
Finally, for our third example, we introduce cloning. Cloning helps if
you require to work with different rule subsets that stem from common
rule set created from some data (actually, cloning is quite useless in
our example, but may be very useful otherwise). So, we use cloning to
make a copy of the set of rules, then sort by first support and then
confidence, and then print out few best rules. We have also lower
required minimal support, just to see how many rules we obtain in this
way (:download:`assoc3.py <code/assoc3.py>`, :download:`imports-85.tab <code/imports-85.tab>`)::

   minSupport = 0.2
   rules = orngAssoc.build(data, minSupport)
   print "%i rules with support higher than or equal to %5.3f found.\n" % (len(rules), minSupport)
   
   rules2 = rules.clone()
   rules2.sortByConfidence()
   
   n = 5
   print "Best %i rules:" % n
   subset = rules[:n]
   subset.printMeasures(['support','confidence'])

The output of this script is::

   828 rules with support higher than or equal to 0.200 found.
   
   Best 5 rules:
   supp    conf    rule
   0.888   0.984   engine-location=front -> fuel-type=gas
   0.888   0.901   fuel-type=gas -> engine-location=front
   0.805   0.982   engine-location=front -> aspiration=std
   0.805   0.817   aspiration=std -> engine-location=front
   0.785   0.958   fuel-type=gas -> aspiration=std


