import orange, orngTest, orngTree

learners = [orange.BayesLearner(name = "bayes"),
            orngTree.TreeLearner(name="tree"),
            orange.MajorityLearner(name="majrty")]

voting = orange.ExampleTable("voting")
res = orngTest.crossValidation(learners, voting)

vehicle = orange.ExampleTable("vehicle")
resVeh = orngTest.crossValidation(learners, vehicle)