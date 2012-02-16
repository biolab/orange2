from Orange import data
from Orange.evaluation import testing, scoring
from Orange.statistics import distribution
import random
ca = scoring.CA
random.seed(0)

def random_learner(data, *args):
    def random_classifier(*args, **kwargs):
        prob = [random.random() for _ in data.domain.class_var.values]
        sprob = sum(prob)
        prob = [i/sprob for i in prob]
        distribution.Discrete(prob)
        return data.domain.class_var[random.randint(0,
            len(data.domain.class_var.values)-1)], prob
    return random_classifier

def test(measures, test_results):
    n = len(measures)
    print "%8s"*n % tuple(m[1] for m in measures)
    print "=" * 8 * n
    for tr in test_results:
        scores = [m[0](tr)[0] for m in measures]
        scores = ["%8.4f" % s if isinstance(s, float) else
                  (" (%6.4f+%6.4f)")% s if isinstance(s, tuple) else
                  ""
                  for s in scores]
        print "".join(scores)
    print

def run_tests(datasets, measures, tests, iterations=10):
    for ds, ds_name in datasets:
        for t, t_name in tests:
            print "Testing %s on %s" % (t_name, ds_name)
            test_results = [t(random_learner, ds) for _ in xrange(iterations)]
            test(measures, test_results)


datasets = (
    (data.Table("iris"), "Iris"),
    (data.Table("monks-1"), "Monks")
    )

measures = (
    (lambda x:ca(x), "CA"),
    (lambda x:ca(x, report_se=False, ignore_weights=False), "CA-SE-W"),
    (lambda x:ca(x, report_se=True, ignore_weights=False), "CA+SE-W"),
    (lambda x:[lambda x:[None]], ""),
    (lambda x:ca(x, report_se=False, ignore_weights=True), "CA-SE+W"),
    (lambda x:ca(x, report_se=True, ignore_weights=True), "CA+SE+W"),
    (lambda x:[lambda x:[None]], ""),
    )

tests = (
    (lambda l, ds: testing.cross_validation([l],ds), "CV"),
    (lambda l, ds: testing.proportion_test([l], ds, .7, 1), "Proportion test"),
    (lambda l, ds: scoring.confusion_matrices(testing.proportion_test([l], ds, .7, 1)), "Confusion matrix"),
    )

run_tests(datasets, measures, tests)