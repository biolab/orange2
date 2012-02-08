from Orange import data
from Orange.evaluation import testing, scoring
from Orange.statistics import distribution
import random
auc = scoring.AUC
random.seed(0)

def random_learner(data, *args):
    def random_classifier(*args, **kwargs):
        prob = [random.random() for _ in data.domain.class_var.values]
        sprob = sum(prob)
        prob = [i/sprob for i in prob]
        distribution.Discrete(prob)
        return data.domain.class_var[0], prob
    return random_classifier

def test(measures, test_results):
    n = len(measures)
    print "%8s"*n % tuple(m[1] for m in measures)
    print "=" * 8 * n
    for tr in test_results:
        print "%8.4f"*n % tuple(m[0](tr)[0] for m in measures)
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
    (lambda x:auc(x), "AUC"),
    (lambda x:auc(x, method=0), "AUC+M0"),
    (lambda x:auc(x, method=1), "AUC+M1"),
    (lambda x:auc(x, method=2), "AUC+M2"),
    (lambda x:auc(x, method=3), "AUC+M3"),
)

tests = (
    (lambda l, ds: testing.cross_validation([l],ds), "CV"),
    (lambda l, ds: testing.proportion_test([l], ds, .7, 1), "Proportion test"),
)

run_tests(datasets, measures, tests)