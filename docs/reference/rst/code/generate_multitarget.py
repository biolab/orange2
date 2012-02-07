import random
import Orange

def generate_data_set(n, sigma=0.1, seed=42):
    """
    Generate a toy multi-target data set.

    :param n: Number of instances
    :type n: :obj:`int`
    :param noise: The standard deviation for gaussian noise: N(0, sigma).
    :type noise: :obj:`float`
    :param seed: Seed for the random number generator.

    """
    vars = [Orange.feature.Continuous('X%i' % i) for i in range(1, 4)]
    cvars = [Orange.feature.Continuous('Y%i' % i) for i in range(1, 5)]
    domain = Orange.data.Domain(vars, False, class_vars=cvars)
    data = Orange.data.Table(domain)
    err = lambda: random.gauss(0, sigma)
    random.seed(seed)
    for i in range(n):
        f1, f2 = random.random(), random.random()
        x1 = f1 + err()
        x2 = f2 + err()
        x3 = f1 + f2 + err()
        y1 = f1 + err()
        y2 = 2 * f1 - 3 * f2 + err()
        y3 = 2 * y2 - f1 + err()
        y4 = random.random()
        instance = Orange.data.Instance(domain, [x1, x2, x3])
        instance.set_classes([y1, y2, y3, y4])
        data.append(instance)
    return data

