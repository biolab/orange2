from Orange.misc import testing
from Orange.data import utils
import Orange

try:
    import unittest2 as unittest
except:
    import unittest
import random


@testing.datasets_driven
class TestTake(testing.DataTestCase):
    @testing.test_on_data
    def test_take_domain(self, data):
        size = len(data.domain)
        indices = range(size)
        to_mask = lambda inds: [i for i in indices if i in inds]

        indices1 = [random.randrange(size)]
        indices2 = indices1 + [random.randrange(size)]
        indices3 = indices1 + [random.randrange(size)]

        mask1 = to_mask(indices1)
        mask2 = to_mask(indices2)
        mask3 = to_mask(indices3)

        data1 = utils.take(data, indices1, axis=1)
        data1m = utils.take(data, mask1, axis=1)
        self.assertEquals(len(data1.domain), len(indices1))
        self.assertEquals(len(data1m.domain), len(set(indices1)))
#        self.assertEquals(list(data1), list(data1m))

        data2 = utils.take(data, indices2, axis=1)
        data2m = utils.take(data, mask2, axis=1)
        self.assertEquals(len(data2.domain), len(indices2))
        self.assertEquals(len(data2m.domain), len(set(indices2)))
#        self.assertEquals(list(data2), list(data2m))

        data3 = utils.take(data, indices3, axis=1)
        data3m = utils.take(data, mask3, axis=1)
        self.assertEquals(len(data3.domain), len(indices3))
        self.assertEquals(len(data3m.domain), len(set(indices3)))
#        self.assertEquals(list(data3), list(data3m))

    @testing.test_on_data
    def test_take_instances(self, data):
        size = len(data)
        indices = range(len(data))
        to_mask = lambda inds: [i for i in indices if i in inds]

        indices1 = [random.randrange(size)]
        indices2 = indices1 + [random.randrange(size)]
        indices3 = indices1 + [random.randrange(size)]

        mask1 = to_mask(indices1)
        mask2 = to_mask(indices2)
        mask3 = to_mask(indices3)

        data1 = utils.take(data, indices1, axis=0)
        data1m = utils.take(data, mask1, axis=0)
        self.assertEquals(len(data1), len(indices1))
        self.assertEquals(len(data1m), len(set(indices1)))

        data2 = utils.take(data, indices2, axis=0)
        data2m = utils.take(data, mask2, axis=0)
        self.assertEquals(len(data2), len(indices2))
        self.assertEquals(len(data2m), len(set(indices2)))

        data3 = utils.take(data, indices3, axis=0)
        data3m = utils.take(data, mask3, axis=0)
        self.assertEquals(len(data3), len(indices3))
        self.assertEquals(len(data3m), len(set(indices3)))

def split(table):
    size = len(table.domain)
    indices = range(size)
    to_mask = lambda inds: [i for i in indices if i in inds]

    indices = [random.randrange(size) for i in range(2)]
    part1 = utils.take(table, indices, axis=1)
    complement = [i for i in range(len(table.domain)) if i not in indices]
    part2 = utils.take(table, complement, axis=1)

    return part1, part2

@testing.datasets_driven
class TestJoins(unittest.TestCase):
    @testing.test_on_data
    def test_left_join(self, table):
        utils.add_row_id(table)
        part1, part2 = split(table)
        utils.left_join(part1, part2, utils._row_meta_id, utils._row_meta_id)

    @testing.test_on_data
    def test_right_join(self, table):
        utils.add_row_id(table)
        part1, part2 = split(table)
        utils.right_join(part1, part2, utils._row_meta_id, utils._row_meta_id)

if __name__ == "__main__":
    unittest.main()
