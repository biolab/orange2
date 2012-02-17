from Orange.misc import testing
from Orange.misc.testing import test_on_data, datasets_driven

try:
    import unittest2 as unittest
except:
    import unittest
from Orange.projection import mds
from Orange.distance import distance_matrix, Euclidean

@datasets_driven
class TestMDS(unittest.TestCase):

    @test_on_data
    def test_mds_on(self, data):
        matrix = distance_matrix(data, Euclidean)
        self.__mds_test_helper(matrix, proj_dim=1)
        self.__mds_test_helper(matrix, proj_dim=2)
        self.__mds_test_helper(matrix, proj_dim=3)

    def __mds_test_helper(self, matrix, proj_dim):
        proj = mds.MDS(matrix, dim=proj_dim)
        proj.torgerson()
        proj.smacof_step()
        proj.run(100)
        proj.smacof_step()
        self.assertEquals(len(proj.points), matrix.dim)
        self.assertTrue(all(len(p) == proj_dim for p in proj.points))
        self.assertEquals(matrix, proj.distances,
                        "The input distance matrix was changed in place")
        self.assertEquals(matrix.dim, proj.projected_distances.dim)



if __name__ == "__main__":
    unittest.main()
