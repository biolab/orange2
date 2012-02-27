try:
    import unittest2 as unittest
except:
    import unittest

from Orange.misc import selection


class TestBestOnTheFly(unittest.TestCase):
    def test_compare_on_1st(self):
        best = selection.BestOnTheFly(call_compare_on_1st=True)
        test = [1, 2, 0, 5, 4, 5, 4]
        # Test on integers
        for t in test:
            best.candidate((t, str(t)))

        winner = best.winner()
        self.assertTrue(winner == (5, "5"))
        self.assertIsInstance(winner, tuple)

        index = best.winner_index()
        self.assertIsInstance(index, int)


    def test_compare_first_bigger(self):
        best = selection.BestOnTheFly(selection.compare_first_bigger)
        test = [1, 2, 0, 5, 4, 5, 4]
        # Test on integers
        for t in test:
            best.candidate((t, str(t)))

        winner = best.winner()
        self.assertTrue(winner == (5, "5"))
        self.assertIsInstance(winner, tuple)

        index = best.winner_index()
        self.assertIsInstance(index, int)


if __name__ == "__main__":
    unittest.main()