"""Test read write
"""
from ...gui import test
from ...registry import global_registry

from .. import Scheme, SchemeNode, SchemeLink
from ..readwrite import scheme_to_ows_stream, parse_scheme


class TestReadWrite(test.QAppTestCase):
    def test_io(self):
        from StringIO import StringIO
        reg = global_registry()

        base = "Orange.OrangeWidgets"
        file_desc = reg.widget(base + ".Data.OWFile.OWFile")
        discretize_desc = reg.widget(base + ".Data.OWDiscretize.OWDiscretize")
        bayes_desc = reg.widget(base + ".Classify.OWNaiveBayes.OWNaiveBayes")

        scheme = Scheme()
        file_node = SchemeNode(file_desc)
        discretize_node = SchemeNode(discretize_desc)
        bayes_node = SchemeNode(bayes_desc)

        scheme.add_node(file_node)
        scheme.add_node(discretize_node)
        scheme.add_node(bayes_node)

        scheme.add_link(SchemeLink(file_node, "Data",
                                   discretize_node, "Data"))

        scheme.add_link(SchemeLink(discretize_node, "Data",
                                   bayes_node, "Data"))

        stream = StringIO()
        scheme_to_ows_stream(scheme, stream)

        stream.seek(0)

        scheme_1 = parse_scheme(Scheme(), stream)

        self.assertTrue(len(scheme.nodes) == len(scheme_1.nodes))
        self.assertTrue(len(scheme.links) == len(scheme_1.links))
