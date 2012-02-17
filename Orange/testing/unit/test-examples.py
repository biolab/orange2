try:
    import unittest2 as unittest
except:
    import unittest
import os, sys
from optparse import OptionParser

import orange

option = OptionParser()
option.add_option("-m", "--module", type=str, dest="module", default="orange", help="List the modules to test (Possible values: orange, obi, text)")
option.add_option("-o", "--output", type=str, dest="output", default=None, help="Output file (default is standard output)")

options, args = option.parse_args()

module = options.module
output = options.output

dirs = {"orange": "doc/modules",
        "obi": "add-ons/Bioinformatics/doc/modules",
        "text": "add-ons/Text/doc/modules"
        }

if module not in dirs:
    print "Unknown module name!"
    sys.exit(1)

orangedir = os.path.dirname(orange.__file__)

from Orange.testing import test_suite_scripts

suite = test_suite_scripts(os.path.join(orangedir, dirs[module]))
unittest.TextTestRunner(verbosity=2).run(suite)

