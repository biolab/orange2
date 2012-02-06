"""
Tests for orange2to25 fixers.

"""
import sys, os
import unittest

from Orange.fixes import fix_changed_names
from Orange.fixes import fix_orange_imports
 
NAME_MAPPING = fix_changed_names.MAPPING
IMPORT_MAPPING = fix_orange_imports.MAPPING

def q_name_split(name):
    """ Split the name into the module name and object name
    within it using the same convention as fix_changed_names 
    tool.
    
    """
    if ":" in name:
        mod_name, obj_name = name.split(":")
    else:
        mod_name, obj_name = name.rsplit(".", 1)
    return mod_name, obj_name
    
    
def rhasattr(obj, name):
    while "." in name:
        first, name = name.split(".", 1)
        if hasattr(obj, first):
            obj = getattr(obj, first)
        else:
            return False
    return hasattr(obj, name)

def rgetattr(obj, name):
    while "." in name:
        first, name = name.split(".", 1)
        if hasattr(obj, first):
            obj = getattr(obj, first)
        else:
            return None
        
    return getattr(obj, name)

def import_package(name):
    mod = __import__(name)
    if "." in name:
        _, name = name.split(".", 1)
        return rgetattr(mod, name)
    else:
        return mod
    
class TestMapping(unittest.TestCase):
    """
    """
    def test_name_mapping(self):
        """ Tests the existance of mapped named pairs.
        """
        for old, new in NAME_MAPPING.items():
            old_mod, old_name = q_name_split(old)
            new_mod, new_name = q_name_split(new)
            
            old_mod = import_package(old_mod)
            new_mod = import_package(new_mod)
            
            
            self.assertTrue(rhasattr(old_mod, old_name), "{0} is missing".format(old))
            self.assertTrue(rhasattr(new_mod, new_name), "{0} is missing".format(new))
            
    def test_import_mapping(self):
        for old_import, new_import in IMPORT_MAPPING.items():
            __import__(old_import)
            __import__(new_import)
            
    
            
            
if __name__ == "__main__":
    unittest.main()
    
