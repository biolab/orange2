from Orange import data
try:
    import unittest2 as unittest
except:
    import unittest

class TestPickClass(unittest.TestCase):
    def __init__(self, testCaseName):
        unittest.TestCase.__init__(self, testCaseName)
        self.orig = data.Table('multitarget-synthetic')

    def test_pick_first(self):
        d = data.Table(self.orig)

        #picks first with no class_var
        d.pick_class(d.domain.class_vars[0])
        self.assertEquals(d.domain.class_var, self.orig.domain.class_vars[0])
        self.assertEquals(d.domain.class_vars[0], self.orig.domain.class_vars[1])
        self.assertEquals(d.domain.class_vars[1], self.orig.domain.class_vars[2])
        self.assertEquals(d.domain.class_vars[2], self.orig.domain.class_vars[3])
        for i in range(len(d)):
            self.assertEquals(d[i].get_class(), self.orig[i].get_classes()[0])
            self.assertEquals(d[i].get_classes()[0], self.orig[i].get_classes()[1])
            self.assertEquals(d[i].get_classes()[1], self.orig[i].get_classes()[2])
            self.assertEquals(d[i].get_classes()[2], self.orig[i].get_classes()[3])
        
        #picks first with existing class_var
        d.pick_class(d.domain.class_vars[0])
        self.assertEquals(d.domain.class_var, self.orig.domain.class_vars[1])
        self.assertEquals(d.domain.class_vars[0], self.orig.domain.class_vars[0])
        self.assertEquals(d.domain.class_vars[1], self.orig.domain.class_vars[2])
        self.assertEquals(d.domain.class_vars[2], self.orig.domain.class_vars[3])
        for i in range(len(d)):
            self.assertEquals(d[i].get_class(), self.orig[i].get_classes()[1])
            self.assertEquals(d[i].get_classes()[0], self.orig[i].get_classes()[0])
            self.assertEquals(d[i].get_classes()[1], self.orig[i].get_classes()[2])
            self.assertEquals(d[i].get_classes()[2], self.orig[i].get_classes()[3])

        #picks None with existing class_var
        d.pick_class(None)
        self.assertEquals(d.domain.class_vars[0], self.orig.domain.class_vars[1])
        self.assertEquals(d.domain.class_vars[1], self.orig.domain.class_vars[0])
        self.assertEquals(d.domain.class_vars[2], self.orig.domain.class_vars[2])
        self.assertEquals(d.domain.class_vars[3], self.orig.domain.class_vars[3])
        for i in range(len(d)):
            self.assertEquals(d[i].get_classes()[0], self.orig[i].get_classes()[1])
            self.assertEquals(d[i].get_classes()[1], self.orig[i].get_classes()[0])
            self.assertEquals(d[i].get_classes()[2], self.orig[i].get_classes()[2])
            self.assertEquals(d[i].get_classes()[3], self.orig[i].get_classes()[3])

    def test_pick_nonfirst(self):
        d = data.Table(self.orig)

        #picks not first with no class_var
        d.pick_class(d.domain.class_vars[2])
        self.assertEquals(d.domain.class_var, self.orig.domain.class_vars[2])
        self.assertEquals(d.domain.class_vars[0], self.orig.domain.class_vars[1])
        self.assertEquals(d.domain.class_vars[1], self.orig.domain.class_vars[0])
        self.assertEquals(d.domain.class_vars[2], self.orig.domain.class_vars[3])
        for i in range(len(d)):
            self.assertEquals(d[i].get_class(), self.orig[i].get_classes()[2])
            self.assertEquals(d[i].get_classes()[0], self.orig[i].get_classes()[1])
            self.assertEquals(d[i].get_classes()[1], self.orig[i].get_classes()[0])
            self.assertEquals(d[i].get_classes()[2], self.orig[i].get_classes()[3])

        #picks not first with existing class_var
        d.pick_class(d.domain.class_vars[2])
        self.assertEquals(d.domain.class_var, self.orig.domain.class_vars[3])
        self.assertEquals(d.domain.class_vars[0], self.orig.domain.class_vars[1])
        self.assertEquals(d.domain.class_vars[1], self.orig.domain.class_vars[0])
        self.assertEquals(d.domain.class_vars[2], self.orig.domain.class_vars[2])
        for i in range(len(d)):
            self.assertEquals(d[i].get_class(), self.orig[i].get_classes()[3])
            self.assertEquals(d[i].get_classes()[0], self.orig[i].get_classes()[1])
            self.assertEquals(d[i].get_classes()[1], self.orig[i].get_classes()[0])
            self.assertEquals(d[i].get_classes()[2], self.orig[i].get_classes()[2])
        
        #picks None with existing class_var
        d.pick_class(None)
        self.assertEquals(d.domain.class_vars[0], self.orig.domain.class_vars[3])
        self.assertEquals(d.domain.class_vars[1], self.orig.domain.class_vars[1])
        self.assertEquals(d.domain.class_vars[2], self.orig.domain.class_vars[0])
        self.assertEquals(d.domain.class_vars[3], self.orig.domain.class_vars[2])
        for i in range(len(d)):
            self.assertEquals(d[i].get_classes()[0], self.orig[i].get_classes()[3])
            self.assertEquals(d[i].get_classes()[1], self.orig[i].get_classes()[1])
            self.assertEquals(d[i].get_classes()[2], self.orig[i].get_classes()[0])
            self.assertEquals(d[i].get_classes()[3], self.orig[i].get_classes()[2])

    #uncomment when bug is fixed
    #def test_pick_none(self):
    #    d = data.Table(self.orig)
    #    d.pick_class(None)
    #    self.assertEquals(d.domain.class_vars,self.orig.domain.class_vars)
