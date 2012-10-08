
from .. import NodeItem

from . import TestItems


class TestNodeItem(TestItems):

    def test_nodeitem(self):
        from ....registry.tests import small_testing_registry
        reg = small_testing_registry()

        data_desc = reg.category("Data")

        file_desc = reg.widget("Orange.OrangeWidgets.Data.OWFile.OWFile")

        file_item = NodeItem()
        file_item.setWidgetDescription(file_desc)
        file_item.setWidgetCategory(data_desc)

        file_item.setTitle("File Node")
        self.assertEqual(file_item.title(), "File Node")

        file_item.setProcessingState(True)
        self.assertEqual(file_item.processingState(), True)

        file_item.setProgress(50)
        self.assertEqual(file_item.progress(), 50)

        file_item.setProgress(100)
        self.assertEqual(file_item.progress(), 100)

        file_item.setProgress(101)
        self.assertEqual(file_item.progress(), 100, "Progress overshots")

        file_item.setProcessingState(False)
        self.assertEqual(file_item.processingState(), False)
        self.assertEqual(file_item.progress(), -1,
                         "setProcessingState does not clear the progress.")

        self.scene.addItem(file_item)
        file_item.setPos(100, 100)

        discretize_desc = reg.widget(
            "Orange.OrangeWidgets.Data.OWDiscretize.OWDiscretize"
        )

        discretize_item = NodeItem()
        discretize_item.setWidgetDescription(discretize_desc)
        discretize_item.setWidgetCategory(data_desc)

        self.scene.addItem(discretize_item)
        discretize_item.setPos(300, 100)

        classify_desc = reg.category("Classify")

        bayes_desc = reg.widget(
            "Orange.OrangeWidgets.Classify.OWNaiveBayes.OWNaiveBayes"
        )

        nb_item = NodeItem()
        nb_item.setWidgetDescription(bayes_desc)
        nb_item.setWidgetCategory(classify_desc)

        self.scene.addItem(nb_item)
        nb_item.setPos(500, 100)

        positions = []
        anchor = file_item.newOutputAnchor()
        anchor.scenePositionChanged.connect(positions.append)

        file_item.setPos(110, 100)
        self.assertTrue(len(positions) > 0)

        def progress():
            self.singleShot(10, progress)
            nb_item.setProgress((nb_item.progress() + 1) % 100)

        progress()

        self.app.exec_()
