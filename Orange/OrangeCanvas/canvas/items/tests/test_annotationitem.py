from PyQt4.QtCore import Qt, QRectF, QLineF

from ..annotationitem import TextAnnotation, ArrowAnnotation, ArrowItem, \
                             ControlPointRect

from . import TestItems


class TestAnnotationItem(TestItems):
    def test_textannotation(self):
        text = "Annotation"
        annot = TextAnnotation()
        annot.setPlainText(text)
        self.assertEqual(annot.toPlainText(), text)

        annot2 = TextAnnotation()
        self.assertEqual(annot2.toPlainText(), "")

        text = "This is an annotation"
        annot2.setPlainText(text)
        self.assertEqual(annot2.toPlainText(), text)

        annot2.setDefaultTextColor(Qt.red)
        control_rect = QRectF(0, 0, 100, 200)
        annot2.setGeometry(control_rect)
        self.assertEqual(annot2.geometry(), control_rect)

        annot.setTextInteractionFlags(Qt.TextEditorInteraction)
        annot.setPos(400, 100)
        annot.adjustSize()
        annot._TextAnnotation__textItem.setFocus()
        self.scene.addItem(annot)
        self.scene.addItem(annot2)

        self.app.exec_()

    def test_arrowannotation(self):
        item = ArrowItem()
        self.scene.addItem(item)
        item.setLine(QLineF(100, 100, 100, 200))
        item.setLineWidth(5)

        item = ArrowAnnotation()
        item.setPos(10, 10)
        item.setLine(QLineF(10, 10, 200, 400))

        self.scene.addItem(item)
        item.setLineWidth(5)

        self.app.exec_()

    def testcontrol(self):
        cp = ControlPointRect()
        cp.setRect(QRectF(30, 30, 50, 50))
        self.scene.addItem(cp)
        self.app.exec_()
