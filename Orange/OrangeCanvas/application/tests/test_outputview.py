from datetime import datetime

from ...gui.test import QAppTestCase

from ..outputview import OutputText


class TestOutputView(QAppTestCase):
    def test_outputview(self):
        output = OutputText()
        output.show()

        line1 = "A line \n"
        line2 = "A different line\n"
        output.write(line1)
        self.assertEqual(unicode(output.toPlainText()), line1)

        output.write(line2)
        self.assertEqual(unicode(output.toPlainText()), line1 + line2)

        output.clear()
        self.assertEqual(unicode(output.toPlainText()), "")

        output.writelines([line1, line2])
        self.assertEqual(unicode(output.toPlainText()), line1 + line2)

        output.setMaximumLines(5)

        def advance():
            now = datetime.now().strftime("%c\n")
            output.write(now)

            text = unicode(output.toPlainText())
            self.assertLessEqual(len(text.splitlines()), 5)

            self.singleShot(500, advance)

        advance()

        self.app.exec_()
