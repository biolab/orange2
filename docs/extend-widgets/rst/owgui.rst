#####################################
OWGUI: Library of Common GUI Controls
#####################################

Orange Widgets wrap Orange's classes in an easy to use interactive graphical
interface. As such, much of their code is about the interface, event control
and maintaining the state of the GUI controls.

In the spirit of constructive laziness, we wrote a library using which a single
line of code can construct a check box, line edit or a combo, make it being
synchronized with a Python object's attribute (which, by the way, gets
automatically saved and retrieved when the widgets is closed and reopened),
attaches a callback function to the control, make it disable or enable other controls...

*****************
Common Attributes
*****************

Many functions that construct controls share one or more common arguments,
usually in the same order. These are described here. Descriptions of individual
controls will only list their specific arguments, while the arguments which are
presented here will only be described in cases where they have a different meaning.

widget (required)
Widget on which control will be drawn - can be widget's :obj:`controlArea` or another box.

master (required)
Object which includes an attribute that are used to store control's
state, most often the object which called the function that
initialized the control.

value (required)
String with the name of the master's attribute that synchronizes with the
state of the control (and vice-versa - when this attribute is changed, the control changes as well). This attribute should usually be also included the master's :obj:`settingsList`, so that it is automatically saved and retrieved.

box (default: None)
Indicates if there should be a box that is drawn around the control. If :obj:`box` is :obj:`None`, no box is drawn; if it is a string, it is also used as box's name. If :obj:`box` is any other true value (such as :obj:`True` :), an unlabeled box is drawn.

callback (default: None)
A function to be called when the state of the control is changed. Can include a single function, or a list of functions that will be called in the order provided. If callback function changes the value of the controlled attribute (the one given as the :obj:`value` argument described above) it may trigger a cycle; a simple trick to avoid this is shown in the description of <a href="#listBox">listBox function</a>.

tooltip (default: None)
A string that is displayed in a tooltip that appears when mouse is over the control.

label (default: None)
A string that is displayed as control's label.

labelWidth (default: None)
Sets the label's width. This is useful for aligning the controls.

orientation (default: "vertical")
When label is used, determines the relative placement of the label and the control. Label can be above the control, "vertical", or in the same line with control, "horizontal". Instead of "vertical" and "horizontal" you can also use :obj:`True` and :obj:`False` or 1 and 0, respectively. (Remember this as "vertical" being the usual order of controls in the widgets, so vertical is "true".)

disabled (default: False)
Tells whether the control be disabled upon the initialization.

addSpace (default: False)
If true, a space of 8 pixels is added after the widget by calling :obj:`OWGUI.separator`. :obj:`addSpace` can also be an integer specifying the height of the added space.


********
Controls
********

This section describes the OWGUI wrappers for controls like check boxes, buttons
and similar. All the important Qt's controls can be constructed through this
functions. You should always use them instead of calling Qt directly, not only
because they are convenient, but also because they set up a lot of things that happen in behind.


Check Box
*********

Check box, a wrapper around QCheckBox, adding a label, box, tooltip, callback
and synchronization with the designated widget's attribute.

checkBox(widget, master, value, label[, box, tooltip, callback, disabled, labelWidth, disables])

disables (default: [])
If the check box needs to disable some other controls they can be given in list  :obj:`disables`, e.g. :obj:`disables=[someOtherCheckBox, someLineEdit]`. If the other control should be disabled when the checkbox is checked, do it like this: :obj:`disables=[someOtherCheckBox, (-1, someLineEdit)]` - now :obj:`someOtherCheckBox` will be enabled when this check box is checked, while :obj:`someLineEdit` will be enabled when the check box is unchecked.

labelWidth (default: None)
:obj:`labelWidth` can be used to align this widget with others.


Line Edit
*********

Edit box, a wrapper around QLineEdit.

lineEdit(widget, master, value[, label, labelWidth, orientation, box, tooltip, callback, valueType, validator, controlWidth])


valueType (default: str)
A type into which the value is cast.

validator (default: None)
A standard Qt validator that can be associated with the control.


Button
******

A wrapper around QPushButton, just to be able to define a button
and its callback in a single line.

button(widget, master, label[, callback, disabled, tooltip])


Radio Buttons
*************

OWGUI can create an individual radio button or a box of radio buttons or an individual radio button.

An individual radio button is created by :obj:`radioButton`.

radioButton(widget, master, value, label[, box, tooltip, callback, addSpace])

The function provides the usual capabilities of OWGUI controls. It is though 
your responsibility to put it in something like a :obj:`QVButtonGroup`.

A box of radio buttons is created by function :obj:`radioButtonsInBox`.


radioButtonsInBox(widget, master, value, btnLabels[, box, tooltips, callback)

value (required)
Synchronized with the index of the selected radio button.

btnLabels (required)
A list with labels for radio buttons. Labels can be strings or pixmaps.

tooltips (default: None)
A list of tooltips, one for each button.


Combo Box
*********

A wrapper around QComboBox.

comboBox(widget, master, value[, box, label, labelWidth, orientation, items, tooltip, callback, sendSelectedValue, valueType, control2attributeDict, emptyString])

<dl class="attributes">
items (default: [])
A list of combo box's items. Unlike most OWGUI, :obj:`items` have one Orange-specific quirk: its element can be either a string, in which case it is used as a label, or a tuple, where the first element is a label name and the last is the attribute type which is used to create an icon. Most attribute lists in Orange Widgets are constructed this way.

sendSelectedValue (default: 0)
If false, attribute :obj:`value` will be assigned the index of the selected item. Otherwise, it is assigned the currently selected item's label.

control2attributeDict (default: {})
A dictionary for translating the item's label into :obj:`value`. It is used only is :obj:`sendSelectedValue` is true, and even then a label is translated only if an item with such a key is found in the dictionary; otherwise, label is written to :obj:`value` as it is. 

emptyString (default: "")
Tells which combo box's item corresponds to an empty :obj:`value`. This is typically used when combo box's labels are attribute names and an item "(none)", which allows user to select no attribute. If we give :obj:`emptyString="(none)"`, :obj:`value` will be an empty string when the user selects "(none)". This is equivalent to specifying :obj:`control2attributeDict = {"(none)": ""}` (and is actually implemented like that), but far more convenient.

valueType (default: str or unicode)
A function through which the currently selected item's label is converted prior to looking into :obj:`control2attributeDict`. Needed to convert Qt's QString.


List Box
********

This control, which might be the most complicated control in OWGUI, is a
sophisticated wrapper around QListBox. It's complexity arises from synchronization.


listBox(widget, master, value, labels[, box, tooltip, callback, selectionMode])

<dl class="attributes">
value (required)
The name of master's attribute containing indices of all selected values.

labels (required)
The name of master's attribute containing the list box's labels. Similar to :obj:`items` in combo box, list :obj:`labels` have one Orange-specific quirk: its element can be either a string, in which case it is used as a label, or a tuple, where the first element is a label name and the second can be either an icon on an integer, representing the attribute type which is used to create an icon. Most attribute lists in Orange Widgets are constructed this way.

selectionMode (default: QListWidget.SingleSelection)
Tells whether the user can select a single item (:obj:`QListWidget.SingleSelection`), multiple items (:obj:`QListWidget.MultiSelection`, :obj:`QListWidget.ExtendedSelection`) or nothing (:obj:`QListWidget.NoSelection`).


:obj:`value` is automatically cast to :obj:`OWGUI.ControlledList` (this is needed because the list should report any changes to the control, the list box; :obj:`OWGUI.ControlledList` is like an ordinary Python :obj:`list` except that it triggers synchronization with the list box at every change).

:obj:`labels` is only partially synchronized with the list box: if a new list is assigning to :obj:`labels` attribute, the list will change. If elements of the existing list are changed or added, the list box won't budge. You should never change the list, but always assign a new list (or reassign the same after it's changed). If the labels are stored in :obj:`self.listLabels` and you write :obj:`self.listLabels[1]="a new label"`, the list box won't change. To trigger the synchronization, you should continue by :obj:`self.listLabels = self.listLabels`. This may seem awkward, but by our experience a list of selected items is seldom changed changed "per-item", so we were too lazy to write the annoyingly complex backward callbacks.

<span>
<span onclick="toggleVisibility(this);" class="hideshow">Show Example</span>
<span class="hideshow"><a href="gui_listbox.py">Download example (gui_listbox.py)</a></span>
<span class="hideshow"><a href="gui_listbox_attr.py">Download example (gui_listbox_attr.py)</a></span>


Spin
****

Spin control, a wrapper around QSpinBox.

spin(widget, master, value, min, max[, step, box, label, labelWidth, orientation, tooltip, callback, controlWidth])


min, max, step=1
Minimal and maximal value, and step.


Slider
******

A wrapper around QSlider that allows user setting a numerical value between the given bounds.

hSlider(widget, master, value[, box, minValue, maxValue, step, callback, labelFormat, ticks, divideFactor])


minValue (default: 0), maxValue (default: 10), step (default: 1)
Minimal and maximal value for the spin control, and its step.

ticks (default: 0)
If non-zero, it gives the interval between two ticks. The ticks will appear below the groove.

labelFormat (default: " %d")
Defines the look of the label on the righthand side of the slider. It has to contain one format character (like %d in the default), but can contain other text as well.

divideFactor (default: 1.0)
The value printed in the label is divided by :obj:`divideFactor`.


For an example of usage, see the second example in the description of <a href="#labels-example">labels</a>.


Check Box with Spin
*******************

Check box with spin, or, essentially, a wrapper around
OWGUI.checkBox and OWGUI.spin.

checkWithSpin(widget, master, label, min, max, checked, value[, posttext, step, tooltip, checkCallback, spinCallback, labelWidth])

min, max, step (required)
Minimal and maximal value for the spin control, and its step.

checked (required)
Master's attribute that is synchronized with the state of the check box.

value (required)
The attribute that is synchronized with the spin.

posttext (default: None)
Text which appears on the right-hand side of the control.

checkCallback (default: None), spinCallback (default: None)
Function that are called when the state of the check box or spin changes.


Labels
******

There are two functions for constructing labels. The first is a simple wrapper around QLabel which differs only in allowing to specify a fixed width without needing an extra line. Note that unlike most other OWGUI widgets, this one does not have the argument :obj:`master`.

widgetLabel(widget, label[, labelWidth])

The second is a label which can synchronize with values of master widget's attributes.

label(widget, master, label[, labelWidth])

label
:obj:`label` is a format string following Python's syntax (see the corresponding Python documentation): the label's content is rendered as :obj:`label % master.__dict__`.


*********
Utilities
*********

Widget box
**********


widgetBox(widget, box=None, orientation='vertical', addSpace=False)
Creates a box in which other widgets can be put. If :obj:`box` is given and not false, the box will be framed. If :obj:`box` is a string, it will be used for the box name (don't capitalize each word; spaces in front or after the string will be trimmed and replaced with a single space). Argument :obj:`orientation` can be :obj:`"vertical"` or :obj:`"horizontal"` (or :obj:`True` and :obj:`False`, or :obj:`1` and :obj:`0`, respectively).


Idented box
***********


indentedBox(widget, sep=20)
Creates an indented box. Widgets which are subsequently put into that box will be arranged vertically and aligned with an indentation of :obj:`sep`.


Inserting Space between Widgets
*******************************

Most widgets look better if we insert some vertical space between the controls
or groups of controls. A few functions have an optional argument :obj:`addSpace`
by which we can request such space to be added. For other occasions, we can use
the following two functions.

separator(widget, width=0, height=8)

Function :obj:`separator` inserts a fixed amount of space into :obj:`widget`.
Although the caller can specify the amount, leaving the default will help the
widgets having uniform look.

rubber(widget[, orientation="vertical"])

Similar to separator, except that the size is (1, 1) and that it expands in the
specified direction if the widget is expanded. Most widgets should have rubber
somewhere in their :obj:`controlArea`.

Attribute Icons
***************

getAttributeIcons()

Returns a dictionary with attribute types (:obj:`orange.VarTypes.Discrete`,
:obj:`orange.VarTypes.Continuous`, :obj:`orange.VarTypes.String`, -1) as keys
and colored pixmaps as values. The dictionary can be used in list and combo
boxes showing attributes for easier distinguishing between attributes of different types.

Send automatically / Send
*************************

Many widgets have a "Send" button (perhaps named "Apply", "Commit"...) accompanied with a check box "Send automatically", having the same effect as if the user pressed the button after each change. A well behaved widget cares to:

* disable the button, when the check box is checked;
* when the user checks the check box, the data needs to be send (or the changes applied), but only if there is any pending change which has not been (manually) sent yet.

Programming this into every widget is annoying and error-prone; at the time when the function described here was written, not many widgets actually did this properly.

setStopper(master, sendButton, stopCheckbox, changedFlag, callback)

sendButton
The button that will be disabled when the check box is checked.

stopCheckbox
Check box that decides whether the changes are sent/commited/applied automatically.

changedFlag
The name of the :obj:`master`'s attribute which tells whether there is a change which has not been sent/applied yet.

callback
The function that sends the data or applies the changes. This is typically the function which is also used as the :obj:`sendButton`'s callback.


:obj:`setStopper` is a trivial three lines long function which connects a few signals. Its true importance is in enforcing the correct procedure for implementing such button-check box combinations. Make sure to carefully observe and follow the example provided below.

