#############################################
Orange Widgets Reference Guide for Developers
#############################################

***********************************
Channels Definitions, Data Exchange
***********************************

Input and output channels are defined anywhere within the
:obj:`__init__` function of a main widget class. The definition
is used when running a widget, but also when registering your widget
within Orange Canvas. Channel definitions are optional, depending on
what your widget does.

Output Channels
***************

Following is an example that defines two output channels::

    self.outputs = [("Sampled Data", orange.ExampleTable), ("Learner", orange.Learner)]

:obj:`self.outputs` should thus be a list of tuples, within
each the first element is a name of the channel, and the second the
type of the tokens that will be passed through. Token types are class
names; most often these are some Orange classes, but they can also be
anything you may define as class in Python.

Widgets send the data by using :obj:`self.send` call,
like::

    self.send("Sampled Data", mydata)

Parameters of :obj:`send` are channel name and a token to be
send (e.g., a variable that holds the data to be send through the
channel).

When tokens are send around, the signaling mechanism annotates
them with a pointer to an object that sent the toke (e.g., a widget
id). Additionally, this annotation can be coupled with some name
passed to :obj:`send`, in case you have a widget that can send
few tokens one after the other and you would like to enable a receiving widget
recognize these are different tokens (and not updates of the same
one)::

    id = 10
    self.send("Sampled Data", mydata, id)

**************
Input Channels
**************

An example of the simplest definition of an input channel is::

    self.inputs = [("Data", orange.ExampleTable, self.receiveData)]

Again, :obj:`self.inputs` is a list of tuples, where the
elements are the name of the channel, followed by a channel type and a
Python function that will be called with any token received. For the
channel defined above, a corresponding receiving function would be of
the type (we would most often define it within the widget class
defintion, hence :obj:`self` for the first attribute)::

    def receiveData(self, data):
    # handle data in some way

Any time our widget would receive a token, :obj:`receiveData`
would be called. Notice there would be no way of knowing anything
about the sender of the token, hence widget would most often replace
the previously received token with the new one, and forget about the
old one.

Widgets can often clear their output by sending a :obj:`None`
as a token. Also, upon deletion of some widget, this is the way that
Orange Canvas would inform all directly connected downstream widgets
about deletion. Similar, when channels connecting two widgets are
deleted, Orange Canvas would automatically send :obj:`None` to
the receiving widget. Make sure your widget handles :obj:`None`
tokens appropriately!`

There are cases when widget would like to know about the origin of
a token. Say, you would like to input several learners to the
evaluation widget, how would this distinguish between the learners of
different origins? Remember (from above) that tokens are actually
passed around with IDs (pointers to widgets that sent them). To
declare a widget is interested about these IDs, one needs to define an
input channel in the following way::

    self.inputs = [("Learners", orange.Learner, self.learner, Multiple)]

where the last argument refers if we have a "Single" (default if not
specified) or a "Multiple" channel. For the above declared channel, the
receiving function should include an extra argument for the ID, like::

   def learner(self, learnertoken, tokenid):
   # handle learnertoken and tokeid in some way

Widgets such as :obj:`OWTestLearners` and alike use such
schema.

Finally, we may have input channels of the same type. If a widget
would declare input channels like::

    self.inputs = [("Data", orange.ExampleTable, self.maindata),
               ("Additional Data", orange.ExampleTable, self.otherdata)]

and we connect this widget in Orange Canvas to a sending widget
that has a single orange.ExampleTable output channel, Canvas would
bring up Set Channels dialog. There, a sending widget's channel could
be connected to both receiving channels. As we would often prefer to
connect to a single (default) channel instead (still allowing user of
Orange Canvas to set up a different schema manually), we set that channel
as the default. We do this by the using the fourth element in the channel
definition list, like::

    self.inputs = [("Data", orange.ExampleTable, self.maindata, Default),
               ("Additional Data", orange.ExampleTable, self.otherdata)]
