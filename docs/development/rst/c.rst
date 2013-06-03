################################
Writing Orange Extensions in C++
################################

This page is a draft of documentation for writing Orange extensions in C++. It has been written quite a while ago and never really polished, but we publish it due to requests and questions about the topic.

This page starts with a simple example of a C function that gets the data from Orange's example table and computes Delaunay triangulation for a chosen pair of (continuous) attributes. The result is returned as ``orange.Graph``. Then follows a more complex example that shows how to derive a new class from one of Orange's classes and make it accessible from Python. Finally, there's a more concise description of Orange's interface to Python and the tools we developed for making our lives easier.

***************************************
A simple example: Exporting function(s)
***************************************

If you plan to develop your own extensions for Orange, you'll need the Orange sources. Besides headers, you will certainly need the C files for stealing parts of the code from them. So download them if you haven't already done so. The example shown in this section is from directory orange/orangeom, files triangulate.cpp and orangeom.cpp.

We also assume that you have at least a vague idea about the Python API. If you don't, read the Python documentation first. Don't be afraid, API is fairly simple and very nicely organized.

We shall write a function that gets examples, indices/names/descriptors of two attributes and the number edge types. The two attributes need to be discrete; the function treats the values of these two attributes for each example as a point in a plane and computes the Delaunay triangulation, which is returned as ``orange.Graph``.

The function should be declared like this::

    PyObject *triangulate(PyObject *, PyObject *args, PyObject *)
        PYARGS(METH_VARARGS, "(examples[, attr1, attr2, nEdgeTypes]) -> Graph")

This all should be written in a single line. Function arguments and result are like for any Python function (as you have learned while reading Python's documentation), while the macro PYARGS marks the function for export to Python. The macro is dummy and serves only as a marker. The first argument tells Python about the types of arguments (again, you've seen the constant METH_VARARGS in Python's docs), and the second string describes the functions arguments and/or what the function does.

We shall close the function in ``PyTRY`` - ``PyCATCH`` block. This catches any C++ exceptions that Orange might throw and turns them into Python error messages. You should do this with any your function that calls orange's functions to be sure that you catch anything that's thrown from in there. ::

    {
        PyTRY
        ...
        PyCATCH
    }

Now for the real code. First we need to get the arguments and check their types. ::

    PExampleGenerator egen;
    int nEdgeTypes = 1;
    PyObject *pyvar1 = NULL, *pyvar2 = NULL;
    int attr1 = 0, attr2 = 1;
    if (!PyArg_ParseTuple(args, "O&|OOi:triangulate", pt_ExampleGenerator, &egen, &pyvar1, &pyvar2, &nEdgeTypes)
        || pyvar1 && !varNumFromVarDom(pyvar1, egen->domain, attr1)
        || pyvar2 && !varNumFromVarDom(pyvar2, egen->domain, attr2))
      return NULL;

    if ((egen->domain->attributes->at(attr1)->varType != TValue::FLOATVAR)
            || (egen->domain->attributes->at(attr2)->varType != TValue::FLOATVAR))
        PYERROR(PyExc_TypeError, "triangulate expects continuous attributes", NULL);

For the arguments of ``PyArg_ParseTuple`` see the corresponding Python documentation. Here we accept from one to four arguments. The first is a table of examples, which gets converted to ``PExampleGenerator`` by a converter function ``pt_ExampleGenerator``. The next two are read as Python objects and can be, as far as ``PyArg_ParseTuple`` is concerned, of any type. The last argument is an integer.

We use Orange's function ``varNumFromVarDom`` to convert what the user passed as an attribute to an attribute index. (This function is not documented anywhere, but you can discover such functions by looking at the orange's functions that do similar things as the function you plan to write.) If the argument is already an integer, it is returned as such; otherwise the example table's domain (``egen->domain``) is used to convert a name or an attribute descriptor to an integer index.

We then check the attribute's type and complain if they are not continuous. Macro ``PYERROR``, defined in Orange, is used to set the error type (``PyExc_TypeError``) and message, and to send the corresponding result (in Python ``NULL`` always signifies an error).

For the actual triangulation we shall use Wml, which expects the arguments in a structure ``Vector2&lt;float&gt;``. ::

    const int nofex = egen->numberOfExamples();

    Vector2<float> *points = new Vector2<float>[nofex];
    Vector2<float> *pi = points;

    PEITERATE(ei, egen) {
        if ((*ei)[attr1].isSpecial() || (*ei)[attr2].isSpecial())
            PYERROR(PyExc_AttributeError, "triangulate cannod handle unknown values", NULL);

        *(pi++) = Vector2<float>((*ei)[attr1].floatV, (*ei)[attr2].floatV);
    }

    int nTriangles;
    int *triangles, *adjacent;
    Delaunay2a<float> delaunay(nofex, points, nTriangles, triangles, adjacent);
    delete adjacent;

We have used a macro ``PEITERATE(ei, egen)`` for iteration across the example table, where ``egen`` is an example generator and ``ei`` is an iterator (declared by the macro). If we wanted, we could use an equivalent ``for`` loop: ``for(TExampleIterator ei(egen->begin()); ei; ++ei)``. Of the two results computed by ``Delaunay2a&lt;float&gt;``, we immediately discard ``adjacent``, while from ``triangles`` we shall construct the graph. ::

    TGraph *graph = new TGraphAsList(nofex, nEdgeTypes, 0);
    PGraph wgraph = graph;
    try {
        for(int *ti = triangles, *te = ti+nTriangles*3; ti!=te; ti+=3) {
            for(int se = 3; se--; ) {
                float *gedge = graph->getOrCreateEdge(ti[se], ti[(se+1) % 3]);
                for(int et = nEdgeTypes; et--; *gedge++ = 1.0);
            }
        }
    }
    catch (...) {
        delete triangles;
        throw;
    }
    delete triangles;

The ``graph`` is immediately wrapped into an instance of ``PGraph``. If anything fails, for instance, if an exception occurs in the code that follows, ``graph`` will be deallocated automatically so we don't need to (and even mustn't!) care about it. However, we have to be careful about the ``triangles``, we need another ``try``-``catch`` to deallocate it in case of errors. Then follows copying the data from ``triangles`` into ``graph`` (we won't explain it here, look at Wml's documentation if you are that curious).

``TGraph`` is Orange's class and ``PGraph`` is a wrapper around it. Python can use neither of them, the only type of result that the function can return is a Python object, so we conclude by::

    PyObject *res = WrapOrange(wgraph);
    return res;

(Don't worry about double wrappers. ``WrapOrange``, which wraps Orange objects into ``PyObject *`` technically does the opposite - it partially unwraps ``PGraph`` which already includes a ``PyObject *``. See the Orange's garbage collector, garbage.hpp, if you need to know.)

There's another detail in the actual function ``triangulate`` which is rather dirty and quite specific and rare, so we won't explain it here. If you're looking at triangulate.cpp and wonder what it means, just ignore it. It's not important.

We have omitted a few #includes above the function. We need a few wml headers. Besides that, all extensions should include orange_api.hpp that defines several macros and similar. In our case, we also needed to include graph.hpp and examplegen.hpp which define the classes used in our function.

What remains is to export this function to Python. We need to construct a new module, it's name will be ``orangeom`` and it will export the function triangulate. We've already mentioned that this has something to do with ``PYARGS`` macro. It has two arguments, the first, ``METH_VARARGS`` giving the number and type of arguments, while the second is the documentation string. Assuming that ``triangulate.cpp`` is in directory source/orangeom, so source/orange is its sibling directory, we need to run ../pyxtract/pyxtract.py like this::

    python ../pyxtract/pyxtract.py -m -n orangeom -l ../orange/px/stamp triangulate.cpp

Do yourself a favour and put this line into a batch file. See the ``_pyxtract.bat`` files in various Orange's directories.

Option ``-m`` tells pyxtract to ``m``ake the file, ``-n orangeom`` gives the name of the module to be produced and ``-l ../orange/px/stamp`` is similar to gcc's ``-l`` - it tells pyxtract to include a "library" with all Orange's objects (don't worry, it's not code, there are just things that pyxtract has written down for himself). Then follow the names of the files with any exported functions and classes. In our case, this is a single file, triangulate.cpp.

What do we get from pyxtract? A few files we don't need and one more we don't care about. The only file of real interest for us is initialization.px. You can see the file for yourself, but omitting a few unimportant points, it looks like this. ::

    PyMethodDef orangeomFunctions[] = {
        {"triangulate", (binaryfunc)triangulate, METH_VARARGS, "(examples[, attr1, attr2, nEdgeTypes]) -> Graph"},
        {NULL, NULL}
    };

    PyObject *orangeomModule;

    extern "C" ORANGEOM_API void initorangeom()
    {
        if (!initorangeomExceptions())
            return;

        gcorangeomUnsafeStaticInitialization();
        orangeomModule = Py_InitModule("orangeom", orangeomFunctions);
    }

``initorangeom`` is a function that will be called by Python when the module is initialized. Basically, it calls ``initorangeomExceptions``, ``gcorangeomUnsafeInitialization`` and then ``Py_InitModule`` whom it tells the module name and gives it the list of function that the module exports - in our case, ``orangeomFunctions`` has only a pointer to ``triangulate`` we wrote above.

Functions ``initorangeomExceptions``, ``gcorangeomUnsafeInitialization`` initialize the exceptions your module will throw to Python and initializes the stuff that cannot be initialized before the class types are ready. Both functions need to be provided by our module, but since we don't have any work for them, we'll just define them empty.

So, we need to write a file that will include initialization.px. It should be like this. ::

    #include "orange_api.hpp"

    #ifdef _MSC_VER
        #define ORANGEOM_API __declspec(dllexport)
    #else
        #define ORANGEOM_API
    #endif


    bool initExceptions()
    { return true; }

    void gcUnsafeStaticInitialization()
    {}

    #include "px/initialization.px"

We simplified the definition of ``ORANGEOM_API`` for this guide. If you ever wrote a DLL yourself, you probably know what the complete definition would look like, but for our module this will suffice. You can see the complete definition in ``orangeom.cpp`` if you want to. Then follow the two functions that we need to provide only because they are called by ``initorangeom``. Finally, we include initialization.px which takes care of the rest.

Setting the compiler options? Eh, just copy them from some project that is delivered together with Orange. If you want to do it yourself, this is the complete recipe (unless we've forgotten about something :)

* Add an environment variable ``PYTHON=c:\python23``, or wherever your
Python is installed. This will simplify the options and also help
you upgrade to a newer version. (If you don't want to do this,
just replace the below reference ``$(PYTHON)`` with ``c:\python23``.)

* Open the Orange workspace (sources/orange.dsw) and add your stuff as new projects.
Add new project into workspace". You need a "Win32 Dynamic-link
Library"; create it as an empty or simple project.
This document will suppose you've put it into a subdirectory
of ``orange/source`` (eg ``orange/source/myproject``)

* Edit the project settings. Make sure to edit the settings for
both Release and Debug version - or for Release, in the unlikely case that you won't
need to debug.
  * In C/C++, Preprocessor add include directories ``../include,../orange,px,ppp,$(PYTHON)/include``.
      If VC later complains that it cannot find Python.h, locate the
      file yourself and fix the last include directory accordingly.
  

  * In C/C++, C++ Language, check Enable Run-Time Type Information.

  * In C/C++, Code generation, under Use run-time library choose
      Multithread DLL for Release and Debug Multithread DLL for Debug
      version.

  * In Link, Input, add ``$(PYTHON)\libs`` and ``../../lib`` to
      Additional Library Path. (The first is the path to Python.lib,
      and the second to orange.lib; locate them manually if you need to.)

  * In Link, General, change the Output file name to ../../mymodule.pyd for the Release Build and to ../../mymodule_d.pyd for debug. You can use .dll instead of .pyd.

  * In Post-build step, add "copy Release\orangeom.lib ..\..\lib\orangeom.lib" for the Release version, and "copy Debug\orangeom_d.lib ..\..\lib\orangeom_d.lib" for the Debug.

  * In Debug Build, go to tab Debug, General, set the Executable for debug session to "c:\python23\python_d.exe".

##################################################################
General mechanism and tools used in Orange's C to Python interface
##################################################################

This page is not meant to be a comprehensive and self-sufficient guide to exporting C++ classes into Python using the Orange's machinery. To learn how to export your functions and classes to Python, you should open some Orange's files (lib_*.cpp and cls_*.cpp) and search for examples there, while this page will hopefully help you to understand them. The easiest way to write your interfaces will be to copy and modify the existing Orange's code. (This is what we do all the time. :)

If you are writing extension modules for Python, you by no means *have to* use Orange's scripts (pyxtract, pyprops) compiling the interface. Compile your interface with any tool you want, e.g. Swig. The only possible complications would arise if you are deriving and exporting new C++ classes from Orange's classes (to those that know something on the topic: Orange uses a slightly extended version of ``PyTypeObject``, and the derived classes can't return to the original). If need be, ask and we may try to provide some more insight and help to overcome them.

################################
Orange's C++ to Python interface
################################

Instead of general 3rd party tools (Swig, Sip, PyBoost...) for interfacing between C++ and Python, Orange uses it's own set of tools. (We are working towards making them general, ie also useful for other applications.) These tools (two Python scripts, actually, pyprops and pyxtract) require more manual programming than other tools, but on the other hand, the result is a tighter and nicer coupling of Orange with Python.

In short, to expose a C++ object to Python, we mark the attributes to be exported by a comment line starting with ``//P``. Besides, we need to either declare which general constructor to use or program a special one (this constructor will be in place of what would in Python be defined as the function __new__), and program the interfaces to those C++ member functions that we want exported. In order for pyxtract to recognize them, the function name should be composed of the class name and method name, separated by an underscore, and followed by a certain keyword. When we simply give the access to unaltered C++ functionality, the interface functions will only have a few-lines. When we want to make the "Python" version of the function more friendly, eg. allow various types of arguments or fitting the default arguments according to the given, these functions will be longer, but the pay-off is evident. We argue that a few-line function is not more of inconvenience than having to write export declarations (as is the case with Sip at least, I guess).

To define a non-member function, we write the function itself according to the instructions in Python's manual (see the first chapter of "Extending and Embedding the Python Interpreter") and then mark it with a specific keyword. Pyxtract will recognize the keyword and add it to the list of exported functions.

Orange's core C++ objects are essentially unaware of Python above them. However, to facilitate easier interface with Python, each Orange class contains a static pointer to a list of its ``properties'' (attributes, in Python terminology). Accessing the object's attributes from Python goes through that list. These lists, however, are general and would be equally useful if we were to interface Orange to some other language (eg Perl) or technology (ActiveX, Corba).

The only exception to the Orange's independency of Python is garbage collection: Orange uses Python's garbage collection for the sake of efficiency and simplicity. Each Orange's pointer (except for the short-term ones) is wrapped into a wrapper of type ``PyObject *``. Dependency of Orange on Python is not strong - if we wanted to get rid of it, we'd only need to write our own garbage collection (or steal the Python's). ``PyObject *`` is the basic Python's type which stores some garbage collection related stuff, a pointer to the class type (``PyTypeObject *``) and the class specific data. The specific data is, in Orange's case, a pointer to the Orange object. Class type is a structure that contains the class name, pointers to function that implement the class special methods (such as indexing, printing, memory allocation, mathematical operations) and class members.

We won't go deeper into explaining ``PyTypeObject`` since this is done in Python documentation. What you need to know is that for each Orange class that is accessible from Python, there is a corresponding ``PyTypeObject`` that defines its methods. For instance, the elements of ``ExampleTable`` (examples) can be accessed through indexing because we defined a C function that gets an index (and the table, of course) and returns the corresponding example, and we've put a pointer to this method into the ``ExampleTable``'s ``PyTypeObject`` (actually, we didn't do it manually, this is what pyxtract is responsible for). This is equivalent to overloading the operator [] in C++. Here's the function (with error detection removed for the sake of clarity). ::

    PyObject *ExampleTable_getitem_sq(PyObject *self, int idx)
    {
        CAST_TO(TExampleTable, table);
        return Example_FromExampleRef((*table)[idx], EXAMPLE_LOCK(PyOrange_AsExampleTable(self)));
    }

Also, ``ExampleTable`` has a method ``sort([list-of-attributes])``. This is implemented through a C function that gets a list of attributes and calls the C++ class' method ``TExampleTable::sort(const vector&lt;int&gt; order)``. To illustrate, this is a slightly simplified function (we've removed some flexibility regarding the parameters and the exception handling). ::

    PyObject *ExampleTable_sort(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
    {
        CAST_TO(TExampleTable, table);

        if (!args || !PyTuple_Size(args)) {
            table->sort();
            RETURN_NONE;
        }

        TVarList attributes;
        varListFromDomain(PyTuple_GET_ITEM(args, 0), table->domain, attributes, true, true);
        vector<int> order;
        for(TVarList::reverse_iterator vi(attributes.rbegin()), ve(attributes.rend()); vi!=ve; vi++)
            order.push_back(table->domain->getVarNum(*vi));

        table->sort(order);
        RETURN_NONE;
    }

Basically, what the function does is it casts the ``PyObject *`` into the corresponding C++ object ("unwrapping"), reads the arguments, calls the C++ functions and returns the result (``None``, in this case).

There seem to be a lot of manual work involved in interfacing. Indeed there is, but this is exactly why Orange is so user friendly. The manual control of argument conversion gives a programmer the opportunity to program a function which accepts many different forms of arguments. The above function, for instances, can accept the list where the attributes can be specified by indices, names or descriptors, all corresponding to the ``ExampleTable`` which is being sorted. Inheritance of methods, on the other hand, ensures that only the methods that are truly specific for a class need to be coded.


The part of the interface that is programmed automatically is taken care of by two scripts. ``pyprops.py`` parses all Orange's header files and extracts all the class built-in properties. The result are lists of properties (attributes); nothing else needs to be done for the ``ExampleTable``'s attribute ``domain`` to be visible in Python, except for putting a ``// P`` after its declaration.

The second script is ``pyxtract.py``. It goes through the C++ files that contain the interface functions, such as those above. It recognizes the functions that implement special or member methods and constructs the corresponding ``PyTypeObject``s. It therefore relieves us from the most boring part of work, but permits us to program things like more intelligent arguments or error handling.

*******
pyprops
*******

Pyprops scans each hpp file for classes we want to export to Python (or, in general, some other scripting language). Properties can be ``bool``, ``int``, ``float``, ``string``, ``TValue`` or a wrapped Orange type.

Pyprops' parser is essentially a trivial finite automaton. Class definition needs to look as follows. ::

    class [ORANGE_API] <classname>; [: public <parentclass> ]

This should be in a single line. To mark the class for export, this should be followed by ``__REGISTER_CLASS`` or ``__REGISTER_ABSTRACT_CLASS`` before any properties or components are defined. The difference between the two, as far as pyprops is concerned, is that abstract classes don't define the ``clone`` method.

To export a property, it should be defined like this. ::

    <type> <name> //P[R|O] [>|+<alias>] <description>

Pyprops doesn't check the type and won't object if you use other types than those listed above. Linker will complain about missing symbols, though. ``//P`` signals that we want to export the property. If followed by R or O, the property is read-only or obsolete. The property can also have an alias name; > renames it and + adds an alias. Description is not used at the time, but it is nevertheless a good practice to provide it.

Each property needs to be declared in a separate line, e.g. ::

    int x; //P;
    int y; //P;

If we don't want to export a certain property, we don't need to. Just omit the ``//P``. An exception to this are wrapped Orange objects: for instance, if a class has a (wrapped) pointer to the domain, ``PDomain`` and it doesn't export it, pyxtract should still now about them because of the cyclic garbage collection. You should mark them by ``//C`` so that they are put into the list of objects that need to be counted. If you fail to do so, you'll have a memory leak. Luckily, this is a very rare situation; there are only two such components in relief.hpp.

If a class directly or indirectly holds references to any wrapped objects that are neither properties nor components, it will need to declare ``traverse`` and ``clear`` to include them in the garbage collection. Python documentation will tell you what these functions need to do, and you can look at several instances where we needed them in Orange.

Pyprops creates a ppp file for each hpp. The ppp file first ``#include``s the corresponding hpp file and then declares the necessary definition for each exported file. A list of properties store their names, descriptions, typeid's (RTTI), a class description for the properties' type, the properties' offset and the flags denoting read-only and obsolete properties.

Then comes a list of components' offsets, followed by a definition of classes static field ``st_classDescription`` and a virtual function ``classDescription`` that returns a pointer to it. Finally, if the class is not abstract, a virtual function ``clone`` is defined that returns a ``new`` instance of this class initialized, through a copy constructor, with an existing one.

ppp file contains definitions, so it has to be compiled only once. The most convenient way to do it is to include it in the corresponding cpp file. For instance, while many Orange's cpp files include domain.hpp, only domain.cpp includes domain.ppp instead.

********
pyxtract
********

Pyxtract's job is to detect the functions that define special methods (such as printing, conversion, sequence and arithmetic related operations...) and member functions. Based on what it finds for each specific class, it constructs the corresponding ``PyTypeObject``s. For the functions to be recognized, they must follow a specific syntax.

There are two basic mechanisms used. Special functions are recognized by their definition (they need to return ``PyObject *``, ``void`` or ``int`` and their name must be of form &lt;classname&gt;_&lt;functionname&gt;). Member functions, inheritance relations, constants etc. are marked by macros such as ``PYARGS`` in the above definition of ``ExampleTable_sort``. These macros usually don't do anything, so C++ compiler ignores them - they are just markups for pyxtract.

Class declaration
=================

Each class needs to be declared: pyxtract must know from which parent class the class it's derived. If it's a base class, pyxtract need to know the data structure for the instances of this class. As for all Python objects the structure must be "derived" from ``PyObject`` (the quotation mark are here because Python is written in C, so the subclasses are not derived in the C++ sense). Most objects are derived from Orange; the only classes that are not are ``orange.Example``, ``orange.Value`` and ``orange.DomainDepot`` (I forgot why the depot, but there had to be a strong reason).

Pyxtract should also know how the class is constructed - it can have a specific constructor, one of the general constructors or no constructor at all.


The class is declared in one of the following ways.

``BASED_ON(EFMDataDescription, Orange)``
    This tells pyxtract that ``EFMDataDescription`` is an abstract class derived from ``Orange``: there is no constructor for this class in Python, but the C++ class itself is not abstract and can appear and be used in Python. For example, when we construct an instance of ``ClassifierByLookupTable`` with more than three attributes, an instance of ``EFMDataDescription`` will appear in one of its fields.

``ABSTRACT(ClassifierFD, Classifier)``
    This defines an abstract class, which will never be constructed in the C++ code and not even pretend to be seen in Python. At the moment, the only difference between this ``BASED_ON`` and ``ABSTRACT`` is that the former can have pickle interface, while the latter don't need one. 

Abstract C++ classes are not necessarily defined as ``ABSTRACT`` in the Python interface. For example, ``TClassifier`` is an abstract C++ class, but you can seemingly construct an instance of ``Classifier`` in Python. What happens is that there is an additional C++ class ``TClassifierPython``, which poses as Python's class ``Classifier``. So the Python class ``Classifier`` is not defined as ``ABSTRACT`` or ``BASED_ON`` but using the ``Classifier_new`` function, as described below.


``C_NAMED(EnumVariable, Variable, "([name=, values=, autoValues=, distributed=, getValueFrom=])")``
    ``EnumVariable`` is derived from ``Variable``. Pyxtract will also create a constructor which will as an optional argument accept the object's name. The third argument is a string that describes the constructor, eg. gives a list of arguments. IDEs for Python, such as PythonWin, will show this string in a balloon help while the programmer is typing.

``C_UNNAMED(RandomGenerator, Orange, "() -> RandomGenerator")``
    This is similar as ``C_NAMED``, except that the constructor accepts no name. This form is rather rare since all Orange objects can be named.

``C_CALL(BayesLearner, Learner, "([examples], [weight=, estimate=] -/-> Classifier")``
    ``BayesLearner`` is derived from ``Learner``. It will have a peculiar constructor. It will, as usual, first construct an instance of ``BayesLearner``. If no arguments are given (except for, possibly, keyword arguments), it will return the constructed instance. Otherwise, it will call the ``Learner``'s call operator and return its result instead of ``BayesLearner``.``

``C_CALL3(MakeRandomIndices2, MakeRandomIndices2, MakeRandomIndices, "[n | gen [, p0]], [p0=, stratified=, randseed=] -/-> [int]")``
    ``MakeRandomIndices2`` is derived from ``MakeRandomIndices`` (the third argument). For a contrast from the ``C_CALL`` above, the corresponding constructor won't call ``MakeRandomIndices`` call operator, but the call operator of ``MakeRandomIndices2`` (the second argument). This constructor is often used when the parent class doesn't provide a suitable call operator.

``HIDDEN(TreeStopCriteria_Python, TreeStopCriteria)``
    ``TreeStopCriteria_Python`` is derived from ``TreeStopCriteria``, but we would like to hide this class from the user. We use this definition when it is elegant for us to have some intermediate class or a class that implements some specific functionality, but don't want to bother the user with it. The class is not completely hidden - the user can reach it through the ``type`` operator on an instance of it. This is thus very similar to a ``BASED_ON``.

``DATASTRUCTURE(Orange, TPyOrange, orange_dict)``
    This is for the base classes. ``Orange`` has no parent class. The C++ structure that stores it is ``TPyOrange``; ``TPyOrange`` is essentially ``PyObject`` (again, the structure always has to be based on ``PyObject``) but with several additional fields, among them a pointer to an instance of ``TOrange`` (the C++ base class for all Orange's classes). ``orange_dict`` is a name of ``TPyOrange``'s field that points to a Python dictionary; when you have an instance ``bayesClassifier`` and you type, in Python, ``bayesClassifier.someMyData=15``, this gets stored in ``orange_dict``. The actual mechanism behind this is rather complicated and you most probably won't need to use it. If you happen to need to define a class with ``DATASTRUCTURE``, you can simply omit the last argument and give a 0 instead.

``PyObject *ClassifierByLookupTable1_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(ClassifierByLookupTable, "(class-descriptor, descriptor)")``
    ``ClassifierByLookupTable1_new`` has a specific constructor. The general ones above couldn't be used since this constructor needs to have some arguments or for some other reason (eg. the user doesn't need to specify the arguments, but the C++ constructor for the corresponding C++ class requires them, so this interface function will provide the defaults). The constructor function needs to be defined like above (ie. &lt;classname&gt;_new), followed by a ``BASED_ON`` macro giving the parent class and the comment. The declaration must be written in a single line. Just for the illustration, the (simplified, no error handling) constructor looks like this ::

    PyObject *ClassifierByLookupTable1_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(ClassifierByLookupTable, "(class-descriptor, descriptor)")
    {
        PyTRY
            PVariable vcl, vvl;
            PyObject *pyvlist = NULL, *pydlist = NULL;
            if (!PyArg_ParseTuple(args, "O&O&|OO", cc_Variable, &vcl, cc_Variable, &vvl, &pyvlist, &pydlist))
                PYERROR(PyExc_TypeError,
                    "invalid parameter; two variables and, optionally, ValueList and DistributionList expected",
                     NULL);

            TClassifierByLookupTable1 *cblt = mlnew TClassifierByLookupTable1(vcl, vvl);
            return initializeTables(pyvlist, pydlist, cblt) ? WrapNewOrange(cblt, type) : NULL;
        PyCATCH
    }

Function parses the arguments by calling ``PyArg_ParseTuple``, constructs an instance of ``ClassifierByLookupTable1``, initializes it and returns either the constructed object or NULL which signifies an error. ``PyTRY`` and ``PyCATCH`` catch the possible Orange's exception and turn them into Python exceptions.

Even if the class is defined by ``DATASTRUCTURE``, you can still specify a different constructor, most probably the last form of it (the ``_new`` function). In this case, specify a keyword ``ROOT`` as a parent and pyxtract will understand that this is the base class.

Object construction in Python is divided between two methods. The constructors we discussed above construct the essential part of the object - they allocate the necessary memory and initialize the fields far enough that the object is valid to enter the garbage collection. The second part is handled by the ``init`` method. It is, however, not forbidden to organize the things so that ``new`` does all the job. This is also the case in Orange. The only task left for ``init`` is to set any attributes that user gave as the keyword arguments to the constructor.

For instance, Python's statement ``orange.EnumVariable("a", values=["a", "b", "c"])`` is executed so that ``new`` constructs the variable and gives it the name, while ``init`` sets the ``values`` field. You don't need to do anything about it.

The ``new`` operator, however, sometimes also accepts keyword arguments. For instance, when constructing an ``ExampleTable`` by reading the data from file, you can specify a domain (using keyword argument ``domain``), a list of attributes to reuse if possible (``use``), you can tell it not to reuse the stored domain or not to store the newly constructed domain (``dontCheckStored``, ``dontStore``). After the ``ExampleTable`` is constructed, ``init`` is called to set the attributes. To tell it to ignore the keyword arguments that the constructor might (or had) used, we write the following. ::

    CONSTRUCTOR_KEYWORDS(ExampleTable, "domain use useMetas dontCheckStored dontStore filterMetas")

``init`` will ignore all the keywords from the list.

Talking about attributes, here's another macro. You know you can assign arbitrary attributes to Orange classes. Let ``ba`` be an orange object, say ``orange.BayesLearner``. If you assign new attributes as usual directly, eg. ``ba.myAttribute = 12``, you will get a warning (you should use the object's method ``setattr(name, value)`` to avoid it). Some objects have some attributes that cannot be implemented in C++ code, yet they are usual and useful. For instance, ``Graph`` can use attributes ``objects``, ``forceMapping`` and ``returnIndices``, which can only be set from Python (if you take a look at the documentation on ``Graph`` you will see why these cannot be implemented in C++). Yet, since user are allowed to set these attributes and will do so often, we don't want to give warnings. We achieve this by ::

    RECOGNIZED_ATTRIBUTES(Graph, "objects forceMapping returnIndices")

How do these things function? Well, it is not hard to imagine: pyxtract catches all such exceptions and stores the corresponding lists for each particular class. The ``init`` constructor then checks the list prior to setting attributes. Also the method for setting attributes that issues warnings for unknown attributes checks the list prior to complaining.


Special methods
===============

Special methods act as the class built-in methods. They define what the type can do: if it, for instance, supports multiplication, it should define the operator that gets the object itself and another object and return the product (or throw an exception). If it allows for indexing, it defines an operator that gets the object itself and the index and returns the element. These operators are low-level; most can be called from Python scripts but they are also internally by Python. For instance, if ``table`` is an ``ExampleTable``, then ``for e in table:`` or ``reduce(f, table)`` will both work by calling the indexing operator for each table's element.

We shall here avoid the further discussion of this since the topic is adequately described in Python manuals (see "Extending and Embedding the Python Interpreter", chapter 2, "Defining New Types").

To define a method for Orange class, you need to define a function named, ``&lt;classname&gt;_&lt;methodname&gt;``; the function should return either ``PyObject *``, ``int`` or ``void``. The function's head has to be written in a single line. Regarding the arguments and the result, it should conform to Python's specifications. Pyxtract will detect the methods and set the pointers in ``PyTypeObject`` correspondingly.

Here's a list of methods: the left column represents a method name that triggers pyxtract (these names generally correspond to special method names of Python classes as a programmer in Python sees them) and the second next is the name of the field in ``PyTypeObject`` or subjugated structures. See Python documentation for description of functions' arguments and results. Not all methods can be directly defined; for those that can't, it is because we either use an alternative method (eg. ``setattro`` instead of ``setattr``) or pyxtract gets or computes the data for this field in some other way. If you really miss something ... wasn't the idea of open source that you are allowed to modify the code (e.g. of pyxtract) yourself?

General methods
---------------

+--------------+-----------------------+-----------------------------------------------------------+
| pyxtract     | PyTypeObject          |                                                           |
+==============+=======================+===========================================================+
| ``dealloc``  | ``tp_dealloc``        | Frees the memory occupied by the object. You will need to |
|              |                       | define this for the classes with a new ``DATASTRUCTURE``; |
|              |                       | if you only derive a class from some Orange class, this   |
|              |                       | has been taken care of. If you have a brand new object,   |
|              |                       | copy the code of one of Orange's deallocators.            |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_print``          | This function can't be redefined as it seem to crash      |
|              |                       | Python (due to difference in compilers?!)                 |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_getattr``        | Can't be redefined since we use ``tp_getattro`` instead.  |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_setattr``        | Can't be redefined since we use ``tp_setattro`` instead.  |
+--------------+-----------------------+-----------------------------------------------------------+
| ``cmp``      | ``tp_compare``        |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``repr``     | ``tp_repr``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``as_number``         | (pyxtract will initialize this field if you give any of   |
|              |                       | the methods from the number protocol; you needn't care    |
|              |                       | about this field)                                         |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``as_sequence``       | (pyxtract will initialize this field if you give any of   |
|              |                       | the methods from the sequence protocol)                   |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``as_mapping``        | (pyxtract will initialize this field if you give any of   |
|              |                       | the methods from the mapping protocol)                    |
+--------------+-----------------------+-----------------------------------------------------------+
| ``hash``     | ``tp_hash``           | Class ``Orange`` computes a hash value from the pointer;  |
|              |                       | you don't need to overload it if your object inherits the |
|              |                       | function. If you write an independent class, just copy the|
|              |                       | code.                                                     |
+--------------+-----------------------+-----------------------------------------------------------+
| ``call``     | ``tp_call``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``call``     | ``tp_call``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``str``      | ``tp_str``            |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``getattr``  | ``tp_getattro``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``setattr``  | ``tp_setattro``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_as_buffer``      | Pyxtract doesn't support the buffer protocol.             |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_flags``          | Flags are set by pyxtract.                                |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_doc``            | Documentation is read from the constructor definition     |
|              |                       | (see above).                                              |
+--------------+-----------------------+-----------------------------------------------------------+
| ``traverse`` | ``tp_traverse``       | Traverse is tricky (as is garbage collection in general). |
|              |                       | There's something on it in a comment in root.hpp; besides |
|              |                       | that, study the examples. In general, if a wrapped member |
|              |                       | is exported to Python (just as, for instance,             |
|              |                       | ``Classifier`` contains a ``Variable`` named              |
|              |                       | ``classVar``), you don't need to care about it. You should|
|              |                       | manually take care of any wrapped objects not exported to |
|              |                       | Python. You probably won't come across such cases.        |
+--------------+-----------------------+-----------------------------------------------------------+
| ``clear``    | ``tp_clear``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``richcmp``  | ``tp_richcmp``        |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_weaklistoffset`` |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``iter``     | ``tp_iter``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``iternext`` | ``tp_iternext``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_methods``        | Set by pyxtract if any methods are given.                 |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_members``        |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``getset``            | Pyxtract initializes this by a pointer to manually        |
|              |                       | written getters/setters (see below).                      |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_base``           | Set by pyxtract to a class specified in constructor       |
|              |                       | (see above).                                              |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_dict``           | Used for class constants (eg. ``Classifier.GetBoth``)     |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_descrget``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_descrset``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_dictoffset``     | Set by pyxtract to the field given in ``DATASTRUCTURE``   |
|              |                       | (if there is any).                                        |
+--------------+-----------------------+-----------------------------------------------------------+
| ``init``     | ``tp_init``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_alloc``          | Set to ``PyType_GenericAlloc``                            |
+--------------+-----------------------+-----------------------------------------------------------+
| ``new``      | ``tp_new``            |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_free``           | Set to ``_PyObject_GC_Del``                               |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_is_gc``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_bases``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_mro``            |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_cache``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_subclasses``     |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_weaklist``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+

Numeric protocol
----------------

+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``add``    |  ``nb_add``      | ``pow``     | ``nb_power``    | ``lshift`` | ``nb_lshift`` | ``int``   | ``nb_int``   |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``sub``    | ``nb_subtract``  | ``neg``     | ``nb_negative`` | ``rshift`` | ``nb_rshift`` | ``long``  | ``nb_long``  |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``mul``    | ``nb_multiply``  | ``pos``     | ``nb_positive`` | ``and``    | ``nb_and``    | ``float`` | ``nb_float`` |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``div``    | ``nb_divide``    | ``abs``     | ``nb_absolute`` | ``or``     | ``nb_or``     | ``oct``   | ``nb_oct``   |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``mod``    | ``nb_remainder`` | ``nonzero`` | ``nb_nonzero``  | ``coerce`` | ``nb_coerce`` | ``hex``   | ``nb_hex``   |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``divmod`` | ``nb_divmod``    | ``inv``     | ``nb_invert``   |            |               |           |              |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+

Sequence protocol
-----------------

+----------------+---------------+----------------+------------------+
| ``len_sq``     | ``sq_length`` | ``getslice``   | ``sq_slice``     |
+----------------+---------------+----------------+------------------+
| ``concat``     | ``sq_concat`` | ``setitem_sq`` | ``sq_ass_item``  |
+----------------+---------------+----------------+------------------+
| ``repeat``     | ``sq_slice``  | ``setslice``   | ``sq_ass_slice`` |
+----------------+---------------+----------------+------------------+
| ``getitem_sq`` | ``sq_item``   | ``contains``   | ``sq_contains``  |
+----------------+---------------+----------------+------------------+

Mapping protocol
----------------

+-------------+----------------------+
| ``len``     | ``mp_length``        |
+-------------+----------------------+
| ``getitem`` | ``mp_subscript``     |
+-------------+----------------------+
| ``setitem`` | ``mp_ass_subscript`` |
+-------------+----------------------+

For example, here's what gets called when you want to know the length of an example table. ::

    int ExampleTable_len_sq(PyObject *self)
    {
        PyTRY
            return SELF_AS(TExampleGenerator).numberOfExamples();
        PyCATCH_1
    }

``PyTRY`` and ``PyCATCH`` take care of C++ exceptions. ``SELF_AS`` is a macro for casting, ie unwrapping the points (this is an alternative to ``CAST_TO`` you've seen earlier).


Getting and Setting Class Attributes
====================================

Exporting of most of C++ class fields is already taken care by the lists that are compiled by pyprops. There are only a few cases in the entire Orange where we needed to manually write specific handlers for setting and getting the attributes. This needs to be done if setting needs some special processing or when simulating an attribute that does not exist in the underlying C++ class.

An example for this is class ``HierarchicalCluster``. It contains results of a general, not necessarily binary clustering, so each node in the tree has a list ``branches`` with all the node's children. Yet, as the usual clustering is binary, it would be nice if the node would also support attributes ``left`` and ``right``. They are not present in C++, but we can write a function that check the number of branches; if there are none, it returns ``None``, if there are more than two, it complains, while otherwise it returns the first branch. ::

    PyObject *HierarchicalCluster_get_left(PyObject *self)
    {
        PyTRY
            CAST_TO(THierarchicalCluster, cluster);

            if (!cluster->branches)
                RETURN_NONE

            if (cluster->branches->size() > 2)
                PYERROR(PyExc_AttributeError,
                        "'left' not defined (cluster has more than two subclusters)",
                        NULL);

            return WrapOrange(cluster->branches->front());
        PyCATCH
    }

As you can see from the example, the function needs to accept a ``PyObject *`` (the object it``self``) and return a ``PyObject *`` (the attribute value). The function name needs to be ``&lt;classname&gt;_get_&lt;attributename&gt;``. Setting an attribute is similar; function name should be ``&lt;classname&gt;_set_&lt;attributename&gt;``, it should accept two Python objects (the object and the attribute value) and return an ``int``, where 0 signifies success and -1 a failure.

If you define only one of the two handlers, you'll get a read-only or write-only attribute.


Member functions
================

You've already seen an example of a member function - the ``ExampleTable``'s method ``sort``. The general template is ``PyObject *&lt;classname&gt;_&lt;methodname&gt;(&lt;arguments&gt;) PYARGS(&lt;arguments-keyword&gt;, &lt;documentation-string&gt;)``. In the case of the ``ExampleTable``'s ``sort``, this looks like this. ::

    PyObject *ExampleTable_sort(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")

Argument type can be any of the usual Python constants stating the number and the kind of arguments, such as ``METH_VARARGS`` or ``METH_O`` - this constant gets copied to the corresponding list (browse Python documentation for ``PyMethodDef``).

If you want more examples, just search Orange's files for the keyword ``PYARGS``.

Class constants
===============

Orange classes, as seen from Python, can also have constants, such as ``orange.Classifier.GetBoth``. Classifier's ``GetBoth`` is visible as a member of the class, the derived classes and all their instances (eg. ``BayesClassifier.GetBoth`` and ``bayes.GetBoth``).

There are several ways to define such constants. If they are simple integers or floats, you can use ``PYCLASSCONSTANT_INT`` or ``PYCLASSCONSTANT_FLOAT``, like in ::

    PYCLASSCONSTANT_INT(Classifier, GetBoth, 2)

You can also use the enums from the class, like ::

    PYCLASSCONSTANT_INT(C45TreeNode, Leaf, TC45TreeNode::Leaf)

Pyxtract will convert the given constant to a Python object (using ``PyInt_FromLong`` or ``PyFloat_FromDouble>``).

When the constant is an object of some other type, use ``PYCLASSCONSTANT``. In this form (not used in Orange so far), the third argument can be either an instance of ``PyObject *`` or a function call. In either case, the object or function must be known at the point where the pyxtract generated file is included.


Pickling
========

Pickling is taken care of automatically if the class provides a Python constructor which can construct the object without arguments (it may *accept* arguments, but should be able to do without them. If there is no such constructor, the class should provide a ``__reduce__`` method or it should explicitly declare that it cannot be pickled. If it doesn't pyxtract will issue a warning that the class will not be picklable.

Here are the rules:

* Classes that provide a ``__reduce__`` method (details follow below) are pickled through that method.
* Class ``Orange``, the base class, already provides a ``__reduce__`` method, which is only useful if the constructor accepts empty arguments. So, if the constructor is declared as ``C_NAMED``, ``C_UNNAMED``, ``C_CALL`` or ``C_CALL3``, the class is the class will be picklable. See the warning below.
* If the constructor is defined by ``_new`` method, and the ``BASED_ON`` definition is followed be ``ALLOWS_EMPTY``, this signifies that it accepts empty arguments, so it will be picklable just as in the above point. For example, the constructor for the class ``DefaultClassifier`` is defined like this ::

    PyObject *DefaultClassifier_new(PyTypeObject *tpe, PyObject *args)
        BASED_ON(Classifier, "([defaultVal])") ALLOWS_EMPTY
  and is picklable through code ``Orange.__reduce__``. But again, see the warning below.

* If the constructor is defined as ``ABSTRACT``, there cannot be any instances of this class, so pyxtract will give no warning that it is not picklable.
* The class can be explicitly defined as not picklable by ``NO_PICKLE`` macro, as in ::

    NO_PICKLE(TabDelimExampleGenerator)
  Such classes won't be picklable even if they define the appropriate constructors. This effectively defined a ``__reduce__`` method which yields an exception; if you manually provide a ``__reduce__`` method for such a class, pyxtract will detect that the method is multiply defined.
* If there are no suitable constructors, no ``__reduce__`` method and no ``ABSTRACT`` or ``NO_PICKLE`` flag, pyxtract will warn you about that.

When the constructor is used, as in points 2 and 3, pickling will only work if all fields of the C++ class can be set "manually" from Python, are set through the constructor, or are set when assigning other fields (search the source code for the ``afterSet`` method). In other words, if there are fields that are not marked as ``//P`` for pyprops, you will most probably need to manually define a ``__reduce__`` method, as in point 1.

The details of what the ``__reduce__`` method must do are described in the Python documentation. In our circumstances it can be implemented in two ways which differ in what function is used for unpickling: it can either use the class' constructor or we can define a special method for unpickling.

The former usually happens when the class has a read-only property (``//PR``) which is set by the constructor. For instance, ``AssociationRule`` has read-only fields ``left`` and ``right``, which are needs to be given to the constructor. This is the ``__reduce__`` method for the class. ::

    PyObject *AssociationRule__reduce__(PyObject *self)
    {
        PyTRY
            CAST_TO(TAssociationRule, arule);
            return Py_BuildValue("O(NN)N", self->ob_type,
                                       Example_FromWrappedExample(arule->left),
                                       Example_FromWrappedExample(arule->right),
                                       packOrangeDictionary(self));
        PyCATCH
    }

As you can learn from the Python documentation, the ``__reduce__`` should return the tuple in which the first element is the function that will do the unpickling, and the second argument are the arguments for that function. Our unpickling function is simply the classes' type (calling a type corresponds to calling a constructor) and the arguments for the constructor are the left- and right-hand side of the rule. The third element of the tuple is classes' dictionary.

When unpickling is more complicated, usually when the class has no constructor and contains fields of type ``float *`` or similar, we need a special unpickling function. The function needs to be directly in the modules' namespace (it cannot be a static method of a class), so we named them ``__pickleLoader&lt;classname&gt;``. Search for examples of such functions in the source code; note that the instance's true class need to be pickled, too. Also, check how we use ``TCharBuffer`` throughout the code to store and pickle binary data as Python strings.

Be careful when manually writing the unpickler: if a C++ class derived from that class inherits its ``__reduce__``, the corresponding unpickler will construct an instance of a wrong class (unless the unpickler functions through Python's constructor, ``ob_type->tp_new``). Hence, classes derived from a class which defines an unpickler have to define their own ``__reduce__``, too.

Non-member functions and constants
==================================

Most Orange's functions are members of classes. About the only often used exception to this is ``orange.newmetaid`` which returns a new ID for a meta attribute. These functions are defined in the same way as member function except that the function name doesn't have the class name (and the underscore - that's how pyxtract distinguishes between the two). Here's the ``newmetaid`` function. ::

    PyObject *newmetaid(PyObject *, PyObject *) PYARGS(0,"() -> int")
    {
        PyTRY
            return PyInt_FromLong(getMetaID());
        PyCATCH
    }

Orange also defines some non-member constants. These are defined in a similar fashion as the class constants. ``PYCONSTANT_INT(&lt;constant-name&gt;, &lt;integer&gt;)`` defines an integer constant and ``PYCONSTANT_FLOAT`` would be used for a continuous one. ``PYCONSTANT`` is used for objects of other types, as the below example that defines an (obsolete) constant ``MeasureAttribute_splitGain`` shows. ::

    PYCONSTANT(MeasureAttribute_splitGain, (PyObject *)&PyOrMeasureAttribute_gainRatio_Type)

Class constants from the previous chapter are put in a pyxtract generated file that is included at the end of the file in which the constant definitions and the corresponding classes are. Global constant modules are included in an other file, far away from their actual definitions. For this reason, ``PYCONSTANT`` cannot reference any functions (the above example is an exception - all class types are declared in this same file and are thus available at the moment the above code is used). Therefore, if the constant is defined by a function call, you need to use another keyword, ``PYCONSTANTFUNC``::

    PYCONSTANTFUNC(globalRandom, stdRandomGenerator)

Pyxtract will generate a code which will, prior to calling ``stdRandomGenerator``, declare it as a function with no arguments that returns ``PyObject *``. Of course, you will have to define the function somewhere in your code, like this::

    PyObject *stdRandomGenerator()
    {
        return WrapOrange(globalRandom);
    }

Another example are ``VarTypes``. You've probably already used ``orange.VarTypes.Discrete`` and ``orange.VarTypes.Continuous`` to check an attribute's type. ``VarTypes`` is a tiny module inside Orange that contains nothing but five constants, representing various attribute types. From pyxtract perspective, ``VarTypes`` is a constant. Here's the complete definition. ::

    PyObject *VarTypes()
    {
        PyObject *vartypes=PyModule_New("VarTypes");
        PyModule_AddIntConstant(vartypes, "None", (int)TValue::NONE);
        PyModule_AddIntConstant(vartypes, "Discrete", (int)TValue::INTVAR);
        PyModule_AddIntConstant(vartypes, "Continuous", (int)TValue::FLOATVAR);
        PyModule_AddIntConstant(vartypes, "Other", (int)TValue::FLOATVAR+1);
        PyModule_AddIntConstant(vartypes, "String", (int)STRINGVAR);
        return vartypes;
    }

    PYCONSTANTFUNC(VarTypes, VarTypes)

If you want to understand the constants completely, check the Orange's pyxtract generated file initialization.px.

How does it all fit together
============================

This part of the text's main purpose is to remind the pyxtract author of the structure of the files pyxtract creates. (I'm annoyed when I don't know how my programs work. And I happen to be annoyed quite frequently. :-) If you think you can profit from reading it, you are welcome.

File specific px files
----------------------

For each compiled cpp file, pyxtract creates a px file with the same name. The file starts with externs declaring the base classes for the classes whose types are defined later on.

Then follow class type definitions.

* Method definitions (``PyMethodDef``). Nothing exotic here, just a table with the member functions that is pointed to by ``tp_methods`` of the ``PyTypeObject``.

* GetSet definitions (``PyGetSetDef``). Similar to methods, a list to be pointed to by ``tp_getset``, which includes the attributes for which special handlers were written.

* Definitions of doc strings for call operator and constructor.

* Constants. If the class has any constants, there will be a function named ``void &lt;class-name&gt;_addConstants()``. The function will create a class dictionary in the type's ``tp_dict``, if there is none yet. Then it will store the constants in it. The functions is called at the module initialization, file initialization.px.

* Constructors. If the class uses generic constructors (ie, if it's defined by ``C_UNNAMED``, ``C_NAMED``, ``C_CALL`` or ``C_CALL3``), they will need to call a default object constructor, like the below one for ``FloatVariable``. (This supposes the object is derived from ``TOrange``! We will need to get rid of this we want pyxtract to be more general. Maybe an additional argument in ``DATASTRUCTURE``?) ::

    POrange FloatVariable_default_constructor(PyTypeObject *type)
    {
        return POrange(mlnew TFloatVariable(), type);
    }
  If the class is abstract, pyxtract defines a constructor that will call ``PyOrType_GenericAbstract``. ``PyOrType_GenericAbstract`` checks the type that the caller wishes to construct; if it is a type derived from this type, it permits it, otherwise it complains that the class is abstract.

* Aliases. A list of renamed attributes.

* ``PyTypeObject`` and the numeric, sequence and mapping protocols. ``PyTypeObject`` is named ``PyOr&lt;classname&gt;_Type_inh``.

* Definition of conversion functions. This is done by macro ``DEFINE_cc(&lt;classname&gt;)`` which defines ``int ccn_&lt;classname&gt;(PyObject *obj, void *ptr)`` - functions that can be used in ``PyArg_ParseTuple`` for converting an argument (given as ``PyObject *`` to an instance of ``&lt;classname&gt;``. Nothing needs to be programmed for the conversion, it is just a cast: ``*(GCPtr< T##type > *)(ptr) = PyOrange_As##type(obj);``). The difference between ``cc`` and ``ccn`` is that the latter accepts null pointers.

* ``TOrangeType``. Although ``PyTypeObject`` is a regular Python object, it unfortunately isn't possible to derive new objects from it. Obviously the developers of Python didn't think anyone would need it, and this part of Python's code is messy enough even without it. Orange nevertheless uses a type ``TOrangeType`` that begins with ``PyTypeObject`` (essentially inheriting it). The new definition also includes the RTTI used for wrapping (this way Orange nows which C++ class corresponds to which Python class), a pointer to the default constructor (used by generic constructors), a pointer to list of constructor keywords (``CONSTRUCTOR_KEYWORDS``, keyword arguments that should be ignored in a later call to ``init``) and recognized attributes (``RECOGNIZED_ATTRIBUTES``, attributes that don't yield warnings when set), a list of aliases, and pointers to ``cc_`` and ``ccn_`` functions. The latter are not used by Orange, since it can call the converters directly. They are here because ``TOrangeType`` is exported in a DLL while ``cc_`` and ``ccn_`` are not (for the sake of limiting the number of exported symbols).


initialization.px
-----------------

Initialization.px defines the global module stuff.

First, here is a list of all ``TOrangeTypes``. The list is used for checking whether some Python object is of Orange's type or derived from one, for finding a Python class corresponding to a C++ class (based on C++'s RTTI). Orange also exports the list as ``orange._orangeClasses``; this is a ``PyCObject`` so it can only be used by other Python extensions written in C.

Then come declarations of all non-member functions, followed by a ``PyMethodDef`` structure with them.

Finally, here are declarations of functions that return manually constructed constants (eg ``VarTypes``) and declarations of functions that add class constants (eg ``Classifier_addConstants``). The latter functions were generated by pyxtract and reside in the individual px files. Then follows a function that calls all the constant related functions declared above. This function also adds all class types to the Orange module. Why not in a loop over ``orangeClasses``?

The main module now only needs to call ``addConstants``.

externs.px
----------

Externs.px declares symbols for all Orange classes, for instance ::

    extern ORANGE_API TOrangeType PyOrDomain_Type;
    #define PyOrDomain_Check(op) PyObject_TypeCheck(op, (PyTypeObject *)&PyOrDomain_Type)
    int cc_Domain(PyObject *, void *);
    int ccn_Domain(PyObject *, void *);
    #define PyOrange_AsDomain(op) (GCPtr< TDomain >(PyOrange_AS_Orange(op)))

*****************
Where to include?
*****************

As already mentioned, ppp files should be included (at the beginning) of the corresponding cpp files, instead of the hpp file. For instance, domain.ppp is included in domain.cpp. Each ppp should be compiled only once, all other files needing the definition of ``TDomain`` should still include domain.hpp as usual.

File-specific px files are included in the corresponding cpp files. lib_kernel.px is included at the end of lib_kernel.cpp, from which it was generated. initialization.px should preferably be included in the file that initializes the module (function ``initorange`` needs to call ``addConstants``, which is declared in initialization.px. These px files contain definitions and must be compiled only once. externs.px contains declarations and can be included wherever needed.

Some steps in these instructions are only for Visual Studio 6.0. If you use a newer version of Visual Studio or if you use Linux, adapt them.

Create a new, blank workspace. If your orange sources are in d:\ai\orange\source (as are mine :), specify this directory as a "Location". Add a new project of type "Win 32 Dynamic-Link Library"; change the location back to d:\ai\orange\source. Make it an empty DLL project.

Whatever names you give your module, make sure that the .cpp and .hpp files you create as you go on are in orange\source\something (replace "something" with something), since the further instructions will suppose it.
