/*
    This file is part of Orange.


    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#include "vars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "classify.hpp"
#include "learn.hpp"
#include "getarg.hpp"

#include "c4.5.ppp"

DEFINE_TOrangeVector_classDescription(PC45TreeNode, "TC45TreeNodeList", true, ORANGE_API)

bool c45Loaded = false;

typedef void *learnFunc(int trials, char gainRatio, char subset, char batch, char probThresh,
                       int minObjs, int window, int increment, float cf, char prune);
typedef void garbageFunc();

learnFunc *c45learn;
garbageFunc *c45garbage;
void *pc45data;

extern PyObject *orangeModule;


typedef  union  _attribute_value {
  DiscrValue _discr_val;
  float _cont_val;
} AttValue, *Description;


#define Unknown  -999

#define BrDiscr 1
#define ThreshContin 2
#define BrSubset 3

#define Bit(b) (1 << (b))
#define In(b,s) ((s[(b) >> 3]) & Bit((b) & 07))


struct {
  short		*rMaxAtt, *rMaxClass, *rMaxDiscrVal;
  int *rMaxItem;
  Description	**rItem;
  DiscrValue	**rMaxAttVal;
  char **rSpecialStatus, ***rClassName, ***rAttName, ****rAttValName; 
} c45data;

#ifdef _DEBUG
#define C45STEM "c45_d"
#else
#define C45STEM "c45"
#endif

#ifdef _MSC_VER
#define PATHSEP '\\'
#define C45NAME "\\" C45STEM ".dll"
#else
#define PATHSEP '/'
#define C45NAME "/" C45STEM ".so"
#endif

#if defined _MSC_VER

#include <direct.h>
#define getcwd _getcwd

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
#include <windows.h>

void *getsym(HINSTANCE handle, const char *name)
{
  void *sym = GetProcAddress(handle, name);
  if (!sym)
    raiseErrorWho("C45Loader", "invalid %s, cannot find symbol %s", C45NAME, name);
  return sym;
}

void dynloadC45(const char *pathname)
{
  HINSTANCE c45Dll = LoadLibrary(pathname);
  if (!c45Dll)
    raiseErrorWho("C45Loader", "cannot load %s (%s)", C45NAME, pathname);

  pc45data = getsym(c45Dll, "c45Data");
  c45learn = (learnFunc *)(getsym(c45Dll, "learn"));
  c45garbage = (garbageFunc *)(getsym(c45Dll, "guarded_collect"));
}

#elif defined LINUX || defined FREEBSD || defined DARWIN

#include <dlfcn.h>
#include <unistd.h>

void *getsym(void *handle, const char *name)
{
  void *sym = dlsym(handle, name);
  if (!sym)
    raiseErrorWho("C45Loader", "invalid %s, cannot find symbol %s", C45NAME, name);
  return sym;
}

void dynloadC45(char pathname[])
{ 
  void *handle = dlopen(pathname, RTLD_NOW /*dlopenflags*/);
  if (handle == NULL)
    raiseErrorWho("C45Loader", dlerror());
  
  pc45data = getsym(handle, "c45Data");
  c45learn = (learnFunc *)getsym(handle, "learn");
  c45garbage = (garbageFunc *)getsym(handle, "guarded_collect");
}
   
#else

void dynloadC45(char [])
{ raiseErrorWho("C45Loader", "c45 is not supported on this platform"); }

#endif

#ifdef IGNORE
#undef IGNORE
#endif

void loadC45()
{
  char *buf = NULL, *bp;

  PyObject *orangeDirName = PyDict_GetItemString(PyModule_GetDict(orangeModule), "__file__");
  if (orangeDirName) {
    char *odn = PyString_AsString(orangeDirName);
    buf = (char *)malloc(strlen(odn) + strlen(C45NAME) + 1);
    strcpy(buf, odn);
    bp = buf + strlen(buf);
    while ((bp!=buf) && (*bp!=PATHSEP))
      bp--;
    *bp = 0;
  }
    
  // If path is empty, orange.so was probably loaded from the working directory
  if (!buf || !*buf) {
    buf = (char *)realloc(buf, 512);
    if (!getcwd(buf, 511))
      raiseErrorWho("C45Loader", C45NAME " cannot be found");
    bp = buf + strlen(buf);
  }
  
  strcpy(bp, C45NAME);

  dynloadC45(buf);
  memcpy(&c45data, pc45data, sizeof(c45data));
  c45Loaded = true;
}

#define MaxAtt (*c45data.rMaxAtt)
#define MaxClass (*c45data.rMaxClass)
#define MaxDiscrVal (*c45data.rMaxDiscrVal)
#define MaxItem (*c45data.rMaxItem)
#define Item (*c45data.rItem)
#define MaxAttVal (*c45data.rMaxAttVal)
#define SpecialStatus (*c45data.rSpecialStatus)
#define ClassName (*c45data.rClassName)
#define AttName (*c45data.rAttName)
#define AttValName (*c45data.rAttValName)

TC45Learner::TC45Learner()
 : gainRatio(true),
   subset(false),
   batch(true),
   probThresh(false),
   minObjs(2),
   window(0),
   increment(0),
   cf(0.25),
   trials(10),
   prune(true),
   convertToOrange(false),
   storeContingencies(false),
   storeExamples(false)
{
    if (!c45Loaded)
      loadC45();
 }


bool TC45Learner::clearDomain()
{ if (ClassName) {
    String *ClassNamei=ClassName;
    MaxClass++;
    while(MaxClass--)
      mldelete *(ClassNamei++);
    mldelete ClassName;
    ClassName=NULL;
  }

  if (AttName) {
    String *AttNamei=AttName;
    int atts=MaxAtt+1;
    while(atts--)
      mldelete *(AttNamei++);
    mldelete AttName;
    AttName=NULL;
  }

  if (AttValName && MaxAttVal) {
    String **AttValNamei=AttValName; 
    DiscrValue *MaxAttVali=MaxAttVal; 
    for(int atts=MaxAtt; atts--; MaxAttVali++) {
      String *AttValNameii = *AttValNamei+1; // the first one is NULL...
      while((*MaxAttVali)--)
        mldelete *(AttValNameii++);
      mldelete *(AttValNamei++);
    }
    mldelete AttValName;
    mldelete MaxAttVal;
    AttValNamei=NULL;
    MaxAttVal=NULL;
  }

  if (SpecialStatus) {
    mldelete SpecialStatus;
    SpecialStatus=NULL;
  }

  return true;
}      


bool TC45Learner::convertDomain(PDomain dom)
{ 
  TEnumVariable *classVar=dom->classVar.AS(TEnumVariable);
  if (!classVar)
    raiseError("domain with discrete class attribute expected");

  MaxAtt = dom->attributes->size()-1;
  MaxClass = classVar->noOfValues()-1;
  MaxDiscrVal=2; // increased below

  ClassName = mlnew String[MaxClass+1];
  String *ClassNamei=ClassName;
  PITERATE(TStringList, ni, classVar->values) {
    *ClassNamei = mlnew char[(*ni).length()+1];
    strcpy(*(ClassNamei++), (*ni).c_str());
  }
    
  AttName = mlnew String[MaxAtt+1];
  String *AttNamei = AttName;

  AttValName = mlnew String *[MaxAtt+1];
  String **AttValNamei = AttValName;

  MaxAttVal = mlnew DiscrValue[MaxAtt+1];
  DiscrValue *MaxAttVali = MaxAttVal;

  SpecialStatus = mlnew char [MaxAtt+1];
  char *SpecialStatusi = SpecialStatus;

  PITERATE(TVarList, vi, dom->attributes) {
    *(SpecialStatusi++) = NULL;

    *AttNamei = mlnew char[(*vi)->name.length()+1];
    strcpy(*(AttNamei++), (*vi)->name.c_str());

    if ((*vi)->varType==TValue::INTVAR) {
      int noOfValues = (*vi).AS(TEnumVariable)->noOfValues();
      if (noOfValues>MaxDiscrVal)  
        MaxDiscrVal=noOfValues;
      *(MaxAttVali++) = noOfValues;

      *AttValNamei = mlnew String[noOfValues+1];
      String *AttValNameii = *(AttValNamei++);
      *(AttValNameii++)=NULL;
      PITERATE(TStringList, ni, (*vi).AS(TEnumVariable)->values) {
        *AttValNameii = mlnew char[(*ni).length()+1];
        strcpy(*(AttValNameii++), (*ni).c_str());
      }
    }
    else {
      *(AttValNamei++) = NULL;
      *(MaxAttVali++) = 0;
    }
  }

  return true;
}


Description convertExample(const TExample &example)
{
  Description item = mlnew AttValue[MaxAtt+2];
  Description itemi = item;
  const_ITERATE(TExample, eii, example)
    if ((*eii).varType == TValue::INTVAR)
      (itemi++)->_discr_val = (*eii).isSpecial() ? 0 : int(*eii)+1;
    else if ((*eii).varType == TValue::FLOATVAR)
      (itemi++)->_cont_val = (*eii).isSpecial() ? Unknown : float(*eii);
    else {
      mldelete item;
      item = NULL;
      raiseError("invalid attribute type");
    }
  // Decrease class!
  itemi[-1]._discr_val--;
  return item;
}


bool TC45Learner::convertExamples(PExampleGenerator table)
{ Item = mlnew Description[table->numberOfExamples()];
  Description *Itemi = Item;
  MaxItem = 0;
  PEITERATE(ei, table)
    if (!(*ei).getClass().isSpecial()) {
      *(Itemi++) = convertExample(*ei);
      MaxItem ++;
    }

  if (!MaxItem) {
    mldelete Item;
    raiseError("empty data set or no examples with defined class");
  }
  
  MaxItem--;
  return true;
}


bool TC45Learner::clearExamples()
{ if (Item) {
    Description *Itemi = Item;
    MaxItem++;
    while(MaxItem--)
      mldelete *(Itemi++);
    mldelete Item;
    Item=NULL;
  }
  return true;
}


bool TC45Learner::convertGenerator(PExampleGenerator gen)
{ 
  return convertDomain(gen->domain) && convertExamples(gen);
}


bool TC45Learner::clearGenerator()
{ 
  return clearExamples() && clearDomain();
}


bool TC45Learner::parseCommandLine(const string &line)
{
  TProgArguments args("f: b u p v: t: w: i: g s m: c:", line);
  if (args.direct.size())
    raiseError("parseCommandLine: invalid parameter %s", args.direct.front().c_str());

  ITERATE(TMultiStringParameters, oi, args.options)
    switch ((*oi).first[0]) {
      case 'f':
      case 'u':
      case 'v':
      raiseError("parseCommandLine: option -%s not accepted", (*oi).first.c_str());

      case 'b':
        batch = true;
		    break;

      case 'p':   
        probThresh = true;
        break;

	    case 't':  
        trials = atoi((*oi).second.c_str());
		    batch = false;
        if ((trials<1) || (trials>10000)) {
          trials=10;
          raiseError("parseCommandLine: invalid argument for -t");
        }
        break;

      case 'w':  
        window = atoi((*oi).second.c_str());
        batch = false;
        if ((window<1) || (window>1000000)) {
          window = 0;
          raiseError("parseCommandLine: invalid argument for -w");
        }
        break;

	    case 'i':
        increment = atoi((*oi).second.c_str());
		    batch = false;
        if ((increment<1) || (increment>1000000)) {
          increment = 0;
          raiseError("parseCommandLine: invalid argument for -i");
        }
        break;

	    case 'g':  
        gainRatio = false;
        break;

	    case 's':  
        subset = true;
        break;

      case 'm':   
        minObjs = atoi((*oi).second.c_str());
        if ((minObjs<1) || (minObjs>1000000)) {
          minObjs = 2;
          raiseError("parseCommandLine: invalid argument for -m");
        }
        break;

      case 'c':   
        cf = atof((*oi).second.c_str());
        if ((cf<=0) || (cf>100)) {
          cf = 0.25;
          raiseError("parseCommandLine: invalid argument for -c");
        }
        break;
    }

  return true;
}


bool TC45Learner::convertParameters()
{ return true;
}




PClassifier TC45Learner::operator ()(PExampleGenerator gen, const int &weight)
{   if (!gen->domain->classVar)
      raiseError("class-less domain");
    if (!gen->numberOfExamples())
      raiseError("no examples");
    if (!gen->domain->attributes->size())
      raiseError("no attributes");
 
    convertGenerator(gen);
    Tree tree = (Tree)c45learn(trials, gainRatio, subset, batch, probThresh, minObjs, window, increment, cf, prune);

    PC45TreeNode root = mlnew TC45TreeNode(tree, gen->domain);
    TC45Classifier *c45classifier = mlnew TC45Classifier(gen->domain, root);
    PClassifier res = c45classifier;

    c45garbage();
    clearGenerator();
    return convertToOrange ? PClassifier(c45classifier->asTreeClassifier(gen, weight, storeContingencies, storeExamples)) : res;
}



TC45TreeNode::TC45TreeNode()
: nodeType(4),
  leaf(TValue::INTVAR),
  items(-1),
  classDist(),
  tested(),
  cut(0),
  lower(0),
  upper(0),
  mapping(),
  branch()
{}


TC45TreeNode::TC45TreeNode(const Tree &node, PDomain domain)
: nodeType(node->NodeType),
  leaf(TValue(node->Leaf)),
  items(node->Items),
  classDist(mlnew TDiscDistribution(domain->classVar)),
  tested(nodeType != Leaf ? domain->attributes->operator[](node->Tested) : PVariable()),
  cut(node->Cut),
  lower(node->Lower),
  upper(node->Upper),
  mapping(),
  branch()
{ 
  float *cd = node->ClassDist; // no +1
  int i, e;
  for(i = 0, e = domain->classVar.AS(TEnumVariable)->values->size(); i!=e; i++, cd++)
    classDist->setint(i, float(*cd));

  if (nodeType != Leaf) {
    branch = mlnew TC45TreeNodeList;
    Tree *bi = node->Branch+1;
    for(i = node->Forks; i--; bi++)
      branch->push_back(mlnew TC45TreeNode(*bi, domain));
  }

  if (nodeType == Subset) {
    int ve = tested.AS(TEnumVariable)->values->size();
    mapping = mlnew TIntList(ve, -1);
    char **si = node->Subset+1;
    for(i = 0, e = node->Forks; i!=e; si++, i++)
      for(int vi = 0; vi<ve; vi++)
        if (In(vi+1, *si))
          mapping->operator [](vi) = i;
  }
}



PDiscDistribution TC45TreeNode::vote(const TExample &example, PVariable classVar)
{
  PDiscDistribution res = mlnew TDiscDistribution(classVar);
  PITERATE(TC45TreeNodeList, bi, branch) {
    PDiscDistribution vote = (*bi)->classDistribution(example, classVar);
    vote->operator *= ((*bi)->items);
    res->operator += (vote);
  }
  res->operator *= (1.0/items);
  return res;       
}


#undef min

PDiscDistribution TC45TreeNode::classDistribution(const TExample &example, PVariable classVar)
{
  if (nodeType == Leaf) {
    if (items > 0) {
      PDiscDistribution res = CLONE(TDiscDistribution, classDist);
      res->operator *= (1.0/items);
      return res;
    }
    else {
      PDiscDistribution res = mlnew TDiscDistribution(classVar);
      res->operator[](leaf.intV) = 1.0;
      return res;
    }
  }

  int varnum = example.domain->getVarNum(tested, false);
  const TValue &val = (varnum != ILLEGAL_INT) ? example[varnum] : tested->computeValue(example);
  if (val.isSpecial())
    return vote(example, classVar);

  switch (nodeType) {
//    case Leaf: - taken care of above

    case Branch:
      if (val.intV >= branch->size())
        return vote(example, classVar);
      else
        return branch->operator[](val.intV)->classDistribution(example, classVar);

    case Cut:
      return branch->operator[](val.floatV <= cut ? 0 : 1)->classDistribution(example, classVar);

    case Subset:
      if ((val.intV > mapping->size()) || (mapping->operator[](val.intV) < 0))
        return vote(example, classVar);
      else
        return branch->operator[](mapping->operator[](val.intV))->classDistribution(example, classVar);

    default:
      raiseError("invalid 'nodeType'");
  }

  return PDiscDistribution();
}


TC45Classifier::TC45Classifier(PDomain domain, PC45TreeNode atree)
: TClassifierFD(domain),
  tree(atree)
{}



/* We need to define this separately to ensure that the first class
   is selected in case of a tie */
TValue TC45Classifier::operator ()(const TExample &oexample)
{
  checkProperty(tree);
  
  PDiscDistribution classDist;
  if (oexample.domain != domain) {
    TExample example(domain, oexample);
    classDist = tree->classDistribution(example, classVar);
  }
  else
    classDist = tree->classDistribution(oexample, classVar);

  int bestClass = 0;
  float bestP = -1;
  TDiscDistribution::const_iterator pi(classDist->begin());
  for(int cl = 0, ce = classVar.AS(TEnumVariable)->values->size(); cl!=ce; cl++, pi++) {
    if (*pi > bestP) {
      bestP = *pi;
      bestClass = cl;
    }
  }

  return TValue(bestClass);
}



PDistribution TC45Classifier::classDistribution(const TExample &oexample)
{ 
  checkProperty(tree);

  PDiscDistribution classDist;
  if (oexample.domain != domain) {
    TExample example(domain, oexample);
    classDist = tree->classDistribution(example, classVar);
  }
  else
    classDist = tree->classDistribution(oexample, classVar);

  classDist->normalize();
  return classDist;
}


void TC45Classifier::predictionAndDistribution(const TExample &oexample, TValue &value, PDistribution &dist)
{
  checkProperty(tree);

  if (oexample.domain != domain) {
    TExample example(domain, oexample);
    dist = tree->classDistribution(example, classVar);
  }
  else
    dist = tree->classDistribution(oexample, classVar);

  int bestClass = 0;
  float bestP = -1;
  for(int cl = 0, ce = classVar.AS(TEnumVariable)->values->size(); cl!=ce; cl++) {
    float td = dist->atint(cl);
    if (td > bestP) {
      bestP = td;
      bestClass = cl;
    }
  }

  value = TValue(bestClass);
  dist->normalize();
}




#include "tdidt.hpp"
#include "classfromvar.hpp"
#include "discretize.hpp"
#include "tdidt_split.hpp"

PTreeNode TC45TreeNode::asTreeNode(PExampleGenerator examples, const int &weightID, bool storeContingencies, bool storeExamples)
{ 
  PTreeNode newNode = mlnew TTreeNode();
  newNode->distribution = classDist;
  newNode->distribution->normalize();

  if (items > 0)
    newNode->nodeClassifier = mlnew TDefaultClassifier(examples->domain->classVar, leaf, classDist);
  else {
    TDiscDistribution *dd = mlnew TDiscDistribution(examples->domain->classVar);
    dd->add(leaf, 1.0);
    newNode->nodeClassifier = mlnew TDefaultClassifier(examples->domain->classVar, leaf, dd);
  }

  if (storeExamples) {
    newNode->examples = examples;
    newNode->weightID = weightID;
  }

  if (nodeType == Leaf)
    return newNode;

  PDistribution branchSizes = mlnew TDiscDistribution;
  int i = 0;
  PITERATE(TC45TreeNodeList, li, branch)
    branchSizes->addint(i++, (*li)->items);
  newNode->branchSizes = branchSizes;
    
  TEnumVariable *dummyVar = mlnew TEnumVariable(tested->name);
  PVariable wdummyVar = dummyVar;

  switch (nodeType) {
    case Branch:
      newNode->branchSelector = mlnew TClassifierFromVar(tested, branchSizes);
      newNode->branchDescriptions = mlnew TStringList(tested.AS(TEnumVariable)->values.getReference());
      break;

    case Cut:
      newNode->branchDescriptions = mlnew TStringList;

      char str[128];
      sprintf(str, "<=%3.3f", cut);
      newNode->branchDescriptions->push_back(str);
      dummyVar->values->push_back(str);
      sprintf(str, ">%3.3f", cut);
      newNode->branchDescriptions->push_back(str);
      dummyVar->values->push_back(str);

      newNode->branchSelector = mlnew TClassifierFromVar(wdummyVar, tested, branchSizes, mlnew TThresholdDiscretizer(cut));
      break;

    case Subset:
      int noval = 1 + *max_element(mapping->begin(), mapping->end());
      dummyVar->values = mlnew TStringList(noval, "");
      TStringList::const_iterator tvi(tested.AS(TEnumVariable)->values->begin());
      PITERATE(TIntList, ni, mapping) {
        if (*ni >= 0) {
          string &val = dummyVar->values->at(*ni);
          if (val.length())
            val += ", ";
          val += *tvi;
        }
        tvi++;
      }
      PITERATE(TStringList, vi, dummyVar->values)
        if ((*vi).find(",") != string::npos)
          *vi = "in [" + *vi + "]";

      newNode->branchSelector = mlnew TClassifierFromVar(wdummyVar, tested, branchSizes,mlnew TMapIntValue(mapping));
      newNode->branchDescriptions = dummyVar->values;
      break;
  }

  vector<int> newWeights;
  PExampleGeneratorList subsets;
  TExampleGeneratorList::const_iterator si;
  if (storeExamples || storeContingencies) {
    subsets = TTreeExampleSplitter_UnknownsAsBranchSizes()(PTreeNode(newNode.getReference()), examples, weightID, newWeights);
    si = subsets->begin();
  }

  newNode->branches = mlnew TTreeNodeList;
  vector<int>::const_iterator wi(newWeights.begin()), we(newWeights.end());
  PITERATE(TC45TreeNodeList, c45bi, branch) {
    if (storeExamples || storeContingencies) {
      newNode->branches->push_back(*c45bi ? (*c45bi)->asTreeNode(*(si++), wi!=we ? *wi : weightID, storeContingencies, storeExamples) : PTreeNode());
      if (wi!=we) {
        examples->removeMetaAttribute(*wi);
        wi++;
      }
    }
    else
      // just call with 'examples' as argument -- they will only be used to extract examples->domain->classVar
      newNode->branches->push_back(*c45bi ? (*c45bi)->asTreeNode(examples, weightID, false, false) : PTreeNode());
  }

  

  return newNode;
}


PTreeClassifier TC45Classifier::asTreeClassifier(PExampleGenerator examples, const int &weightID, bool storeContingencies, bool storeExamples)
{
  if (storeContingencies)
    raiseWarning("'storeContingencies' not supported yet");

  PExampleTable exampleTable = toExampleTable(examples);

  PTreeNode orangeTree = tree->asTreeNode(examples, weightID, storeContingencies, storeExamples);
  return mlnew TTreeClassifier(examples->domain, orangeTree, mlnew TTreeDescender_UnknownMergeAsBranchSizes());
}
