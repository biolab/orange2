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

#include "c4.5.ppp"

DEFINE_TOrangeVector_classDescription(PC45TreeNode, "TC45TreeNodeList")

bool c45Loaded = false;

typedef void *learnFunc(char gainRatio, char subset, char batch, char probThresh,
                       int trials, int minObjs, int window, int increment, float cf, char prune);
typedef void garbageFunc();

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

learnFunc *c45learn;
garbageFunc *c45garbage;
void *pc45data;

extern PyObject *orangeModule;


#ifdef IGNORE
#undef IGNORE
#endif

const char *dynloadC45(char buf[], char *bp);

void loadC45()
{
  char buf[512], *bp;

  PyObject *orangeDirName = PyDict_GetItemString(PyModule_GetDict(orangeModule), "__file__");
  if (orangeDirName) {
    char *odn = PyString_AsString(orangeDirName);
    if (strlen(odn) <= 500) {
      strcpy(buf, odn);
      bp = buf + strlen(buf);
      while ((bp!=buf) && (*bp!='\\') && (*bp !='/'))
        bp--;
    } 
    else
      raiseErrorWho("C45Loader", "cannot load c45.dll (pathname too long)");
  }

  const char *err = dynloadC45(buf, bp);
  if (err)
    raiseErrorWho("C45Loader", err);

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
   prune(true)
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
  PITERATE(TIdList, ni, classVar->values) {
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
      PITERATE(TIdList, ni, (*vi).AS(TEnumVariable)->values) {
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




//#define C45DEBUG(x) x
#define C45DEBUG(x)

PClassifier TC45Learner::operator ()(PExampleGenerator gen, const int &weight)
{   if (!gen->domain->classVar)
      raiseError("class-less domain");
 
    convertGenerator(gen);
    Tree tree = (Tree)c45learn(trials, gainRatio, subset, batch, probThresh, minObjs, window, increment, cf, prune);

    PC45TreeNode root = mlnew TC45TreeNode(tree, gen->domain);
    PClassifier c45classifier = mlnew TC45Classifier(gen->domain->classVar, root);

    c45garbage();
    clearGenerator();
    return c45classifier;
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


TC45Classifier::TC45Classifier(PVariable classVar, PC45TreeNode atree)
: TClassifier(classVar),
  tree(atree)
{}



/* We need to define this separately to ensure that the first class
   is selected in case of a tie */
TValue TC45Classifier::operator ()(const TExample &example)
{
  checkProperty(tree);

  PDiscDistribution classDist = tree->classDistribution(example, classVar);
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



PDistribution TC45Classifier::classDistribution(const TExample &example)
{ 
  checkProperty(tree);

  PDiscDistribution classDist = tree->classDistribution(example, classVar);
  classDist->normalize();
  return classDist;
}


void TC45Classifier::predictionAndDistribution(const TExample &example, TValue &value, PDistribution &dist)
{
  checkProperty(tree);

  dist = tree->classDistribution(example, classVar);
  int bestClass = 0;
  float bestP = -1;
  for(int cl = 0, ce = classVar.AS(TEnumVariable)->values->size()-1; cl!=ce; cl++) {
    float td = dist->atint(cl);
    if (td > bestP) {
      bestP = td;
      bestClass = cl;
    }
  }

  value = TValue(bestClass);
  dist->normalize();
}
