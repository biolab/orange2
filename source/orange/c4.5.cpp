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

#include "../external/c45/defns.i"

DEFINE_TOrangeVector_classDescription(PC45TreeNode, "TC45TreeNodeList")

#ifdef _MSC_VER

#ifdef IGNORE
#undef IGNORE
#endif

//#include "../../external/c45/c45.h"
#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

#include <windows.h>

bool c45Loaded = false;

double **rClassSum;
short		*rMaxAtt, *rMaxClass, *rMaxDiscrVal;
ItemNo *rMaxItem;
Description	**rItem;
DiscrValue	**rMaxAttVal;
char **rSpecialStatus;
String **rClassName, **rAttName, ***rAttValName, *rFileName;
short *rVERBOSITY, *rTRIALS;
Boolean	*rGAINRATIO, *rSUBSET, *rBATCH, *rUNSEENS, *rPROBTHRESH;
ItemNo *rMINOBJS, *rWINDOW, *rINCREMENT;
double *rCF;
Tree **rPruned, **rRaw;
Boolean *rAllKnown;
ItemNo **rTargetClassFreq;

int (*rSoftenThresh)(Tree);
Tree (__cdecl *rFormTree)(ItemNo, ItemNo);

Tree (__cdecl *rCopyTree)(Tree);
Tree (*rIterate)(ItemNo, ItemNo);
Boolean (__cdecl *rPrune)(Tree);
int (__cdecl *rFormTarget)(ItemNo), (*rFormInitialWindow)();

int (__cdecl *rInitialiseTreeData)();
int (__cdecl *rInitialiseWeights)();

void (*rReleaseTree)(Tree Node, bool clearSubsets = true);


#define ClassSum (*rClassSum)
#define MaxAtt (*rMaxAtt)
#define MaxClass (*rMaxClass)
#define MaxDiscrVal (*rMaxDiscrVal)
#define MaxItem (*rMaxItem)
#define Item (*rItem)
#define MaxAttVal (*rMaxAttVal)
#define SpecialStatus (*rSpecialStatus)
#define ClassName (*rClassName)
#define AttName (*rAttName)
#define MaxAtt (*rMaxAtt)
#define AttValName (*rAttValName)
#define FileName (*rFileName)
#define VERBOSITY (*rVERBOSITY)
#define TRIALS (*rTRIALS)
#define GAINRATIO (*rGAINRATIO)
#define SUBSET (*rSUBSET)
#define BATCH (*rBATCH)
#define UNSEENS (*rUNSEENS)
#define PROBTHRESH (*rPROBTHRESH)
#define MINOBJS (*rMINOBJS)
#define WINDOW (*rWINDOW)
#define INCREMENT (*rINCREMENT)
#define CF (*rCF)
#define Pruned (*rPruned)
#define Raw (*rRaw)
#define AllKnown (*rAllKnown)
#define TargetClassFreq (*rTargetClassFreq)

#define SoftenThresh (*rSoftenThresh)
#define FormTree (*rFormTree)

#define CopyTree (*rCopyTree)
#define ReleaseTree (*rReleaseTree)
#define Iterate (*rIterate)
#define Prune (*rPrune)
#define FormTarget (*rFormTarget)
#define FormInitialWindow (*rFormInitialWindow)
#define InitialiseTreeData (*rInitialiseTreeData)
#define InitialiseWeights (*rInitialiseWeights)

typedef void **getDataFunc();

extern PyObject *orangeModule;

void loadC45()
{
  char buf[512], *bp = buf;
  
  PyObject *orangeDirName = PyDict_GetItemString(PyModule_GetDict(orangeModule), "__file__");
  if (orangeDirName) {
    char *odn = PyString_AsString(orangeDirName);
    if (strlen(odn) <= 500) {
      strcpy(buf, odn);
      bp = buf + strlen(buf);
      while ((bp!=buf) && (*bp!='\\'))
        bp--;
    } 
    else
      raiseErrorWho("C45Loader", "cannot load c45.dll (pathname too long)");
  }

  #ifdef _DEBUG
  strcpy(bp, "\\c45_d.dll");
  #else
  strcpy(bp, "\\c45.dll");
  #endif

  HINSTANCE c45Dll = LoadLibrary(buf);
  if (!c45Dll)
    raiseErrorWho("C45Loader", "cannot load c45.dll");

  char funcname[258];
  PyOS_snprintf(funcname, sizeof(funcname), "%s", "getc45Data");
  getDataFunc *p = (getDataFunc *)(GetProcAddress(c45Dll, funcname));

  if (!p)
    raiseErrorWho("C45Loader", "c45.dll is invalid");

  void **data = (*p)();
  if (!data)
    raiseErrorWho("C45Loader", "c45.dll cannot initialize");

  #define COPY(x,i) (void *&)(r##x) = data[i];
  COPY(ClassSum, 0);           COPY(MaxAtt, 1);              COPY(MaxClass, 2);          COPY(MaxDiscrVal, 3);      COPY(MaxItem, 4);
  COPY(Item, 5);               COPY(MaxAttVal, 6);           COPY(SpecialStatus, 7);     COPY(ClassName, 8);        COPY(AttName, 9);
  COPY(AttValName, 10);        COPY(FileName, 11);

  COPY(VERBOSITY, 12);         COPY(TRIALS, 13);             COPY(GAINRATIO, 14);        COPY(SUBSET, 15);          COPY(BATCH, 16);
  COPY(UNSEENS, 17);           COPY(PROBTHRESH, 18);         COPY(MINOBJS, 19);          COPY(WINDOW, 20);          COPY(INCREMENT, 21);
  COPY(CF, 22);

  COPY(Pruned, 23);            COPY(Raw, 24);                COPY(AllKnown, 25);         COPY(TargetClassFreq, 26);
  
  COPY(SoftenThresh, 27);      COPY(ReleaseTree, 28);        /* was: Category, 29 */     /* was: Classify, 30 */    COPY(FormTree, 31);
  COPY(CopyTree, 32);          COPY(Iterate, 33);            COPY(Prune, 34);            /* was: PrintTree, 35 */   COPY(FormTarget, 36);
  COPY(FormInitialWindow, 37); COPY(InitialiseTreeData, 38); COPY(InitialiseWeights, 39);
  #undef COPY
  
  c45Loaded = true;
}

#else

extern "C" {
  short BestTree();
  int OneTree(), SoftenThresh(Tree);

  Tree FormTree(ItemNo, ItemNo);
  Tree CopyTree(Tree), Iterate(ItemNo, ItemNo);
  Boolean Prune(Tree);
  int FormTarget(ItemNo), FormInitialWindow();
  void ReleaseTree(Tree Node, bool clearSubsets = true);

  int InitialiseTreeData();
  int InitialiseWeights();
}

extern double *ClassSum;		/* ClassSum[c] = total weight of class c */

extern  short MaxAtt = 0 , MaxClass = 0, MaxDiscrVal = 2;
extern  ItemNo		MaxItem = 0;
extern  Description	*Item = NULL;
extern  DiscrValue	*MaxAttVal = NULL;
extern  char *SpecialStatus = NULL;
extern  String *ClassName = NULL, *AttName = NULL, **AttValName = NULL, FileName = "DF";
extern  short VERBOSITY = 0, TRIALS = 10;
extern  Boolean		GAINRATIO  = true, SUBSET = false, BATCH = true, UNSEENS = false, PROBTHRESH = false;
extern  ItemNo		MINOBJS   = 2, WINDOW = 0, INCREMENT = 0;
extern  double		CF = 0.25;
extern  Tree		*Pruned = NULL, *Raw = NULL;
extern  Boolean		AllKnown = true;
extern  ItemNo *TargetClassFreq;

extern double atof();

#endif // else of ifdef _MSC_VER

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
  #ifdef _MSC_VER
    if (!c45Loaded)
      loadC45();
  #endif
}


bool TC45Learner::clearDomain()
{ if (ClassName) {
    String *ClassNamei=ClassName;
    while(MaxClass--)
      mldelete *(ClassNamei++);
    mldelete ClassName;
    ClassName=NULL;
  }

  if (AttName) {
    String *AttNamei=AttName;
    int atts=MaxAtt;
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
    *(SpecialStatusi++) = Nil;

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
  PEITERATE(ei, table) {
    *(Itemi++) = convertExample(*ei);
    MaxItem ++;
  }

  MaxItem--;
  return true;
}


bool TC45Learner::clearExamples()
{ if (Item) {
    Description *Itemi = Item;
    while(MaxItem--)
      mldelete *(Itemi++);
    mldelete Item;
    Item=NULL;
  }
  return true;
}


bool TC45Learner::convertGenerator(PExampleGenerator gen)
{ PExampleGenerator table = toExampleTable(gen); // ensure that we know about all the attribute values
  convertDomain(gen->domain);
  convertExamples(gen);
  return true;
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
{ VERBOSITY  = 0;
  UNSEENS    = 0;

  TRIALS     = trials;

  GAINRATIO  = gainRatio;
  SUBSET     = subset;
  BATCH      = batch;

  PROBTHRESH = probThresh;

  MINOBJS    = minObjs;
  WINDOW     = window;
  INCREMENT  = increment;
  CF         = cf;
  return true;
}




//#define C45DEBUG(x) x
#define C45DEBUG(x)

void O_OneTree()
/*  ---------  */
{
    InitialiseTreeData();
    InitialiseWeights();

    Raw = (Tree *) calloc(1, sizeof(Tree));
    Pruned = (Tree *) calloc(1, sizeof(Tree));

    AllKnown = true;
    Raw[0] = FormTree(0, MaxItem);

    Pruned[0] = CopyTree(Raw[0]);
    Prune(Pruned[0]);
}



short O_BestTree()
{
    short t, Best=0;

    InitialiseTreeData();

    TargetClassFreq = (ItemNo *) calloc(MaxClass+1, sizeof(ItemNo));

    Raw    = (Tree *) calloc(TRIALS, sizeof(Tree));
    Pruned = (Tree *) calloc(TRIALS, sizeof(Tree));

    if ( ! WINDOW )
      WINDOW = ItemNo(Max(2 * sqrt(MaxItem+1.0), (MaxItem+1) / 5));

    if ( ! INCREMENT )
      INCREMENT = Max(WINDOW / 5, 1);

    FormTarget(WINDOW);

    ForEach(t, 0, TRIALS-1 ) {
        FormInitialWindow();
        Raw[t] = Iterate(WINDOW, INCREMENT);

        Pruned[t] = CopyTree(Raw[t]);
        Prune(Pruned[t]);

        if ( Pruned[t]->Errors < Pruned[Best]->Errors )
          Best = t;
    }

    return Best;
}


PClassifier TC45Learner::operator ()(PExampleGenerator gen, const int &weight)
{   if (!gen->domain->classVar)
      raiseError("class-less domain");
 
    convertGenerator(gen);
    convertParameters();

    short Best;
 
    if ( BATCH ) {
      TRIALS = 1;
      O_OneTree();
      Best = 0;
    }
    else
      Best = O_BestTree();

    if ( PROBTHRESH )
      SoftenThresh((prune ? Pruned : Raw)[Best]);

    PC45TreeNode root = mlnew TC45TreeNode((prune ? Pruned : Raw)[Best], gen->domain);
    PClassifier c45classifier = mlnew TC45Classifier(gen->domain->classVar, root);

    Tree *Prunedi = Pruned;
    Tree *Rawi = Raw;
    for(int tr=0; tr!=TRIALS; tr++, Prunedi++, Rawi++) {
      ReleaseTree(*Rawi);
      ReleaseTree(*Prunedi, false);
    }

    mldelete Raw;
    Raw = NULL;
    mldelete Pruned;
    Pruned = NULL;

    clearDomain();
    clearExamples();
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
  tested(domain->attributes->operator[](node->Tested)),
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
