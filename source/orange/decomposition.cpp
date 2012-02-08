/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "table.hpp"

// the following are here just for IMByRowsByRelief -- move it to a separate file?
#include "distance.hpp"
#include "random.hpp"
#include <set>

#include "decomposition.ppp"


TExample_nodeIndex::TExample_nodeIndex(PExample ex)
  : example(ex), nodeIndex(0)
  {}



TSortedExamples_nodeIndices::TSortedExamples_nodeIndices(PExampleGenerator eg, const vector<bool> &bound, const vector<bool> &free)
// We need to make a copy of examples since we shall store pointers to them.
: exampleTable(mlnew TExampleTable(eg)),
  maxIndex(-1)
{ 
  // Copy pointers to examples to a vector
  vector<TExample_nodeIndex> eni;
  { eni.reserve(exampleTable->numberOfExamples());
    PEITERATE(ei, exampleTable)
      if (!(*ei).getClass().isSpecial())
      // We wrap references to examples; exampleTable will live at least as long as those references need to
        eni.push_back(TExample_nodeIndex(PExample(*ei)));
  }

  // copy pointers to a table for sorting
  vector<TEnIIterator> *sorting = mlnew vector<TEnIIterator>();
  try {
    { sorting->reserve(eni.size());
      ITERATE(vector<TExample_nodeIndex>, enii, eni)
        sorting->push_back(enii);
    }

    /* Sorts the vector according to values of bound attributes,
       and assigns indices by multiplications.
       We use a combination of bucket and counting sort with linear time complexity. */
    int mult=1;
    { TVarList::iterator attr = eg->domain->attributes->begin();
      const_ITERATE(vector<bool>, bi, bound) {
        if (*bi) {
          if ((*attr)->varType!=TValue::INTVAR)
            raiseError("bound attribute '%s' is not discrete", (*attr)->get_name().c_str());

          int values=(*attr)->noOfValues();
          if (values<=0)
            raiseError("attribute '%s' has invalid number of values", (*attr)->get_name().c_str());

          sortByAttr_Mult(bi-bound.begin(), sorting, values);
          mult *= values;
        }
        attr++;
      }
    }

    // sort by class and free attributes
    { sortByAttr(eg->domain->attributes->size(), sorting, eg->domain->classVar->noOfValues());
      TVarList::iterator attr = eg->domain->attributes->begin();
      const_ITERATE(vector<bool>, fi, free) {
        if (*fi) {
          if ((*attr)->varType != TValue::INTVAR)
            raiseError("free attribute '%s' is not discrete", (*attr)->get_name().c_str());
          sortByAttr(fi-free.begin(), sorting, (*attr)->noOfValues());
        }
        attr++;
      }
    }

    // initialize the vector
    { reserve(sorting->size());
      ITERATE(vector<TEnIIterator>, enii, *sorting)
        push_back(**enii); }

    // compress the index table (remove the unoccupied numbers)
    { vector<int> indices(mult, 0);
      { this_ITERATE(ei)
          indices[(*ei).nodeIndex]++; }
      ITERATE(vector<int>, bi, indices)
        if (*bi>0)
          *bi = ++maxIndex;
      { this_ITERATE(ei)
          (*ei).nodeIndex=indices[(*ei).nodeIndex]; }
    }
  }
  catch (exception) {
    mldelete sorting;
    throw;
  }

  mldelete sorting;
}

        
void TSortedExamples_nodeIndices::sortByAttr_Mult(int attrNo, vector<TEnIIterator> *&sorting, int values)
{ vector<int> valf(values, 0);
  ITERATE(vector<TEnIIterator>, ii, *sorting) {
    TValue &val = (*ii)->example->operator[](attrNo);
    if (val.isSpecial())
      raiseError("attribute '%s' has undefined values", (*ii)->example->domain->getVar(attrNo)->get_name().c_str());
    valf[val.intV]++;
  }

  int id = 0;
  for(vector<int>::iterator ni=valf.begin(); ni!=valf.end(); ++ni)
	  *ni = (id+=*ni)-*ni;

  vector<TEnIIterator> *newPtrs = mlnew vector<TEnIIterator>(sorting->size(), sorting->front());
  ITERATE(vector<TEnIIterator>, si, *sorting) {
    int valind=(*si)->example->operator[](attrNo).intV;
    (*newPtrs)[valf[valind]++] = *si;
    (**si).nodeIndex = (**si).nodeIndex * values + valind;
  }

  mldelete sorting;
  sorting = newPtrs;
}


void TSortedExamples_nodeIndices::sortByAttr(int attrNo, vector<TEnIIterator> *&sorting, int values)
{ vector<int> valf(values, 0);
  ITERATE(vector<TEnIIterator>, ii, *sorting) {
    TValue &val = (*ii)->example->operator[](attrNo);
    if (val.isSpecial())
      raiseError("attribute '%s' has undefined values", (*ii)->example->domain->getVar(attrNo)->get_name().c_str());
    valf[val.intV]++;
  }

  int id = 0;
  for(vector<int>::iterator ni=valf.begin(); ni!=valf.end(); ++ni)
	  *ni = (id+=*ni)-*ni;

  vector<TEnIIterator> *newPtrs = mlnew vector<TEnIIterator>(sorting->size(), sorting->front());
  ITERATE(vector<TEnIIterator>, si, *sorting)
    (*newPtrs)[valf[(*si)->example->operator[](attrNo).intV]++] = *si;

  mldelete sorting;
  sorting=newPtrs;
}


TIMColumnNode::TIMColumnNode(const int &anind, TIMColumnNode *anext, float nerr)
: index(anind),
  next(anext),
  nodeQuality(nerr)
{}


TIMColumnNode::~TIMColumnNode()
{ while(next) {
    TIMColumnNode *nn=next->next;
    next->next=NULL; // prevents stack overflow when the list deletes itself
    mldelete next;
    next=nn;
  }
}


TDIMColumnNode::TDIMColumnNode(const int &anind, const int &noofval, float *adist, TIMColumnNode *anext)
: TIMColumnNode(anind, anext),
  noOfValues(noofval),
  distribution(adist ? adist : mlnew float[noofval])
{ if (adist)
    computeabs();
  else {
    float *di = distribution;
    for(int c = noOfValues; c--; *(di++) = 0.0); 
    abs = -1.0;
  }
}


TDIMColumnNode::~TDIMColumnNode()
{ mldelete distribution; }


TIMColumnNode &TDIMColumnNode::operator += (const TIMColumnNode &other)
{ float *di = distribution, *de = distribution+noOfValues;
  const float *ddi = dynamic_cast<const TDIMColumnNode &>(other).distribution;
  while (di!=de)
    *(di++) += *(ddi++);
  return *this;
}


TFIMColumnNode::TFIMColumnNode(int anind, TIMColumnNode *anext, const float asum, const float asum2, const float aN)
: TIMColumnNode(anind, anext),
  sum(asum),
  sum2(asum2),
  N(aN)
{}


void TFIMColumnNode::add(const float &value, const float weight)
{ sum  += weight*value;
  sum2 += weight*value*value;
  N += weight;
}

TIMColumnNode &TFIMColumnNode::operator += (const TIMColumnNode &other)
{ TFIMColumnNode const &nde = dynamic_cast<const TFIMColumnNode &>(other);
  sum += nde.sum;
  sum2+= nde.sum2;
  N += nde.N;
  return *this;
}


T_ExampleIMColumnNode::T_ExampleIMColumnNode(PExample anexample, TIMColumnNode *anode)
: example(anexample),
  column(anode)
{}


T_ExampleIMColumnNode::T_ExampleIMColumnNode(const T_ExampleIMColumnNode &other)
: example(other.example),
  column(other.column)
{ const_cast<T_ExampleIMColumnNode &>(other).column = NULL; }


T_ExampleIMColumnNode::~T_ExampleIMColumnNode()
{ mldelete column; }


T_ExampleIMColumnNode &T_ExampleIMColumnNode::operator =(const T_ExampleIMColumnNode &other)
{ example = other.example;
  column = other.column;
  const_cast<T_ExampleIMColumnNode &>(other).column=NULL;
  return *this;
}



TIM::TIM(const int &avarType)
: varType(avarType),
  columns()
{}


int TIM::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);
  const_ITERATE(vector<T_ExampleIMColumnNode>, pi, columns)
    PVISIT((*pi).example);
  return 0;
}


int TIM::dropReferences()
{ DROPREFERENCES(TOrange::dropReferences);
  columns.clear();
  return 0;
}


bool TIM::fuzzy()
{ ITERATE(vector<T_ExampleIMColumnNode>, ci, columns)
    if (varType==TValue::INTVAR)
      for(TDIMColumnNode *dnode=dynamic_cast<TDIMColumnNode *>((*ci).column); dnode; dnode = (TDIMColumnNode *)(dnode->next)) {
        float *di=dnode->distribution;
        int c;
        for (c = dnode->noOfValues; c--; di++)
          if ((*di!=0.0) && (*di != dnode->abs))
            return true; // if !c, all is empty or only the last is non-empty
    }
    else
      for(TFIMColumnNode *fnode = dynamic_cast<TFIMColumnNode *>((*ci).column); fnode; fnode = (TFIMColumnNode *)(fnode->next))
        if (fnode->N*fnode->sum2 != sqr(fnode->sum))
          return true;

  return false;
}


TIMConstructor::TIMConstructor(const bool &anRE)
: recordRowExamples(anRE)
{}


PIM TIMConstructor::operator()(PExampleGenerator gen, const TVarList &aboundSet, const int &weightID)
{
  // Identify bound attributes
  vector<bool> bound = vector<bool>(gen->domain->attributes->size(), false);
  vector<bool> free = vector<bool>(gen->domain->attributes->size(), true);
  const_ITERATE(TVarList, evi, aboundSet) {
    int vn = gen->domain->getVarNum(*evi);
    bound[vn] = true;
    free[vn] = false;
  }

  return operator()(gen, bound, aboundSet, free, weightID);
}


PIM TIMConstructor::operator()(PExampleGenerator gen, const TVarList &aboundSet, const TVarList &afreeSet, const int &weightID)
{
  // Identify bound attributes
  vector<bool> bound=vector<bool>(gen->domain->attributes->size(), false);
  { const_ITERATE(TVarList, evi, aboundSet)
      bound[gen->domain->getVarNum(*evi)] = true;
  }

  vector<bool> free=vector<bool>(gen->domain->attributes->size(), false);
  { const_ITERATE(TVarList, evi, afreeSet)
      free[gen->domain->getVarNum(*evi)] = true;
  }

  return operator()(gen, bound, aboundSet, free, weightID);
}


PIM TIMBySorting::operator()(PExampleGenerator gen, const vector<bool> &bound, const TVarList &aboundSet, const vector<bool> &free, const int &weightID)
{
  PIM im=mlnew TIM(gen->domain->classVar->varType);

  vector<bool>::const_iterator freee(free.end());

  // prepare free domain for rowExamples if recordRowExamples==true
  PDomain freeDomain;
  PExampleTable rowTable;
  if (recordRowExamples) {
    TVarList freeSet;
    { TVarList::const_iterator vi(gen->domain->attributes->begin());
      const_ITERATE(vector<bool>, bi, free) {
        if (*bi)
          freeSet.push_back(*vi);
        vi++;
      }
    }
    freeDomain = mlnew TDomain(PVariable(), freeSet);
    rowTable = mlnew TExampleTable(freeDomain);
    im->rowExamples = rowTable;
  }

  PDomain boundDomain = mlnew TDomain(PVariable(), aboundSet);

  // prepare a sorted table with examples and graph node indices
  TSortedExamples_nodeIndices sorted(gen, bound, free);
  if (sorted.empty())
    raiseError("no examples");

  im->columns = vector<T_ExampleIMColumnNode>(sorted.maxIndex+1);

  // pointers to last elements of the lists
  vector<TIMColumnNode *> lastEl(im->columns.size(), (TIMColumnNode *)NULL);

  int classes = (im->varType==TValue::INTVAR) ? gen->domain->classVar->noOfValues() : -1;

  // Extract the incompatibility matrix
  if (recordRowExamples)
    rowTable->addExample(sorted.front().example.getReference()); // no problem - this example will be converted anyway...

  int rowIndex = 0;
  for(TSortedExamples_nodeIndices::iterator ebegin(sorted.begin()), eend(sorted.end()), eprev(ebegin);
      ebegin!=eend;
      eprev = ebegin++) {

    // Check equality of free attributes and increase rowIndex, if needed
    // We check for equality without converting the example - this is much cheaper
    if (ebegin!=eprev) {
      TExample::iterator ti((*ebegin).example->begin()), pi((*eprev).example->begin());
      vector<bool>::const_iterator bi(free.begin());
	    for( ; (bi!=freee) && (!*bi || (*ti==*pi)); bi++, ti++, pi++);
      if (bi!=freee) {
        rowIndex++;
        if (recordRowExamples)
          rowTable->addExample((*ebegin).example.getReference());
      }
    }

    // Add the example to the matrix
    int colIndex = (*ebegin).nodeIndex;
    if (classes>=0) {
      TIMColumnNode *coli=lastEl[colIndex];
      if (!coli) {
        T_ExampleIMColumnNode &node = im->columns[colIndex];
        node.example = mlnew TExample(boundDomain, (*ebegin).example.getReference());
        coli = lastEl[colIndex] = node.column = mlnew TDIMColumnNode(rowIndex, classes);
      }
      else if (coli->index!=rowIndex) {
        // This element is complete; let us compute abs
        dynamic_cast<TDIMColumnNode *>(coli)->computeabs();

        // Create new element, link the previous last to it and state that this one's now the last
        coli = mlnew TDIMColumnNode(rowIndex, classes);
        lastEl[colIndex]->next = coli;
        lastEl[colIndex] = coli;
      }
      // no need to check whether the class is special -- checked when building TSortedExamples_nodeIndices
      (dynamic_cast<TDIMColumnNode *>(coli))->distribution[(*ebegin).example->getClass().intV] += WEIGHT((*ebegin).example.getReference());
    }
    else {
      TIMColumnNode *coli=lastEl[colIndex];
      if (!coli)
        coli = lastEl[colIndex] = im->columns[colIndex].column = mlnew TFIMColumnNode(rowIndex);
      else if (coli->index!=rowIndex) {
        coli = mlnew TFIMColumnNode(rowIndex);
        lastEl[colIndex]->next = coli;
        lastEl[colIndex] = coli;
	    }
      // no need to check whether the class is special -- checked when building TSortedExamples_nodeIndices
      (dynamic_cast<TFIMColumnNode *>(coli))->add((*ebegin).example->getClass().floatV, WEIGHT((*ebegin).example.getReference()));
    }
  }

  // Complete the uncompleted elements by computing abs
  if (classes>=0)
    ITERATE(vector<TIMColumnNode *>, li, lastEl)
      if (*li)
        dynamic_cast<TDIMColumnNode *>(*li)->computeabs();

  return im;
}


PIM TIMConstructor::operator ()(PIMByRows imrows)
{
  PIM im = mlnew TIM(imrows->varType);
  im->columns = vector<T_ExampleIMColumnNode>();

  int column = 0;
  ITERATE(vector<PExample>, ei, imrows->columnExamples) 
    if (*ei) {
      im->columns.push_back(T_ExampleIMColumnNode(*ei));
      TIMColumnNode **lastnode = &im->columns.back().column;
      for(vector<TDIMRow>::const_iterator rbi(imrows->rows.begin()), ri(rbi), rei(imrows->rows.end()); ri!=rei; ri++) {
        const float *di = (*ri).nodes[column], *de = di+(*ri).noOfValues;
        while ((di!=de) && !*di);
        if (di!=de) { // column is not empty
          (*lastnode) = mlnew TDIMColumnNode(ri-rbi, (*ri).noOfValues, (*ri).nodes[column]);
          lastnode = &(*lastnode)->next;
        }
      }
      column++;
    }

  if (recordRowExamples) {
    im->rowExamples = mlnew TExampleTable(imrows->rows.front().example->domain);
    ITERATE(vector<TDIMRow>, ri, imrows->rows)
      im->rowExamples->addExample((*ri).example.getReference());
  }

  return im;
}


TDIMRow::TDIMRow(PExample ex, const int &n, const int &classes)
: example(ex)
{ nodes.reserve(n);
  for(int i=n; i--;) {
    float *ndist = mlnew float[classes];
    nodes.push_back(ndist);
    for(float *de = ndist + classes; ndist != de; *(ndist++) = 0.0);
  }
}


TDIMRow::~TDIMRow()
{ ITERATE(vector<float *>, di, nodes)
    mldelete *di;
}


PIMByRows TIMByRowsConstructor::operator()(PExampleGenerator gen, const TVarList &aboundSet, const int &weightID)
{
  // Identify bound attributes
  vector<bool> bound = vector<bool>(gen->domain->attributes->size(), false);
  vector<bool> free = vector<bool>(gen->domain->attributes->size(), true);
  const_ITERATE(TVarList, evi, aboundSet) {
    int vn = gen->domain->getVarNum(*evi);
    bound[vn] = true;
    free[vn] = false;
  }

  return operator()(gen, bound, aboundSet, free, weightID);
}


TIMByRows::TIMByRows(const int &avarType)
: varType(avarType)
{}



int TIMByRows::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);
  const_ITERATE(vector<PExample>, ei, columnExamples)
    PVISIT(*ei);
  const_ITERATE(vector<TDIMRow>, ri, rows)
    PVISIT((*ri).example);
  return 0;
}


int TIMByRows::dropReferences()
{ DROPREFERENCES(TOrange::dropReferences);
  columnExamples.clear();
  rows.clear();
  return 0;
}



PIMByRows TIMByRowsConstructor::operator()(PExampleGenerator gen, const TVarList &aboundSet, const TVarList &afreeSet, const int &weightID)
{
  // Identify bound attributes
  vector<bool> bound = vector<bool>(gen->domain->attributes->size(), false);
  { const_ITERATE(TVarList, evi, aboundSet)
      bound[gen->domain->getVarNum(*evi)]=true;
  }

  vector<bool> free = vector<bool>(gen->domain->attributes->size(), false);
  { const_ITERATE(TVarList, evi, afreeSet)
      free[gen->domain->getVarNum(*evi)]=true;
  }

  return operator()(gen, bound, aboundSet, free, weightID);
}


PIMByRows TIMByRowsBySorting::operator()(PExampleGenerator gen, const vector<bool> &bound, const TVarList &aboundSet, const vector<bool> &free, const int &weightID)
{
  int classes =   (gen->domain->classVar->varType==TValue::INTVAR)
                ? gen->domain->classVar->noOfValues()
                : -1;

  if (classes==-1)
    raiseError("these is no class or it is not discrete.");

  PIMByRows im = mlnew TIMByRows(gen->domain->classVar->varType);

  vector<bool>::const_iterator freee(free.end());

  // prepare free domain for rowExamples if recordRowExamples==true
  PDomain freeDomain;
  {
    TVarList freeSet;
    { TVarList::const_iterator vi(gen->domain->attributes->begin());
      const_ITERATE(vector<bool>, fi, free) {
        if (*fi)
          freeSet.push_back(*vi);
        vi++;
      }
    }
    freeDomain = mlnew TDomain(PVariable(), freeSet);
  }

  PDomain boundDomain = mlnew TDomain(PVariable(), aboundSet);

  // prepare a sorted table with examples and graph node indices
  TSortedExamples_nodeIndices sorted(gen, bound, free);
  if (sorted.empty())
    raiseError("no examples");
  int columns = sorted.maxIndex+1;
  im->columnExamples = vector<PExample>(columns, PExample());

  // Extract the incompatibility matrix
  im->rows.push_back(TDIMRow(mlnew TExample(freeDomain, sorted.front().example.getReference()), columns, classes));
  for(TSortedExamples_nodeIndices::iterator ebegin(sorted.begin()), eend(sorted.end()), eprev(ebegin);
      ebegin!=eend;
      eprev=ebegin++) {

    // Check equality of free attributes and increase rowIndex, if needed
    // We check for equality without converting the example - this is much cheaper
    if (ebegin!=eprev) {
      TExample::iterator ti((*ebegin).example->begin()), pi((*eprev).example->begin());
      vector<bool>::const_iterator fi(free.begin());
	    for( ; (fi!=freee) && (!*fi || (*ti==*pi)); fi++, ti++, pi++);
      if (fi!=freee)
        im->rows.push_back(TDIMRow(mlnew TExample(freeDomain, (*ebegin).example.getReference()), columns, classes));
    }

    int &nodeIndex = (*ebegin).nodeIndex;
    if (!im->columnExamples[nodeIndex])
      im->columnExamples[nodeIndex] = PExample(mlnew TExample(boundDomain, (*ebegin).example.getReference()));

    // Add the example to the matrix
    // no need to check whether the class is special -- checked when building TSortedExamples_nodeIndices
    im->rows.back().nodes[nodeIndex][(*ebegin).example->getClass().intV] += WEIGHT((*ebegin).example.getReference());
  }

  return im;
}


PIM TIMByIMByRows::operator()(PExampleGenerator gen, const vector<bool> &bound, const TVarList &aboundSet, const vector<bool> &free, const int &weightID)
{ PIMByRows imrows=TIMByRowsBySorting()(gen, bound, aboundSet, free, weightID);
  return TIMConstructor::operator()(imrows); 
}


TIMByRelief::TIMByRelief()
: distance(PExamplesDistance_Relief()),
  k(10),
  m(50),
  kFromColumns(0.0),
  ignoreSameExample(false),
  convertToBinary(false),
  correctClassFirst(false),
  allExamples(false),
  allSameNeighbours(false)
{}


PIM TIMByRelief::operator()(PExampleGenerator gen, const vector<bool> &bound, const TVarList &aboundSet, const vector<bool> &free, const int &weightID)
{ TIMByRowsByRelief imrr;
  imrr.k = k;
  imrr.m = m;
  imrr.kFromColumns = kFromColumns;

  imrr.ignoreSameExample = ignoreSameExample;
  imrr.convertToBinary = convertToBinary;
  imrr.correctClassFirst = correctClassFirst;
  imrr.allExamples = allExamples;
  imrr.allSameNeighbours = allSameNeighbours;

  PIMByRows imrows = imrr(gen, bound, aboundSet, free, weightID);
  return TIMConstructor::operator()(imrows);
}



class TCI_w {
public:
  long columnIndex; // index of the column (product of bound values)
  long freeIndex;   // index to freeExamples table
  TCI_w(const long &ci, const long &fi)
  : columnIndex(ci),
  freeIndex(fi)
  {}
};


class TDistRec {
public:
  float dist;
  long randoff;
  vector<TCI_w>::iterator exPointer;

  TDistRec(vector<TCI_w>::iterator ep, const int &roff, const float &adist)
  : dist(adist),
    randoff(roff),
    exPointer(ep)
  {};

  bool operator <(const TDistRec &other) const
  { return (dist==other.dist) ? (randoff<other.randoff) : (dist<other.dist); }
  bool operator !=(const TDistRec &other) const
  { return (dist!=other.dist) || (randoff!=other.randoff); }
};


TIMByRowsByRelief::TIMByRowsByRelief()
: k(10),
  m(50),
  kFromColumns(0.0),
  ignoreSameExample(false),
  convertToBinary(false),
  correctClassFirst(false),
  allExamples(false),
  allSameNeighbours(false)
{}


PIMByRows TIMByRowsByRelief::operator()(PExampleGenerator gen, const vector<bool> &, const TVarList &aboundSet, const vector<bool> &free, const int &weightID)
{
  int classes =   (gen->domain->classVar && (gen->domain->classVar->varType==TValue::INTVAR))
                ? gen->domain->classVar->noOfValues()
                : -1;

  if (classes==-1)
    raiseError("these is no class or it is not discrete.");

  TRandomGenerator rgen(gen->numberOfExamples());

  PDomain boundDomain=mlnew TDomain(PVariable(), aboundSet);
  PDomain freeDomain;
  {
    TVarList freeSet;
    { TVarList::const_iterator vi(gen->domain->attributes->begin());
      const_ITERATE(vector<bool>, fi, free) {
        if (*fi)
          freeSet.push_back(*vi);
        vi++;
      }
    }
    freeDomain=mlnew TDomain(gen->domain->classVar, freeSet);
  }

  vector<int> values;
  int columns = 1;
  const_ITERATE(TVarList, bvi, aboundSet) {
    int tvalues=(*bvi)->noOfValues();
    values.push_back(tvalues);
    columns*=tvalues;
  }

  PIMByRows im = mlnew TIMByRows(gen->domain->classVar->varType);
  im->columnExamples = vector<PExample>(columns, PExample());
  float myK = (kFromColumns>0.0) ? kFromColumns*columns : k;

  // converts examples to freeDomain
  // and divides them info classTables
  TExampleTable freeExamples(freeDomain);
  vector<vector<TCI_w> > classTables;
  vector<float> gN;
  for(int cl = classes; cl--; ) {
    classTables.push_back(vector<TCI_w>());
    gN.push_back(0.0);
  }

  {
    int freeIndex = 0;
    PEITERATE(ei, gen) {
      if ((*ei).getClass().isSpecial())
        continue;

      TExample boundExample(boundDomain, *ei);
      int ci = 0;
      for (int i = 0, vsize = values.size(); i<vsize; i++)
        if (boundExample[i].isSpecial())
          raiseError("attribute '%s' has undefined values", aboundSet[i]->get_name().c_str());
        else
          ci = ci*values[i]+boundExample[i].intV;

      TExample freeExample(freeDomain, *ei);
      if (weightID)
        freeExample.setMeta(weightID, TValue(WEIGHT(*ei)));
      freeExamples.addExample(*ei);
      int classIndex = (*ei).getClass().intV;
      classTables[classIndex].push_back(TCI_w(ci, freeIndex++));
      gN[classIndex] += WEIGHT(*ei);

      if (!im->columnExamples[ci])
        im->columnExamples[ci]=mlnew TExample(boundDomain, boundExample);
    }
  }

  PExamplesDistance_Relief useDistance = TExamplesDistanceConstructor_Relief()(PExampleGenerator(freeExamples), weightID);

  float gNtot = 0.0;
  ITERATE(vector<float>, gi, gN)
    gNtot += *gi;
  
  // the total number of examples
  long N = freeExamples.numberOfExamples();
  float actualN = 0;

  long eNum = -1;

  bool myAllExamples = allExamples || (m>N);

  for(float referenceExamples = 0, refWeight; myAllExamples ? (eNum+1<N) : (referenceExamples<m); referenceExamples+=refWeight) {
    // choose a random or consecutive example
    // This is probably not correct - examples with lower weights should have less chance to
    // be chosen. Neither multiplying the line with a low weight does not amortize for this
    // since this same example can be chosen on and on...
    eNum = myAllExamples ? eNum+1 : rgen.randlong(N);

    TExample &example = freeExamples[eNum];
    int eClass = example.getClass();
    refWeight = WEIGHT(example);

    im->rows.push_back(TDIMRow(mlnew TExample(example), columns, classes));

    // for each class
    for(int oClass = 0; oClass<classes; oClass++) 
      if (classTables[oClass].size()>0) {

        int adjustedClassIndex;
        if (convertToBinary)
          adjustedClassIndex = (oClass==eClass) ? 0 : 1;
        else if (correctClassFirst)
          adjustedClassIndex= (oClass==eClass) ? 0
                                               : ( (oClass>eClass) ? oClass : oClass+1);
        else
          adjustedClassIndex=oClass;

        // sort the examples by the distance
        set<TDistRec> neighset;
        ITERATE(vector<TCI_w>, epi, classTables[oClass])
          neighset.insert(TDistRec(epi, rgen.randlong(), useDistance->operator()(example, freeExamples[(*epi).freeIndex])));

        float classWeight = adjustedClassIndex ? gN[oClass] / (gNtot-gN[eClass])  :  1.0;

        set<TDistRec>::iterator ni(neighset.begin()), ne(neighset.end());

        if (ignoreSameExample)
          while(((*ni).dist<=0) && (ni!=ne))
            ni++;

        for(float needwei = myK, compWeight; (ni!=ne) && ((needwei>0) || allSameNeighbours && ((*ni).dist<=0)); needwei-=compWeight, ni++) {
          // determine the weight for the current example
          TCI_w &CI_w =* (*ni).exPointer;
          compWeight = WEIGHT(freeExamples[CI_w.freeIndex]);
          if (compWeight>needwei)
            compWeight = needwei;
          float koe = refWeight*compWeight*classWeight;
          actualN += fabs(koe);
          im->rows.back().nodes[CI_w.columnIndex][adjustedClassIndex] += koe;
        }
      }
  }

  return im;
}



TIMBlurer::TIMBlurer(const float &w, const float &ow, const bool &aw, const bool &oe)
: weight(w),
  origWeight(ow),
  adjustOrigWeight(aw),
  onlyEmpty(oe)
{}


TIMBlurer::TIMBlurer(PFloatList aaw, const float &ow, const bool &aw, const bool &oe)
: weight(-1),
  origWeight(ow),
  attrWeights(aaw),
  adjustOrigWeight(aw),
  onlyEmpty(oe)
{}


bool TIMBlurer::operator()(PIMByRows im)
{ TVarList &attributes = im->rows.front().example->domain->attributes.getReference();
  int attrs = attributes.size();
  int columns = im->rows.front().nodes.size();
  int classes = im->rows.front().noOfValues;
  float actOrigWeight;

  PFloatList myAttrWeights;

  if (weight>0.0) {
    if (weight>1.0)
      raiseError("weight is %5.3f; it should be lower than 1.0", weight);
    myAttrWeights = mlnew TFloatList(attrs, weight);
    actOrigWeight = adjustOrigWeight ? 1.0-weight : origWeight;
  }
  else {
    if (attrWeights && int(attrWeights->size())!=attrs)
      raiseError("invalid 'attrWeights' (size does not match the number of attributes)");

    myAttrWeights = attrWeights;
    if (adjustOrigWeight) {
      float sum = 0.0;
      PITERATE(TFloatList, wi, attrWeights)
        sum += *wi;
      actOrigWeight = sum>1.0 ? origWeight : 1.0-sum;
    }

    else
      actOrigWeight=origWeight;
  }

  vector<vector<float *> > impose;
  impose.reserve(im->rows.size());

  vector<TDIMRow *> *sortedRows = mlnew vector<TDIMRow *>();
  sortedRows->reserve(im->rows.size());

  ITERATE(vector<TDIMRow>, ri, im->rows) {
    impose.push_back(vector<float *>(columns, (float *)NULL));
    for(int i = columns; i; ) {
      float *fc = mlnew float[classes];
      impose.back()[--i] = fc;
      for(float *fe = fc+classes; fc!=fe; *(fc++) = 0.0);
    }
    sortedRows->push_back(& *ri);
  }
  const TDIMRow *firstRow = &im->rows.front();

  for(int attr=0; attr<attrs; attr++) { // we trust it is already sorted

    float attrWeight = myAttrWeights->at(attr);
    if (attrWeight>=0.0) {

      bool ordered = attributes[attr]->ordered;

      // impose by attr
      for(vector<TDIMRow *>::const_iterator grp_begin(sortedRows->begin()), grp_in1, grp_in2, grp_end, grp_totend(sortedRows->end());
          grp_begin!=grp_totend;
          grp_begin=grp_end) {

        // find the end of the group
        grp_end=grp_begin;
        while(++grp_end!=grp_totend) {
          TExample::const_iterator ei1=(*grp_begin)->example->begin(), ei2=(*grp_end)->example->begin(), ei1e=(*grp_begin)->example->end();
          for(int attrNo=0; (ei1!=ei1e) && ((attrNo==attr) || (*ei1==*ei2)); ei1++, ei2++, attrNo++);
          if (ei1!=ei1e)
            break;
        }

        for(grp_in1=grp_begin; grp_in1!=grp_end-1; grp_in1++)
          for(grp_in2=grp_in1+1; grp_in2!=grp_end; grp_in2++) {
            if (ordered) {
              int dif=(*grp_in1)->example->operator[](attr).intV - (*grp_in2)->example->operator[](attr).intV;
              if ((dif!=1) && (dif!=-1))
                continue;
            }
            else
              if ((*grp_in1)->example->operator[](attr).intV == (*grp_in2)->example->operator[](attr).intV)
                continue;
           
            { vector<float *>::iterator odi((*grp_in1)->nodes.begin());
              vector<float *> &imp = impose[*grp_in2-firstRow];
              for(vector<float *>::iterator idi(imp.begin()), ide(imp.end()); idi!=ide; idi++, odi++)
                for (float *idii= *idi, *idie = idii+classes, *odii = *odi; idii!=idie; *(idii++) += *(odii++)*attrWeight);
            }

            { vector<float *>::iterator odi((*grp_in2)->nodes.begin());
              vector<float *> &imp = impose[*grp_in1-firstRow];
              for(vector<float *>::iterator idi(imp.begin()), ide(imp.end()); idi!=ide; idi++, odi++)
                for (float *idii= *idi, *idie = idii+classes, *odii= *odi; idii!=idie; *(idii++) += *(odii++)*attrWeight);
            }
          }
      }
    }

    // sort by attribute attr
    vector<int> valf(attributes[attr]->noOfValues(), 0);
    ITERATE(vector<TDIMRow *>, ii, *sortedRows) {
      TValue &val=(*ii)->example->operator[](attr);
      if (val.isSpecial())
        raiseError("attribute '%s' has undefined values", attributes[attr]->get_name().c_str());
      valf[val.intV]++;
    }

    int id=0;
    for(vector<int>::iterator ni=valf.begin(); ni!=valf.end(); *(ni++)=(id+=*ni)-*ni);

    vector<TDIMRow *> *newPtrs=mlnew vector<TDIMRow *>(sortedRows->size(), (TDIMRow *)NULL);
    ITERATE(vector<TDIMRow *>, si, *sortedRows)
      (*newPtrs)[valf[(*si)->example->operator[](attr).intV]++] = *si;

    mldelete sortedRows;
    sortedRows=newPtrs;

  }

  mldelete sortedRows;

  vector<TDIMRow>::iterator oi=im->rows.begin();

  ITERATE(vector<vector<float *> >, ii, impose) {
    for(vector<float *>::iterator odi((*oi).nodes.begin()), idi((*ii).begin()), ide((*ii).end()); idi!=ide; idi++, odi++) {
      float *di, *de;
      if (onlyEmpty) {
        for (di = *odi, de = di+classes; (di!=de) && !*di; di++);
        if (di==de)
          continue;
      }
      float *idii;
      for (di = *odi, de = di+classes, idii = *idi; di!=de; *(di++) += *(idii++));
    }
    oi++;
  }

  return true;
}
