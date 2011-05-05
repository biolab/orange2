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


#include "random.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "domaindepot.hpp"
#include "filter.hpp"
#include "distvars.hpp"
#include "stladdon.hpp"

#include "crc.h"

#include "table.ppp"


TExampleTable::TExampleTable(PDomain dom, bool owns)
: TExampleGenerator(dom),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsExamples(owns)
{ version = ++generatorVersion; }


/* Careful: the meaning of 'owns' is the opposite of what we have
   in Python: if owns==true, then the table doesn't hold only the
   references to examples (in another table) but has its own examples! */
TExampleTable::TExampleTable(PExampleGenerator gen, bool owns)
: TExampleGenerator(gen->domain),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsExamples(owns)
{ 
  if (!ownsExamples) {
    lock = fixedExamples(gen);
    addExamples(lock);
  }
  else
    addExamples(gen);
}


TExampleTable::TExampleTable(PDomain dom, PExampleGenerator gen, bool filterMetas)
: TExampleGenerator(dom),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsExamples(true)
{ 
  addExamples(gen, filterMetas);
}


TExampleTable::TExampleTable(PExampleGenerator alock, int)
: TExampleGenerator(alock ? alock->domain : PDomain()),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsExamples(false),
  lock(alock)
{ 
  version = ++generatorVersion;
}


TExampleTable::TExampleTable(PExampleGeneratorList tables)
: TExampleGenerator(PDomain()),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsExamples(true),
  lock()
{
  if (!tables->size())
    raiseError("merging constructor was given no datasets to merge");

  TDomainList domains;
  int size = tables->front()->numberOfExamples();
  vector<TExampleIterator> iterators;
  PITERATE(TExampleGeneratorList, sdi, tables) {
    if ((*sdi)->numberOfExamples() != size)
      raiseError("cannot merge dataset of unequal sizes");
    domains.push_back((*sdi)->domain);
    iterators.push_back((*sdi)->begin());
  }

  TDomainMultiMapping mapping;
  domain = combineDomains(PDomainList(domains), mapping);

  int exno = 0;
  for(; iterators.front(); exno++) {
    TExample *example = mlnew TExample(domain);
    addExample(example);
    TDomainMultiMapping::const_iterator dmmi(mapping.begin());
    TExample::iterator ei(example->begin()), ee(example->end());
    TVarList::const_iterator vi(domain->variables->begin());
    for(; ei!=ee; ei++, dmmi++, vi++) {
      bool notfirst = 0;
      for(vector<pair<int, int> >::const_iterator sdmmi((*dmmi).begin()), sdmme((*dmmi).end()); sdmmi!=sdmme; sdmmi++) {
        if (!mergeTwoValues(*ei, (*iterators[(*sdmmi).first])[(*sdmmi).second], notfirst++ != 0))
          raiseError("mismatching value of attribute '%s' in example #%i", (*vi)->get_name().c_str(), exno);
      }
    }

    // copy meta attributes and increase the iterators
    for(vector<TExampleIterator>::iterator ii(iterators.begin()), ie(iterators.end()); ii!=ie; ++*(ii++)) {
      ITERATE(TMetaValues, mvi, (**ii).meta) {
        if (example->hasMeta((*mvi).first)) {
          if (!mergeTwoValues(example->getMeta((*mvi).first), (*mvi).second, true)) {
            PVariable metavar = domain->getMetaVar((*mvi).first, false);
            if (metavar && metavar->get_name().length())
              raiseError("Meta attribute '%s' has ambiguous values on example #%i", metavar->get_name().c_str(), exno);
            else
              raiseError("Meta attribute %i has ambiguous values on example #%i", (*mvi).first, exno);
          }
        }
        else
          example->setMeta((*mvi).first, (*mvi).second);
      }
    }
  }

  version = ++generatorVersion;
}


TExampleTable::~TExampleTable()
{ 
  if (examples) {
    if (ownsExamples)
      for(TExample **t = examples; t != _Last; )
        delete *(t++);
    free(examples);
  }
}


int TExampleTable::traverse(visitproc visit, void *arg) const
{ 
  TRAVERSE(TExampleGenerator::traverse);
  if (ownsExamples)
    for(TExample **ee = examples; ee != _Last; ee++)
      TRAVERSE((*ee)->traverse)

  return 0;
}


int TExampleTable::dropReferences() 
{ 
  DROPREFERENCES(TExampleGenerator::dropReferences);
  clear();
  return 0;
}


void TExampleTable::reserve(const int &i)
{ 
  if (!examples) {
    if (i) {
      examples = (TExample **)malloc(i * sizeof(TExample *));
      _Last = examples;
      _EndSpace = examples + i;
    }
    else {
      _Last = _EndSpace = examples;
    }
  }

  else {
    if (!i) {
      if (examples) {
        if (_Last == examples) {
          free(examples);
          _Last = _EndSpace = examples = NULL;
        }
        // else do nothing: reserve should not remove examples!
      }
      else {
        _Last = _EndSpace = examples;
      }
    }
    else {
      if (i>_Last - examples) {
        int lastofs = _Last - examples;
        TExample **newexamples = (TExample **)realloc(examples, i * sizeof(TExample));
        if (!newexamples)
          raiseErrorWho("resize", "out of memory");
        examples = newexamples;
        _Last = examples + lastofs;
        _EndSpace = examples + i;
      }
      // else do nothing: i is too small and reserve should not remove examples!
    }
  }
}


void TExampleTable::growTable()
{
  reserve(!examples ? 256 : int(1.25 * (_EndSpace - examples)));
}


void TExampleTable::shrinkTable()
{
  if (_Last == examples)
    reserve(0);
  else {
    int sze = int(1.25 * (_Last-examples));
    if (sze < 256)
      sze = 256;
    if (sze < _EndSpace - examples)
      reserve(sze);
  }
}


TExample &TExampleTable::at(const int &i)
{
  if (_Last == examples)
    raiseError("no examples");
  if ((i<0) || (i >= _Last-examples))
    raiseError("index %i out of range 0-%i", i, _Last-examples-1);

  return *examples[i];
}


const TExample &TExampleTable::at(const int &i) const
{
  if (_Last == examples)
    raiseError("no examples");
  if ((i<0) || (i >= _Last-examples))
    raiseError("index %i out of range 0-%i", i, _Last-examples-1);

  return *examples[i];
}


TExample &TExampleTable::back() 
{
  if (_Last == examples)
    raiseError("no examples");

  return *_Last[-1];
}


const TExample &TExampleTable::back() const
{
  if (_Last == examples)
    raiseError("no examples");

  return *_Last[-1];
}


void TExampleTable::clear()
{
  if (examples) {
    if (ownsExamples)
      while (_Last != examples)
        delete *--_Last;
    free(examples);
  }
  _Last = _EndSpace = examples = NULL;
  examplesHaveChanged();
}


bool TExampleTable::empty() const
{ 
  return (_Last == examples);
}


TExample &TExampleTable::front() 
{
  if (_Last == examples)
    raiseError("no examples");

  return **examples;
}


const TExample &TExampleTable::front() const
{
  if (_Last == examples)
    raiseError("no examples");

  return **examples;
}


TExample &TExampleTable::operator[](const int &i)
{
  return *examples[i];
}


const TExample &TExampleTable::operator[](const int &i) const
{
  return *examples[i];
}


void TExampleTable::push_back(TExample *x)
{
  if (_Last == _EndSpace)
    growTable();
  *(_Last++) = x;

  examplesHaveChanged();
}


TExample &TExampleTable::new_example()
{
  TExample *x = mlnew TExample(domain);
  push_back(x);
  return *x;
}


void TExampleTable::delete_last()
{ if (_Last == examples)
    raiseError("no examples");
  erase(_Last-1);
}


int TExampleTable::size() const
{
  return examples ? _Last - examples : 0;
}


void TExampleTable::erase(const int &sti)
{
  if (_Last == examples)
    raiseError("no examples");
  if (sti >= _Last-examples)
    raiseError("index %i out of range 0-%i", sti, _Last-examples-1);
  erase(examples + sti);
}


void TExampleTable::erase(const int &sti, const int &eni)
{
  if (_Last == examples)
    raiseError("no examples");
  if (sti >= _Last-examples)
    raiseError("index %i out of range 0-%i", sti, _Last-examples-1);
  erase(examples + sti, examples + eni);
}


void TExampleTable::erase(TExample **ptr)
{
  if (ownsExamples)
    delete *ptr;
  memmove(ptr, ptr+1, sizeof(TExample **)*(_Last - ptr - 1));
  _Last--;
  examplesHaveChanged();
}


void TExampleTable::erase(TExample **fromPtr, TExample **toPtr)
{
  if (ownsExamples) {
    TExample **ee = fromPtr;
    while (ee != toPtr)
      delete *(ee++);
  }

  memmove(fromPtr, toPtr, sizeof(TExample **)*(_Last - toPtr));

  _Last -= (toPtr - fromPtr);

  shrinkTable();
  examplesHaveChanged();
}


void TExampleTable::insert(const int &sti, const TExample &ex)
{
  if (ex.domain != domain)
    raiseError("examples has invalid domain (ExampleTable.insert doesn't convert)");
  if (sti > _Last-examples)
    raiseError("index %i out of range 0-%i", sti, _Last-examples);
  
  if (_Last == _EndSpace)
    growTable();

  TExample **sp = examples + sti;
  memmove(sp+1, sp, sizeof(TExample **)*(_Last - sp));
  *sp = ownsExamples ? CLONE(TExample, &ex) : const_cast<TExample *>(&ex);
  _Last++;

  examplesHaveChanged();
}


TExampleIterator TExampleTable::begin()
{
  return TExampleIterator(this, examples ? *examples : NULL, (void *)examples);
}


void TExampleTable::copyIterator(const TExampleIterator &src, TExampleIterator &dest)
{ 
  TExampleGenerator::copyIterator(src, dest);
  dest.data = src.data;
}


void TExampleTable::increaseIterator(TExampleIterator &it)
{
  if (++((TExample **&)(it.data)) == _Last)
    deleteIterator(it);
  else
    it.example = *(TExample **)(it.data);
}


bool TExampleTable::sameIterators(const TExampleIterator &i1, const TExampleIterator &i2)
{
  return (i1.data==i2.data);
}


bool TExampleTable::remove(TExampleIterator &it)
{
  erase( (TExample **)it.data );
  examplesHaveChanged();
  return true;
}


bool TExampleTable::randomExample(TExample &ex)
{
  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator();

  if (!size())
    return 0;
  ex = operator[](randomGenerator->randint(size()));
  return true;
}


/*  Searches for the example in the table and returns its class, if found. If there are more
    different classes, it raises an exception, if example has no DK's, or returns DK otherwise.
    If there's no corresponding example, it returns DK. 
    IN FUTURE: This method should return distributed values if more answers are possible.
*/
TValue TExampleTable::operator ()(const TExample &exam)
{
  if (empty())
    return domain->classVar->DK();
  TExample cexam(exam);
  cexam.setClass(domain->classVar->DK());

  bool hasValue = false;
  TValue toret;
  for(TExample **ri = examples; ri!=_Last; ri++)
    if (cexam.compatible(**ri)) {
      if (!hasValue) {
        hasValue = true;
        toret = (**ri).getClass(); 
      }
      else if (!toret.compatible((**ri).getClass())) {
        // returns DK if the query contains specials, raises an exception otherwise
        int Na = domain->attributes->size();
        for(TExample::iterator vi(cexam.begin()); !((*vi).isSpecial()) && --Na; ++vi);
        if (Na)
          return domain->classVar->DK();
        else
          raiseError("ambiguous example (cannot determine the class value)");
      }
    }
  return hasValue ? toret : domain->classVar->DK();
}


int TExampleTable::numberOfExamples()
{
  return size();
}


float TExampleTable::weightOfExamples(const int &weightID) const
{
  float weight = 0;
  for(TExample **ri = examples; ri != _Last; ri++)
    weight += WEIGHT(**ri);
  return weight;
}


void TExampleTable::addExample(const TExample &example, bool filterMetas)
{
  if (ownsExamples)
    if (example.domain == domain)
      push_back(CLONE(TExample, &example));
    else
      push_back(mlnew TExample(domain, example, !filterMetas));
  else
    if (example.domain == domain)
      push_back(const_cast<TExample *>(&example));
    else
      raiseError("domain mismatch (cannot convert a reference to example)");

  examplesHaveChanged();
}


void TExampleTable::addExample(TExample *example)
{
  if (example->domain != domain)
    raiseError("cannot add pointers to examples of different domains");
  push_back(example);
  examplesHaveChanged();
}

void TExampleTable::addExamples(PExampleGenerator gen, bool filterMetas)
{
  if (ownsExamples)
    if (gen->domain == domain)
      PEITERATE(ei, gen)
        push_back(CLONE(TExample, &*ei)); 
    else
      PEITERATE(ei, gen)
        push_back(mlnew TExample(domain, *ei, !filterMetas)); 

  else {
    if (gen->domain == domain)
      PEITERATE(ei, gen)
        push_back(&*ei); 
    else
      raiseError("domain mismatch (cannot convert a reference to example)");
  }

  examplesHaveChanged();
}


bool TExampleTable::removeExamples(TFilter &filter)
{ 
  TExample **ri = examples, **ci;
  for( ; (ri!=_Last) && !filter(**ri); ri++);
  if ((ci=ri)==_Last)
    return 0;

  while(++ri!=_Last)
    if (!filter(**ri)) {
      if (ownsExamples)
        delete *ci;
      **(ci++) = **ri;
    }
  erase(ci, _Last);

  examplesHaveChanged();
  return 1;
}


bool TExampleTable::removeExample(TExample &exam)
{ // 'filter' dies before 'exam' and may contain a wrapped reference
  TFilter_sameExample filter = TFilter_sameExample(PExample(exam));
  return removeExamples(filter);
}


bool TExampleTable::removeCompatible(TExample &exam)
{ // 'filter' dies before 'exam' and may contain a wrapped reference
  TFilter_compatibleExample filter = TFilter_compatibleExample(PExample(exam));
  return removeExamples(filter);
}


class TExI {
public:
  TExample *example;
  int i;
  
  TExI(TExample *ex = NULL, const int ii=0)
  : example(ex),
    i(ii)
  {}
};


bool lesstexi(const TExI &a, const TExI &b)
{ return *a.example < *b.example; }


void TExampleTable::removeDuplicates(const int &weightID)
{ 
  if (empty())
    return;

  vector<TExI> exi(_Last - examples);
  int i = 0;
  for(TExample **ep = examples; ep!=_Last; exi[i] = TExI(*(ep++), i), i++);
  stable_sort(exi.begin(), exi.end(), lesstexi);

  bool removed = false;
  vector<TExI>::iterator fromPtr(exi.begin()), toPtr(fromPtr), ePtr(exi.end()); 
  while(++fromPtr != ePtr) {
    if (*(*fromPtr).example == *(*toPtr).example) {
      if (weightID)
        (*(*toPtr).example)[weightID].floatV += WEIGHT(*(*fromPtr).example);
      if (ownsExamples)
        delete examples[(*fromPtr).i];
      examples[(*fromPtr).i] = NULL;
      removed = true;
    }
    else
      toPtr = fromPtr;
  }

  if (!removed)
    return;

  TExample **fromE = examples;
  while (*fromE) // do not need to check !=_Last; there is a null pointer somewhere
    fromE++;

  TExample **toE = fromE++; // toE points to the next free spot
  for(; fromE != _Last; fromE++)
    if (*fromE)
      *toE++ = *fromE;

  _Last = toE;

  shrinkTable();
  examplesHaveChanged();
}


// Changes the domain and converts all the examples.
void TExampleTable::changeDomain(PDomain dom, bool filterMetas)
{
  domain = dom;
  if (ownsExamples)
    for (TExample **ri = examples; ri!=_Last; ri++) {
      TExample *tmp = mlnew TExample(dom, **ri, !filterMetas);
      delete *ri;
      *ri = tmp;
    }

  else {
    for (TExample **ri = examples; ri!=_Last; ri++)
      *ri = mlnew TExample(dom, **ri, !filterMetas);
    ownsExamples = false;
    lock = PExampleGenerator();
  }

  examplesHaveChanged();
}


void TExampleTable::addMetaAttribute(const int &id, const TValue &value)
{ 
  PEITERATE(ei, this)
    (*ei).setMeta(id, value);

  examplesHaveChanged();
}


void TExampleTable::copyMetaAttribute(const int &id, const int &source, TValue &defaultVal)
{
  if (source) {
    PEITERATE(ei, this)
      (*ei).setMeta(id, (*ei)[source]);
    examplesHaveChanged();
  }
  else
    addMetaAttribute(id, defaultVal);
}


void TExampleTable::removeMetaAttribute(const int &id)
{
  PEITERATE(ei, this)
    (*ei).removeMetaIfExists(id);

  examplesHaveChanged();
}


class TCompVar {
public:
  int varNum;

  TCompVar(int vn) : varNum(vn) {}
  bool operator()(const TExample *e1, const TExample *e2) const { return (*e1)[varNum].compare((*e2)[varNum])<0; }
};


void TExampleTable::sort()
{
  vector<int> empty;
  sort(empty);
}


// Sort order is reversed (least important first!)
void TExampleTable::sort(vector<int> &sortOrder)
{ 
  if (!sortOrder.size()) 
    for(int i = domain->variables->size(); i; )
      sortOrder.push_back(--i);

  int ssize = _EndSpace-examples;
  int lastOfs = _Last - examples;
  TExample **temp = (TExample **)malloc(ssize * sizeof(TExample *));

  try {
    const_ITERATE(vector<int>, bi, sortOrder) {
      int noVal = domain->getVar(*bi)->noOfValues();
      if (noVal>0) {
        vector<int> valf(noVal+1, 0);
        TExample **t;
        int id = 0;

        for(t = examples; t!= _Last; t++) {
          const TValue &val = (**t)[*bi];
          const int intV = val.isSpecial() ? noVal : val.intV;
          if (intV > noVal) {
            free(temp);
            raiseError("value out attribute '%s' of range", domain->variables->operator[](*bi)->get_name().c_str());
          }
          valf[intV]++;
        }

        for(vector<int>::iterator ni = valf.begin(); ni!=valf.end(); ni++) {
          const int ini = *ni;
          *ni = id;
          id += ini;
        }

        for(t = examples; t!= _Last; t++) {
          const TValue &val = (**t)[*bi];
          const int intV = val.isSpecial() ? noVal : val.intV;
          temp[valf[intV]++] = *t;
        }

        t = examples;
        examples = temp;
        temp = t;
        _Last = examples + lastOfs;
        _EndSpace = examples + ssize;
      }

      else
        stable_sort(examples, _Last, TCompVar(*bi));
    }
  }
  catch (...) {
    examplesHaveChanged();
    free(temp);
    throw;
  }

  free(temp);

  examplesHaveChanged();
}

void TExampleTable::shuffle()
{
  if (size() <= 1)
    return;

  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator();
   
  for(TExample **ei = examples+1; ei != _Last; ei++) {
    const int st = randomGenerator->randint(ei - examples);
    TExample *s = *ei;
    *ei = examples[st];
    examples[st] = s;
  }
}

int TExampleTable::checkSum(const bool includeMetas)
{ unsigned long crc;
  INIT_CRC(crc);

  for(TExample **ei = examples, **ee = _Last; ei!=ee; (*ei++)->addToCRC(crc, includeMetas));

  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}

int TExampleTable::checkSum(const bool includeMetas) const
{ unsigned long crc;
  INIT_CRC(crc);

  for(TExample **ei = examples, **ee = _Last; ei!=ee; (*ei++)->addToCRC(crc, includeMetas));

  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}

bool TExampleTable::hasMissing() const
{
  for(TExample **ei = examples, **ee = _Last; ei != ee; ei++)
    if ((*ei)->hasMissing())
      return true;
  return false;
}


bool TExampleTable::hasMissingClass() const
{
  if (!domain->classVar)
    raiseError("data has no class");
    
  for(TExample **ei = examples, **ee = _Last; ei != ee; ei++)
    if ((*ei)->missingClass())
      return true;
  return false;
}

void TExampleTable::sortByPointers()
{
  std::sort((int *)examples, (int *)_Last);
}
