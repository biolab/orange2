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


#include "random.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "errors.hpp"
#include "filter.hpp"
#include "distvars.hpp"
#include "stladdon.hpp"

#include "table.ppp"


TExampleTable::TExampleTable(PDomain dom)
: TExampleGenerator(dom),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsPointers(true)
{ version = ++generatorVersion; }


TExampleTable::TExampleTable(PExampleGenerator gen)
: TExampleGenerator(gen->domain),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsPointers(true)
{ addExamples(gen); }


TExampleTable::TExampleTable(PDomain dom, PExampleGenerator gen)
: TExampleGenerator(dom),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsPointers(true)
{ addExamples(gen); }


TExampleTable::TExampleTable(PDomain dom, PExampleGenerator alock, int)
: TExampleGenerator(dom),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsPointers(false),
  lock(alock)
{ version = ++generatorVersion; }


TExampleTable::TExampleTable(PExampleGenerator orig, int)
: TExampleGenerator(orig->domain),
  examples(NULL),
  _Last(NULL),
  _EndSpace(NULL),
  ownsPointers(false),
  lock(fixedExamples(orig))
{ addExamples(orig); }


TExampleTable::~TExampleTable()
{ if (examples) {
    if (ownsPointers)
      for(TExample **t = examples; t != _Last; )
        delete *(t++);
    free(examples);
  }
}


int TExampleTable::traverse(visitproc visit, void *arg) 
{ TRAVERSE(TExampleGenerator::traverse);
  for(TExample **ee = examples; ee != _Last; ee++)
    TRAVERSE((*ee)->traverse)

  return 0;
}


int TExampleTable::dropReferences() 
{ DROPREFERENCES(TExampleGenerator::dropReferences);
  clear();
  examplesHaveChanged();
  return 0;
}


void TExampleTable::reserve(const int &i)
{ if (!examples) {
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
{ reserve(!examples ? 256 : int(1.25 * (_EndSpace - examples))); }


void TExampleTable::shrinkTable()
{ if (_Last == examples)
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
{ if (_Last == examples)
    raiseError("no examples");
  if ((i<0) || (i >= _Last-examples))
    raiseError("index %i out of range 0-%i", i, _Last-examples-1);

  return *examples[i];
}


const TExample &TExampleTable::at(const int &i) const
{ if (_Last == examples)
    raiseError("no examples");
  if ((i<0) || (i >= _Last-examples))
    raiseError("index %i out of range 0-%i", i, _Last-examples-1);

  return *examples[i];
}


TExample &TExampleTable::back() 
{ if (_Last == examples)
    raiseError("no examples");

  return **_Last;
}


const TExample &TExampleTable::back() const
{ if (_Last == examples)
    raiseError("no examples");

  return **_Last;
}


void TExampleTable::clear()
{ if (examples) {
    while (_Last != examples)
      delete *--_Last;
    free(examples);
  }
  _Last = _EndSpace = examples = NULL;
  examplesHaveChanged();
}

bool TExampleTable::empty() const
{ return (_Last == examples); }


TExample &TExampleTable::front() 
{ if (_Last == examples)
    raiseError("no examples");

  return **examples;
}


const TExample &TExampleTable::front() const
{ if (_Last == examples)
    raiseError("no examples");

  return **examples;
}


TExample &TExampleTable::operator[](const int &i)
{ return *examples[i]; }


const TExample &TExampleTable::operator[](const int &i) const
{ return *examples[i]; }


void TExampleTable::push_back(TExample *x)
{ if (_Last == _EndSpace)
    growTable();
  *(_Last++) = x;

  examplesHaveChanged();
}


int TExampleTable::size() const
{ return examples ? _Last - examples : 0; }


void TExampleTable::erase(const int &sti)
{ if (_Last == examples)
    raiseError("no examples");
  if (sti >= _Last-examples)
    raiseError("index %i out of range 0-%i", sti, _Last-examples-1);
  erase(examples + sti);
}


void TExampleTable::erase(const int &sti, const int &eni)
{ if (_Last == examples)
    raiseError("no examples");
  if (sti >= _Last-examples)
    raiseError("index %i out of range 0-%i", sti, _Last-examples-1);
  erase(examples + sti, examples + eni);
}


void TExampleTable::erase(TExample **ptr)
{ if (ownsPointers)
    delete ptr;
  memmove(ptr, ptr+1, _Last - ptr - 1);
  _Last--;
}


void TExampleTable::erase(TExample **fromPtr, TExample **toPtr)
{ if (ownsPointers) {
    TExample **ee = fromPtr;
    while (ee != toPtr)
      delete *(ee++);
  }

  memmove(fromPtr, toPtr, _Last - toPtr);

  _Last -= (toPtr - fromPtr);

  shrinkTable();
  examplesHaveChanged();
}


void TExampleTable::insert(const int &sti, const TExample &ex)
{ if (_Last == examples)
    raiseError("no examples");
  if (sti >= _Last-examples)
    raiseError("index %i out of range 0-%i", sti, _Last-examples-1);
  
  if (_Last == _EndSpace)
    growTable();

  TExample **sp = examples + sti;
  memmove(sp+1, sp, _Last - sp);
  *sp = ownsPointers ? CLONE(TExample, &ex) : const_cast<TExample *>(&ex);

  examplesHaveChanged();
}


TExampleIterator TExampleTable::begin()
{ return TExampleIterator(this, *examples, (void *)examples); }


void TExampleTable::copyIterator(const TExampleIterator &src, TExampleIterator &dest)
{ 
  TExampleGenerator::copyIterator(src, dest);
  dest.data = src.data;
}


void TExampleTable::increaseIterator(TExampleIterator &it)
{ if (++((TExample **&)(it.data)) == _Last)
    deleteIterator(it);
  else
    it.example = *(TExample **)(it.data);
}


bool TExampleTable::sameIterators(const TExampleIterator &i1, const TExampleIterator &i2)
{ return (i1.data==i2.data); }


bool TExampleTable::remove(TExampleIterator &it)
{ erase( (TExample **)it.data );
  examplesHaveChanged();
  return true;
}


bool TExampleTable::randomExample(TExample &ex)
{ if (!size())
    return 0;
  ex = operator[](LOCAL_OR_GLOBAL_RANDOM.randint(size()));
  return true;
}


/*  Searches for the example in the table and returns its class, if found. If there are more
    different classes, it raises an exception, if example has no DK's, or returns DK otherwise.
    If there's no corresponding example, it returns DK. 
    IN FUTURE: This method should return distributed values if more answers are possible.
*/
TValue TExampleTable::operator ()(const TExample &exam)
{ if (empty()) return domain->classVar->DK();
  TExample cexam(exam);
  cexam.setClass(domain->classVar->DK());

  bool hasValue=0;
  TValue toret;
  for(TExample **ri = examples; ri!=_Last; ri++)
    if (cexam.compatible(**ri)) {
      if (!hasValue) {
        hasValue=1;
        toret=(**ri).getClass(); 
      }
      else if (!toret.compatible((**ri).getClass())) {
        // returns DK if the query contains specials, raises an exception otherwise
        int Na=domain->attributes->size();
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
{ return size(); }


void TExampleTable::addExample(const TExample &example)
{ if (ownsPointers)
    if (example.domain == domain)
      push_back(CLONE(TExample, &example));
    else
      push_back(mlnew TExample(domain, example));
  else
    if (example.domain == domain)
      push_back(const_cast<TExample *>(&example));
    else
      raiseError("domain mismatch (cannot convert a reference to example)");

  examplesHaveChanged();
}


void TExampleTable::addExamples(PExampleGenerator gen)
{ if (ownsPointers)
    if (gen->domain == domain)
      PEITERATE(ei, gen)
        push_back(CLONE(TExample, &*ei)); 
    else
      PEITERATE(ei, gen)
        push_back(mlnew TExample(domain, *ei)); 

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
{ TExample **ri=examples, **ci;
  for( ; (ri!=_Last) && !filter(**ri); ri++);
  if ((ci=ri)==_Last)
    return 0;
  while(++ri!=_Last)
    if (!filter(**ri)) {
      if (ownsPointers)
        delete *ci;
      **(ci++)=**ri;
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
        (*toPtr).example->meta[weightID].floatV += WEIGHT(*(*fromPtr).example);
      if (ownsPointers)
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
void TExampleTable::changeDomain(PDomain dom)
{ domain = dom;
  if (ownsPointers)
    for (TExample **ri = examples; ri!=_Last; ri++) {
      TExample *tmp = mlnew TExample(dom, **ri);
      delete *ri;
      *ri = tmp;
    }

  else {
    for (TExample **ri = examples; ri!=_Last; ri++)
      *ri = mlnew TExample(dom, **ri);
    ownsPointers = false;
    lock = PExampleGenerator();
  }

  examplesHaveChanged();
}


void TExampleTable::addMetaAttribute(const int &id, const TValue &value)
{ PEITERATE(ei, this)
    (*ei).meta.setValue(id, value);

  examplesHaveChanged();
}


void TExampleTable::copyMetaAttribute(const int &id, const int &source, TValue &defaultVal)
{ if (source) {
    PEITERATE(ei, this)
      (*ei).meta.setValue(id, (*ei).meta[source]);
    examplesHaveChanged();
  }
  else
    addMetaAttribute(id, defaultVal);
}


void TExampleTable::removeMetaAttribute(const int &id)
{ PEITERATE(ei, this)
    (*ei).meta.removeValueIfExists(id);

  examplesHaveChanged();
}


class TCompVar {
public:
  int varNum;

  TCompVar(int vn) : varNum(vn) {}
  bool operator()(const TExample *e1, const TExample *e2) const { return (*e1)[varNum].compare((*e2)[varNum])<0; }
};


void TExampleTable::sort()
{ vector<int> empty;
  sort(empty);
}


void TExampleTable::sort(vector<int> &sortOrder)
{ 
  if (!sortOrder.size()) 
    for(int i=domain->variables->size(); i; )
      sortOrder.push_back(--i);

  int ssize = _EndSpace-examples;
  int lastOfs = _Last - examples;
  TExample **temp = (TExample **)malloc(ssize * sizeof(TExample *));

  const_ITERATE(vector<int>, bi, sortOrder) {
    int noVal = domain->variables->operator[](*bi)->noOfValues();
    if (noVal>0) {
      vector<int> valf(noVal+1, 0);
      TExample **t;
      int id = 0;

      for(t = examples; t!= _Last; t++) {
        const TValue &val = (**t)[*bi];
        const int intV = val.isSpecial() ? noVal : val.intV;
        if (intV > noVal)
          raiseError("value out attribute '%s' of range", domain->variables->operator[](*bi)->name.c_str());
        valf[intV]++;
      }

      for(vector<int>::iterator ni = valf.begin(); ni!=valf.end(); *(ni++)=(id+=*ni)-*ni);

      for(t = examples; t!= _Last; t++) {
        const TValue &val = (**t)[*bi];
        const int intV = val.isSpecial() ? noVal : val.intV;
        temp[valf[intV]++] = *t;
      }

      t = examples;
      examples = temp;
      temp = t;
      _Last = examples + lastOfs;
    }

    else
      stable_sort(examples, _Last, TCompVar(*bi));
  }

  free(temp);

  examplesHaveChanged();
}


TExamplePointerTable::TExamplePointerTable(PDomain dom)
: TExampleTable(dom, PExampleGenerator(), 1)
{}


TExamplePointerTable::TExamplePointerTable(PExampleGenerator orig)
: TExampleTable(orig, 1)
{}


// This doesn't copy examples - the second argument serves only for locking
TExamplePointerTable::TExamplePointerTable(PDomain dom, PExampleGenerator lock)
: TExampleTable(dom, lock, 1)
{}
