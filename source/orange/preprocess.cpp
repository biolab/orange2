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


#include "getarg.hpp"
#include <fstream>
#include <string>

#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

#include "spec_gen.hpp"

#include "preprocessors.hpp"

#include "preprocess.ppp"


DEFINE_TOrangeVector_classDescription(PPreprocessor, "TPreprocessorList")


void TPreprocessor::addFilterAdapter(PFilter filter, vector<PExampleGenerator> &generators)
{ generators.push_back(PExampleGenerator(mlnew TFilteredGenerator(filter, generators.back()))); 
}


PExampleGenerator TPreprocessor::filterExamples(PFilter filter, PExampleGenerator generator)
{ TFilteredGenerator fg(filter, generator);
  return PExampleGenerator(mlnew TExampleTable(PExampleGenerator(fg))); 
}


void TPreprocessor::replaceWithTable(vector<PExampleGenerator> &generators, PExampleGenerator table)
{ generators.clear();
  generators.push_back(table);
}

void TPreprocessor::storeToTable(vector<PExampleGenerator> &generators, PDomain domain)
{ if ((generators.size()>1) || (generators.back()->domain != domain)) {
    TExampleTable *newTable=domain ? mlnew TExampleTable(domain, generators.back())
                                   : mlnew TExampleTable(generators.back());

    generators.clear();
    generators.push_back(PExampleGenerator(newTable));
  }
}


/*~******* TPreprocess class  */

TPreprocess::TPreprocess()
 {}

TPreprocess::TPreprocess(const TProgArguments &args, const string &argname)
{
  const_ITERATE(TMultiStringParameters, mi, args.options)
    if ((*mi).first==argname) {
      const string &line=(*mi).second;
      if (line[0]=='!') {
        ifstream inpstream(string(line.begin()+1, line.end()).c_str());
        readStream(inpstream);
      }
      else
        addAdapter(line, line);
    }
}


TPreprocess::TPreprocess(istream &istr)
{ readStream(istr); }



void TPreprocess::readStream(istream &istr)
{
  int lineNo=0;
  while (!istr.eof()) {
    if (istr.fail() || istr.bad())
      raiseError("error in preprocessor file");

    char errorStr[128];
    sprintf(errorStr, "Preprocessor (%i)", ++lineNo);

    char line[1024], *curr=line;
    istr.getline(line, 1024);
    if (istr.gcount()==1023)
      raiseError("%s: line too long", errorStr);

    for(;*curr && (*curr<=' ');curr++); // skip whitespace

    if (*curr)
      addAdapter(curr, errorStr);
  }
}

void TPreprocess::addAdapter(PPreprocessor pp)
{ push_back(pp); }

void TPreprocess::addAdapter(const string &adapterLine, const string &errorStr)
{
  string::const_iterator anamee(adapterLine.begin()), ae(adapterLine.end());
  for( ; (anamee!=ae) && (*anamee>' '); anamee++);
  string adapterName(adapterLine.begin(), anamee), adapterArguments(anamee, ae);

  char fch=adapterName[0];
  if ((fch=='#') || (fch=='%') || (fch=='|') || (adapterName=="rem")) return; // comment
  
  #define DO(x) else if (adapterName==#x) addAdapter(mlnew TPreprocessor_##x(errorStr, adapterArguments))
  DO(drop);
  DO(take);
  DO(ignore);
  DO(select);
  DO(remove_duplicates);
  DO(skip_missing);
  DO(only_missing);
  DO(skip_missing_classes);
  DO(only_missing_classes);
  DO(noise);
  DO(class_noise);
  DO(missing);
  DO(class_missing);
  DO(cost_weight);
  DO(censor_weight);
  DO(discretize);

  DO(move_to_table);
  #undef DO

  else
    raiseError("%s: unrecognized command", errorStr.c_str());
}

PExampleGenerator TPreprocess::operator()(PExampleGenerator gen, long &weightID, const long &aweightID)
{
  if (empty())
    return mlnew TExampleTable(gen);

  weightID=aweightID;

  vector<PExampleGenerator> generators(1, PExampleGenerator(mlnew TExampleTable(gen)));
  this_ITERATE(ppi)
    (*ppi)->operator()(generators, weightID);

  PExampleGenerator result=mlnew TExampleTable(generators.back()->domain);

  if (weightID!=aweightID) {
    string wname="pp_weight";
    while (result->domain->getVarNum(wname, false)>=0) {
      char vno[32];
      sprintf(vno, "%d", randint(1000));
      wname=string("pp_weight")+vno;
    }
    result->domain->metas.push_back(TMetaDescriptor(weightID, mlnew TFloatVariable(wname)));
  }

  result.AS(TExampleTable)->addExamples(generators.back());
  generators.clear();

  return result;
}



