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
#include "stladdon.hpp"
#include "errors.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>


TProgArguments::TProgArguments() :
allowSpaces(false)
{}

/*  A constructor which is given a list of possible options and standard C-like fields with
    number of arguments and an array of char * pointers to arguments. If rupUnrec=true, unrecognized
    options are reported. */
TProgArguments::TProgArguments(const string &poss_options, int argc, char *argv[], bool repUnrec, bool parenth, bool anallowSpaces)
  : possibleOptions(), options(), unrecognized(), direct(), allowSpaces(anallowSpaces)
{ findPossibleOptions(poss_options);
  vector<string> optionsList;

  if (argc>1)
    if (parenth) {
      string cline(argv[1]);
      for(int i=2; i<argc; ) {
        string as(argv[i++]);
        if (as.find(' ')==string::npos) cline+=' '+as;
        else cline+=" \""+as+"\"";
      }
      string2atoms(cline, optionsList);
    }
    else
      while(--argc) optionsList.push_back(*(++argv));

  defile(optionsList);
  process(optionsList);

  if (repUnrec) reportUnrecognized();
}


/*  A constructor which is given a list of possible options. Arguments are given in as a string which
    is processed (divided into atoms) by constructor. If rupUnrec=true, unrecognized options are reported. */
TProgArguments::TProgArguments(const string &poss_options, const string &line, bool repUnrec, bool anallowSpaces)
 : possibleOptions(), options(), unrecognized(), direct(), allowSpaces(anallowSpaces)
{ findPossibleOptions(poss_options);
  vector<string> optionsList;
  string2atoms(line, optionsList);
  defile(optionsList);
  process(optionsList);

  if (repUnrec) reportUnrecognized();
}


/*  Processes the string which specifies possible options (the first parameter to constructors)
    and stores the results into possibleOptions field. poss_options consists of options' names, divided
    by blanks. If option's name is immediately followed by a colon, option has parameters. For example,
    string 'help f: m: u' specifies options 'help' and 'u' which have no parameters and 'f' and 'm' which
    have (one) parameter. */
void TProgArguments::findPossibleOptions(const string &poss_options)
{ 
  string ops;
  string::const_iterator chi=poss_options.begin();
  for(;;) {
    if (chi==poss_options.end()) {
      possibleOptions[ops] = false;
      break;
    }

    else if ((*chi==':') || (*chi==' ')) {
      char &po=possibleOptions[ops];
      po=(*chi==':') ? 1 : 0;
      while((++chi!=poss_options.end()) && (*chi==':')) po++;
      while((chi!=poss_options.end()) && (*chi==' ')) chi++;
      if (chi==poss_options.end()) break;
      ops= *(chi++);
    } 

    else ops+= *(chi++);
  }
}


/* Processes the list of options (as rewritten from argv or line parameter to the constructor). */
void TProgArguments::process(const vector<string> &optionsList)
{ for(vector<string>::const_iterator si(optionsList.begin()), se(optionsList.end()); si!=se; )
    if ((*si)[0]=='-') {
      string option((*(si++)).c_str()+1);
      if (possibleOptions.find(option)==possibleOptions.end())
         unrecognized.insert(pair<string, string>(option, ((si==optionsList.end()) || ((*si)[0]=='-')) ? string("") : *(si++)));
      else if (possibleOptions[option])
        if (si==optionsList.end())
          raiseError("missing parameter for option '%s'", option.c_str());
        else
          options.insert(pair<string, string>(option, *(si++)));  // for now, it takes only one option
      else options.insert(pair<string, string>(option, ""));
    }
    else {
      string::const_iterator ei((*si).begin()), ee((*si).end());
      for(; (ei!=ee) && (*ei!='='); ei++);
      if (ei==ee) direct.push_back(*(si++));
      else {
        string option((*si).begin(), ei), par(ei+1, (*si).end());
		if (allowSpaces) {
			while((++si != se) && (si->find("=") == string::npos)) {
				par += " "+*si;
			}
		}
		else {
			si++;
		}
        if (possibleOptions.find(option)==possibleOptions.end())
          unrecognized.insert(pair<string, string>(option, par));
        else if (possibleOptions[option]) options.insert(pair<string, string>(option, par));
        else 
          raiseError("option '%s' expects no arguments", option.c_str());
      }
    }
}


// Outputs the unrecognized arguments (the 'unrecognized' field)
void TProgArguments::reportUnrecognized() const
{ if (unrecognized.size())
    raiseError("unrecognized option '%s'", (*unrecognized.begin()).first.c_str());
}


// Returns 'true' if argument par is in 'options' list
bool TProgArguments::exists(const string &par) const
{ return (options.find(par)!=options.end()); }


// Returns the first value of the given parameter (there can be other...)
string TProgArguments::operator[](const string &par) const
{ TMultiStringParameters::const_iterator oi=options.find(par);
  if (oi==options.end())
    raiseError("parameter '%s' not found", par.c_str());
  return (*oi).second;
}



bool readAnAtom(char *&curr, string &atom)
{
  for(;*curr && (*curr<=' ');curr++); // skip whitespace
  if (!*curr) return false;

  char *start=curr;

  if (*curr=='"') {
    for(curr++; (*curr!='"') && (*curr!=10) && (*curr!=13)  && *curr; curr++);
    atom=string(++start, curr);
    if (*curr++!='"')
      raiseErrorWho("string2atoms", "newline in string '%s'", atom.c_str());
  }
  else if (*curr=='(') {
    int parnths=1;
    for(curr++; parnths && *curr && (*curr!=10) && (*curr!=13); curr++)
      if (*curr=='(') parnths++;
      else if (*curr==')') parnths--;
    if (parnths)
      raiseErrorWho("string2atoms", "to many ('s in '%s'", string(start, curr).c_str());

    atom=string(++start, (curr++)-1);
  }
  else {
    while(*(++curr)>' ');
    atom=string(start, curr);
  }
  return true;
}


/*  Converts a string to a list of atoms. Atoms are separated by spaces and/or tabs while string
    can also be finished by null, linefeed or cariage return characters. */
int string2atoms(const string &line, vector<string> &atoms)
{
  if ((line[0]=='"') && (line[line.length()-1]=='"')) {
    char buff[1024], *curr=buff;
    
    for(int i=1, llen = line.length(); i+1<llen;)
      if (line[i]=='"') {
        *(curr++)='"';
        if (line[++i]=='"') i++; // skips if there's another
      }
      else *(curr++)=line[i++];
    *curr=0;
    curr=buff;

    string atom;
    while(readAnAtom(curr, atom))
      atoms.push_back(atom);
  }
  else {
    char cline[1024];
    strcpy(cline, line.c_str());
    char *curr=cline;
    string atom;
    while(readAnAtom(curr, atom))
      atoms.push_back(atom);
  }

  return atoms.size();
}     


string firstAtom(const string &line)
{ string first, others;
  firstAndOthers(line, first, others);
  return first;
}

string butFirstAtom(const string &line)
{ string first, others;
  firstAndOthers(line, first, others);
  return others;
}


void firstAndOthers(const string &line, string &first, string &others)
{ // skip whitespace before first
  string::const_iterator curr(line.begin()), cue(line.end());
  for(; (curr!=cue) && (*curr<=' '); curr++);

  // find the end of first
  string::const_iterator fbeg=curr;
  for(; (curr!=cue) && (*curr>' '); curr++);
  first=string(fbeg, curr);

  // skip the whitespace before others
  for(; (curr!=cue) && (*curr<=' '); curr++);
  others=string(curr, cue);
}


string getSLine(istream &str)
{ char line[1024];
  str.getline(line, 1024);
  if (str.gcount()==1024-1)
    raiseError("line too long");
  return line;
}


void defile(vector<string> &options)
{ bool changed;
  do {
    changed=false;
    vector<string> reoptions=options;
    options.clear();
    ITERATE(vector<string>, oi, reoptions) {
      if ((*oi)[0]=='@') {
        string::iterator be((*oi).end());
        do --be; while ((*be!=':') && (*be!='@'));
        if (*be=='@') {
          ifstream pstr(string(be+1, (*oi).end()).c_str());
          if (!pstr.is_open() || pstr.fail() || pstr.eof())
            raiseError("invalid parameter file");
          while (!pstr.fail() && !pstr.eof()) {
            string nl=getSLine(pstr);
            if (nl.length()) options.push_back(string(nl.begin(), nl.end()));
          }
        }
        else {
          string filename((*oi).begin()+1, be), lineNos(be+1, (*oi).end());
          int lineNo=atoi(lineNos.c_str());
          if (!lineNo)
            raiseError("Invalid line number (%s) for parameter file %s.", lineNos.c_str(), filename.c_str());
          ifstream pstr(filename.c_str());
          if (!pstr.is_open() || pstr.fail() || pstr.eof())
            raiseError("Invalid parameter file (%s).", (*oi).c_str());
          while(--lineNo) {
            getSLine(pstr);
            if (pstr.fail() || pstr.eof())
              raiseError("can't read parameter file %s to line %s", filename.c_str(), lineNos.c_str());
          }
          string nl=getSLine(pstr);
          if (nl[0]=='=') options.push_back(string(nl.begin()+1, nl.end()));
          else {
            vector<string> ns;
            string2atoms(nl, ns);
            ITERATE(vector<string>, ni, ns) options.push_back(*ni);
          }
        }
        changed=true;
      }
      else options.push_back(*oi);
    }
  } while (changed);
}


