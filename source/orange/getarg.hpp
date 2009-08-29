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


#ifndef __GETARG_HPP
#define __GETARG_HPP

#include <map>
#include <vector>
#include <string>

using namespace std;

#ifdef _MSC_VER
 #pragma warning (disable : 4786 4114 4018 4267)
#endif

typedef map<string, string> TStringParameters;
typedef multimap<string, string> TMultiStringParameters;

// An object which inteprets and stores arguments from command line.
class TProgArguments {
public:
  // A list of possible options. The bool field specifies whether option has an additional argument or not. */
  map<string, char> possibleOptions;

  // A map of found options (with arguments, when they were expected)
  TMultiStringParameters options;
  // A list of unrecognized options
  TMultiStringParameters unrecognized;
  // A list of parameters which were given without leading -.
  vector<string> direct;
  bool allowSpaces;

  TProgArguments();
  TProgArguments(const string &, int argc, char *argv[], bool repUnrec=true, bool parenth=true, bool allowSpaces=false);
  TProgArguments(const string &poss_options, const string &line, bool repUnrec=true, bool allowSpaces=false);

  void findPossibleOptions(const string &poss_options);
  void process(const vector<string> &optionsList);

  void reportUnrecognized() const;
  bool exists(const string &par) const;
  string operator[](const string &) const;
};


int string2atoms(const string &line, vector<string> &atoms);
void defile(vector<string> &options);

string firstAtom(const string &line);
string butFirstAtom(const string &line);
void firstAndOthers(const string &line, string &first, string &others);

#endif

