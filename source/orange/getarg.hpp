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

