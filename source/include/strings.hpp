#ifndef __STRINGS_HPP
#define __STRINGS_HPP

#include <string>
#include <string.h>
#include <vector>

using namespace std;

string trim(const string &s);
void trim(char *s);

typedef vector<pair<string::const_iterator, string::const_iterator> > TSplits;

void split(const string &s, TSplits &atoms);

#endif
