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


#ifndef __TABDELIM_HPP
#define __TABDELIM_HPP

#include <string>

#include "filegen.hpp"
#include "domain.hpp"
#include "domaindepot.hpp"
#include "basket.hpp"

using namespace std;

/*  A descendant from TFileExampleGenerator, specialized for tab delimited files.
    File starts with a line containing names of attributes, the second line describes their types.
    and the third can contain 'ignore' (or 'i', 'skip', 's') if attribute is to be ignored or
    'class' (or 'c') if it is class attribute - one column must contain 'class' attribute.
    Third line can also have additional options for the attribute; at the moment, the only option
    is -dc which specifies the values which are treated as don't care; -dc 03k means that 0, 3 and k
    are don't care values.
    Type can be 'continuous' ('c', 'float', 'f') or 'discrete' ('d', 'enum', 'e'). For discrete
    attributes, the values of attribute can be listed instead of keyword. 
    The rest of file are examples, one line for each, containing values of attributes. '?' is interpreted
    as unknown.
    Names of attributes and their values are atoms. Atom consists of any characters
    except \n, \r and \t. Multiple spaces are replaced by a single space. Atoms are separated
    by \t. Lines end with \n or \r. Lines which begin with | are ignored.
    Here is an example of a tab delimited file.

    age         weight       sex         car
    continuous  continuous   M, F        discrete
                ignore                   class
    | an unnecessary comment
    25          85           F           polo
    85          66           M           A4
    14          45           M           no
    ?           ?            F           polo
*/
class ORANGE_API TTabDelimExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS

/*  A kind of each attribute:
            1   pending meta value (used only at construction time)
           -1   normal
            0   skipped
          <-1   meta value. */
  PIntList attributeTypes; //P types of attributes (-1 normal, 0 skip, <-1 = meta ID)

  char *DK; //P general character that denotes DK
  char *DC; //P general character that denotes DC
  int classPos; //P position of the class attribute
  PIntList classPoses; //P positions of class attributes if there are multiple; otherwise None
  int basketPos; //P position of the (virtual) basket attribute
  int headerLines; //P number of header lines (3 for .tab, 1 for .txt)
  bool csv; //P also allow ',' as a separator

  PBasketFeeder basketFeeder; //P takes care of inserting the meta attributes from the basket if need be

  typedef struct {
    char *identifier;
    int matchRoot;
    int varType;
  } TIdentifierDeclaration;

  static const TIdentifierDeclaration typeIdentifiers[] ;

  TTabDelimExampleGenerator::TTabDelimExampleGenerator(const TTabDelimExampleGenerator &old);
  TTabDelimExampleGenerator(const string &, bool autoDetect, bool csv, 
                            const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus,
                            const char *aDK = NULL, const char *aDC = NULL, bool noCodedDiscrete = false, bool noClass = false);
  ~TTabDelimExampleGenerator();

  virtual bool readExample (TFileExampleIteratorData &, TExample &);

  void atomList2Example(vector<string> &atoms, TExample &exam, const TFileExampleIteratorData &fei);

  char *mayBeTabFile(const string &stem);
  PDomain readDomain(const string &stem, const bool autoDetect, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus, bool noCodedDiscrete, bool noClass);
  void readTxtHeader(const string &stem, TDomainDepot::TAttributeDescriptions &);
  void readTabHeader(const string &stem, TDomainDepot::TAttributeDescriptions &);
  int detectAttributeType(TDomainDepot::TAttributeDescription &desc, bool noCodedDiscrete);
  void scanAttributeValues(const string &stem, TDomainDepot::TAttributeDescriptions &desc);
};


#endif

