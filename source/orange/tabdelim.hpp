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


#ifndef __TABDELIM_HPP
#define __TABDELIM_HPP

#include <string>

#include "filegen.hpp"
#include "domain.hpp"
#include "domaindepot.hpp"

using namespace std;

typedef vector<string> TIdList;

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
class TTabDelimExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS

/*  A kind of each attribute:
            1   pending meta value (used only at construction time)
           -1   normal
            0   skipped
          <-1   meta value. */
  PIntList attributeTypes; //P types of attributes (-1 normal, 0 skip, <-1 = meta ID)

  PStringList DCs; //P characters that mean DC (for each attribute)
  int classPos; //P position of the class attribute
  int headerLines; //P number of header lines (3 for .tab, 1 for .txt)
  bool csv; //P also allow ',' as a separator

  TTabDelimExampleGenerator::TTabDelimExampleGenerator(const TTabDelimExampleGenerator &old);
  TTabDelimExampleGenerator(const string &, bool autoDetect, bool csv, PVarList sourceVars = PVarList(), TMetaVector *sourceMetas = NULL, PDomain sourceDomain = PDomain(), bool dontCheckStored = false, bool dontStore = false);
  virtual bool readExample (TFileExampleIteratorData &, TExample &);

  void atomList2Example(TIdList &atoms, TExample &exam, const TFileExampleIteratorData &fei);

  char *mayBeTabFile(const string &stem);
  PDomain readDomain(const string &stem, const bool autoDetect, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored, bool dontStore);
  PDomain domainWithDetection(const string &stem, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored);
  PDomain domainWithoutDetection(const string &stem, PVarList sourceVars, TMetaVector *sourceMetas, PDomain sourceDomain, bool dontCheckStored);

private:
  static TDomainDepot domainDepot_tab, domainDepot_txt;
};


#endif

