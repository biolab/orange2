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


#ifndef __DOMAINDEPOT_HPP
#define __DOMAINDEPOT_HPP

#include <vector>
#include <list>
using namespace std;

VWRAPPER(StringList)
VWRAPPER(VarList)
WRAPPER(Domain)
class TMetaVector;

#define TDomainList TOrangeVector<PDomain> 
VWRAPPER(DomainList)

// For each attribute, the correspongin element of multimapping gives
//   domains and position in domains in which the attribute appears
typedef vector<vector<pair<int, int> > > TDomainMultiMapping;


class ORANGE_API TDomainDepot
{
/* XXX Domain depot has no wrapped pointers to anything, thus it doesn't own any references.
       If you add any, you should implement DomainDepot_traverse and DomainDepot_clear in 
       Python interface. */
public:
  class TAttributeDescription {
  public:
    string name;
    int varType;
    string typeDeclaration;
    bool ordered;
    PStringList values; // not always used, but often comes handy...

    TAttributeDescription(const string &, const int &, const string &, bool = false);
    TAttributeDescription(const string &, const int &);
  };

  ~TDomainDepot();

  typedef vector<TAttributeDescription> TAttributeDescriptions;

  static bool checkDomain(const TDomain *, const TAttributeDescriptions *attributes, bool hasClass,
                          const TAttributeDescriptions *metas, int *metaIDs = NULL);

  PDomain prepareDomain(const TAttributeDescriptions *attributes, bool hasClass,
                        const TAttributeDescriptions *metas, PVarList knownVars, const TMetaVector *knownMetas,
                        const bool dontStore, const bool dontCheckStored,
                        bool *domainIsNew = NULL, int *metaIDs = NULL);

  static void destroyNotifier(TDomain *domain, void *);

  /* Creates a variable with given name and type. */
  static PVariable createVariable(const TAttributeDescription &);
  static PVariable createVariable_Python(const string &typeDeclaration, const string &name);

  /* Tries to find a variable the given name and type in knownVars or metaVector.
     Any of these (or both) can be omitted. If the variable is found in metaVector,
     the id is set as well; if not, id is set to 0. If the variable is not found,
     a new one is created unless dontCreateNew is set to false. */
  static PVariable makeVariable(const TAttributeDescription &, int &id, PVarList knownVars, const TMetaVector * = NULL, bool dontCreateNew = false, bool preferMetas = false);


private:
  list<TDomain *> knownDomains;
};





PDomain combineDomains(PDomainList sources, TDomainMultiMapping &mapping);
void computeMapping(PDomain destination, PDomainList sources, TDomainMultiMapping &mapping);


#endif
