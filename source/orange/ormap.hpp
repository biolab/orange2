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


#define DEFINE_TOrangeMap_classDescription(_Key,_Value,_Kior,_Vior, _NAME) \
  TClassDescription TOrangeMap<_Key,_Value,_Kior,_Vior>::st_classDescription = { _NAME, &typeid(TOrangeMap<_Key,_Value,_Kior,_Vior>), &TOrange::st_classDescription, TOrange_properties, TOrange_components };


#ifndef __ORMAP_HPP
#define __ORMAP_HPP

#include "garbage.hpp"
#include <map>
#include "root.hpp"
#include "stladdon.hpp"

template<class K, class V, bool key_is_orange, bool value_is_orange>
class TOrangeMap : public TOrange
{ public:
    MAP_INTERFACE(K, V, __ormap);

    TOrangeMap()
      {}
 
    TOrangeMap(const map<K, V>& X)
      : __ormap(X)
      {}

    int traverse(visitproc visit, void *arg) const
    { TRAVERSE(TOrange::traverse);
      if (key_is_orange || value_is_orange)
        for(const_iterator be=begin(), ee=end(); be!=ee; be++) {
          if (key_is_orange)
            PVISIT((*be).first);
          if (value_is_orange)
            PVISIT((*be).first);
        }
      return 0;
    }

    int dropReferences()
    { DROPREFERENCES(TOrange::dropReferences);
      clear();
      return 0;
    }

    static TClassDescription st_classDescription;

    virtual TClassDescription const *classDescription() const
      { return &st_classDescription; }
};


#define MWRAPPER(x) typedef GCPtr< T##x > P##x;


#endif