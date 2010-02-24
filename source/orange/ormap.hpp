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


/* _ClassName should be TOrangeMap_, TOrangeMap_K, TOrangeMap_V or TOrangeMap_KV.
   K and V stand for key and value, and should be present if the key (value) is
   an Orange object. */
#ifndef __ORMAP_HPP
#define __ORMAP_HPP


#ifdef _MSC_VER
  #define DEFINE_TOrangeMap_classDescription(_ClassName,_Key,_Value, _NAME) \
    TClassDescription _ClassName<_Key,_Value>::st_classDescription = { _NAME, &typeid(_ClassName<_Key,_Value>), &TOrange::st_classDescription, TOrange_properties, TOrange_components };
#else
  #define DEFINE_TOrangeMap_classDescription(_ClassName,_Key,_Value, _NAME) \
    template <> TClassDescription _ClassName<_Key,_Value>::st_classDescription = { _NAME, &typeid(_ClassName<_Key,_Value>), &TOrange::st_classDescription, TOrange_properties, TOrange_components };
#endif


#include <map>
#include "root.hpp"
#include "stladdon.hpp"

template<class K, class V>
class TOrangeMap : public TOrange
{ public:
    MAP_INTERFACE(K, V, __ormap, typedef typename);

    TOrangeMap()
      {}
 
    TOrangeMap(const map<K, V>& X)
      : __ormap(X)
      {}

    int dropReferences()
    { DROPREFERENCES(TOrange::dropReferences);
      clear();
      return 0;
    }
};


template<class K, class V>
class TOrangeMap_ : public TOrangeMap<K, V>
{ public:
    static TClassDescription st_classDescription;

    virtual TClassDescription const *classDescription() const
      { return &st_classDescription; }
};


template<class K, class V>
class TOrangeMap_K : public TOrangeMap<K, V>
{ public:
    int traverse(visitproc visit, void *arg) const
    { TRAVERSE(TOrange::traverse);
      for(typename TOrangeMap<K, V>::const_iterator be=begin(), ee=end(); be!=ee; be++)
        PVISIT((*be).first);
      return 0;
    }

    static TClassDescription st_classDescription;

    virtual TClassDescription const *classDescription() const
      { return &st_classDescription; }
};


template<class K, class V>
class TOrangeMap_V : public TOrangeMap<K, V>
{ public:
    int traverse(visitproc visit, void *arg) const
    { TRAVERSE(TOrange::traverse);
      for(typename TOrangeMap<K,V>::const_iterator be=begin(), ee=end(); be!=ee; be++)
        if (this->value_is_orange)
          PVISIT((*be).second);
      return 0;
    }

    static TClassDescription st_classDescription;

    virtual TClassDescription const *classDescription() const
      { return &st_classDescription; }
};


template<class K, class V>
class TOrangeMap_KV : public TOrangeMap<K, V>
{ public:
    MAP_INTERFACE(K, V, __ormap, typedef typename);

    int traverse(visitproc visit, void *arg) const
    { TRAVERSE(TOrange::traverse);
      for(const_iterator be=begin(), ee=end(); be!=ee; be++) {
        PVISIT((*be).first);
        PVISIT((*be).second);
      }
      return 0;
    }

    static TClassDescription st_classDescription;

    virtual TClassDescription const *classDescription() const
      { return &st_classDescription; }
};

#define MWRAPPER(x) typedef GCPtr< T##x > P##x;


#endif
