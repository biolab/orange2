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


#ifndef __SLIST_HPP
#define __SLIST_HPP

template<class T>
class slist {
public:
  T *node;
  slist<T> *prev, *next;

  slist(T *anode = NULL, slist<T> *aprev =NULL)
   : node(anode), prev(aprev), next(aprev ? aprev->next : NULL)
   { if (prev) prev->next=this;
     if (next) next->prev=this; }


  ~slist()
  { if (prev) prev->next=next;
    if (next) next->prev=prev;
  }
};

#endif
