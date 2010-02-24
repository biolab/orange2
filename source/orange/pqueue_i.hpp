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


#ifndef __PQUEUE_I_HPP
#define __PQUEUE_I_HPP

#include "random.hpp"

template<class TPQNode>
class TPriorityQueue : public vector<TPQNode *> {
public:
  ~TPriorityQueue()
    { typedef typename vector<TPQNode *>::iterator iterator;
      for(iterator ii(begin()); ii!=end(); mldelete *(ii++)); 
    }

  void sink(int i)
    { TPQNode *sank=at(i);

      int msize = size();
      for(int newi = 2*i+1; newi < msize; newi = 2*(i=newi)+1) {
        if (newi+1<msize) {
          int cmp = at(newi)->compare(*at(newi+1));
          if (cmp<0)
            newi++;
        }

        int cmp = at(newi)->compare(*sank);
        if (cmp>0)
          (at(i) = at(newi))->queueIndex = i;
        else 
          break;
      }

      (operator[](i) = sank)->queueIndex = i;
    }


  void insert(TPQNode *node)
    { push_back((TPQNode *)NULL);
      int down = size()-1;
      for(int up; down; down=up) {
        up = (down-1)/2;
        int cmp=node->compare(*at(up));
        if (cmp>0)
          (at(down) = at(up))->queueIndex = down;
        else
          break;
      }

      (at(down) = node)->queueIndex = down;
    }


  void remove(int oldi)
    { mldelete at(oldi);
      if (oldi == int(size()-1)) {
        at(oldi) = NULL;
        erase(end()-1);
      }
      else {
        (at(oldi) = back())->queueIndex = oldi;
        back() = NULL;
        erase(end()-1);
        sink(oldi);
      }
    }
};

#endif

