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
      for(iterator ii(this->begin()); ii!=this->end(); mldelete *(ii++)); 
    }

  void sink(int i)
    { TPQNode *sank=this->at(i);

      int msize = this->size();
      for(int newi = 2*i+1; newi < msize; newi = 2*(i=newi)+1) {
        if (newi+1<msize) {
          int cmp = this->at(newi)->compare(*this->at(newi+1));
          if (cmp<0)
            newi++;
        }

        int cmp = this->at(newi)->compare(*sank);
        if (cmp>0)
          (this->at(i) = this->at(newi))->queueIndex = i;
        else 
          break;
      }

      (this->operator[](i) = sank)->queueIndex = i;
    }


  void insert(TPQNode *node)
    { this->push_back((TPQNode *)NULL);
      int down = this->size()-1;
      for(int up; down; down=up) {
        up = (down-1)/2;
        int cmp=node->compare(*this->at(up));
        if (cmp>0)
          (this->at(down) = this->at(up))->queueIndex = down;
        else
          break;
      }

      (this->at(down) = node)->queueIndex = down;
    }


  void remove(int oldi)
    { mldelete this->at(oldi);
      if (oldi == int(this->size()-1)) {
        this->at(oldi) = NULL;
        this->erase(this->end()-1);
      }
      else {
        (this->at(oldi) = this->back())->queueIndex = oldi;
        this->back() = NULL;
        this->erase(this->end()-1);
        this->sink(oldi);
      }
    }
};

#endif

