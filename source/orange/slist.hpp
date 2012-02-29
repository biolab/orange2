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

#include <vector>
#include "c2py.hpp"

class TCharBuffer {
public:
  char *buf, *bufe;
  char *bufptr;

  TCharBuffer(const int &size)
  {
    if (size) {
      buf = bufptr = (char *)malloc(size);
      bufe = buf + size;
    }
    else
      buf = bufptr = bufe = NULL;
  }


  TCharBuffer(char *abuf)
  : buf(abuf),
    bufe(NULL),
    bufptr(abuf)
  {}


  ~TCharBuffer()
  {
    if (buf && bufe) // if there's no bufe, we don't own the buffer
      free(buf);
  }

  inline Py_ssize_t length()
  { return bufptr - buf; }

  inline void ensure(const Py_ssize_t &size)
  { 
    if (!buf) {
       Py_ssize_t rsize = size > 1024 ? size : 1024;
       buf = bufptr = (char *)malloc(rsize);
       bufe = buf + rsize;
    }

    else if (bufe - bufptr < size) {
       int tsize = bufe - buf;
       tsize = tsize < 65536 ? tsize << 1 : tsize + 65536;
       const int tpos = bufptr - buf;
       buf = (char *)realloc(buf, tsize);
       bufe = buf + tsize;
       bufptr = buf + tpos;
    }
  }

  inline void writeChar(const char &c)
  {
    ensure(sizeof(char));
    *bufptr++ = c;
  }

  inline void writeShort(const unsigned short &c)
  {
    ensure(sizeof(unsigned short));
    memcpy((void *) bufptr, (void *) &c, sizeof(unsigned short));
    bufptr += sizeof(unsigned short);
  }

  inline void writeInt(const int &c)
  {
    ensure(sizeof(int));
    memcpy((void *) bufptr, (void *) &c, sizeof(int));
    bufptr += sizeof(int);
  }

  inline void writeLong(const long &c)
  {
    ensure(sizeof(long));
    memcpy((void *) bufptr, (void *) &c, sizeof(long));
    bufptr += sizeof(long);
  }

  inline void writeFloat(const float &c)
  {
    ensure(sizeof(float));
    memcpy((void *) bufptr, (void *) &c, sizeof(float));
    bufptr += sizeof(float);
  }

  inline void writeDouble(const double &c)
  {
    ensure(sizeof(double));
    memcpy((void *) bufptr, (void *) &c, sizeof(float));
    bufptr += sizeof(double);
  }

  inline void writeIntVector(const vector<int> &v)
  {
    int size = v.size();
    ensure((size + 1) * sizeof(int));

    memcpy((void *) bufptr, (void *)&size, sizeof(int));
    bufptr += sizeof(int);
    if (size > 0)
    {
        // This is legal as &v[0] is guaranteed to point to a
        // contiguous memory block
        memcpy((void *) bufptr, (void *) &v[0], size * sizeof(int));
        bufptr += sizeof(int) * size;
    }
  }

  inline void writeFloatVector(const vector<float> &v)
  {
    int size = v.size();
    ensure(sizeof(int) + size * sizeof(float));

    memcpy((void *) bufptr, (void *)&size, sizeof(int));
    bufptr += sizeof(int);
    if (size > 0)
    {
        // This is legal as &v[0] is guaranteed to point to a
        // contiguous memory block
        memcpy((void *) bufptr, (void *) &v[0], size * sizeof(float));
        bufptr += sizeof(float) * size;
    }
  }

  inline void writeBuf(const void *abuf, size_t size)
  {
    ensure(size);
    memcpy(bufptr, abuf, size);
    bufptr += size;
  }

  inline char readChar()
  { 
    return *bufptr++;
  }

  inline unsigned short readShort()
  {
    unsigned short &res = (unsigned short &)*bufptr;
    bufptr += sizeof(short);
    return res;
  }

  inline int readInt()
  { 
    int &res = (int &)*bufptr;
    bufptr += sizeof(int);
    return res;
  }

  inline long readLong()
  { 
    long &res = (long &)*bufptr;
    bufptr += sizeof(long);
    return res;
  }

  inline float readFloat()
  {
    float &res = (float &)*bufptr;
    bufptr += sizeof(float);
    return res;
  }

  inline double readDouble()
  {
    double &res = (double &)*bufptr;
    bufptr += sizeof(double);
    return res;
  }

  inline void readIntVector(vector<int> &v)
  {
    int *&buff = (int *&)bufptr;
    int size = *buff++;
    v.resize(size);

    for(vector<int>::iterator vi = v.begin(); size--; *vi++ = *buff++);
  }

  inline void readFloatVector(vector<float> &v)
  {
    int size = readInt();
    v.resize(size);

    float *&buff = (float *&)bufptr;
    for(vector<float>::iterator vi = v.begin(); size--; *vi++ = *buff++);
  }

  inline void readBuf(void *abuf, size_t size)
  {
    memcpy(abuf, bufptr, size);
    bufptr += size;
  }
};


#endif
