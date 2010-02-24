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

#ifndef __CRC_H
#define __CRC_H

extern unsigned long crc_table[256];

#define INIT_CRC(x) (x) = 0xffffffff
#define FINISH_CRC(x) (x) = (x) ^ 0xffffffff

#define ADD_CRC \
  for(unsigned char const *b = (unsigned char const *)(&data), *e = b + sizeof(data); \
      b != e; \
      crc = (crc >> 8) ^ crc_table[(crc & 0xFF) ^ *(b++)]); \

inline void add_CRC(const unsigned long data, unsigned long &crc)
{ ADD_CRC }

inline void add_CRC(const float data, unsigned long &crc)
{ ADD_CRC }

inline void add_CRC(const unsigned char c, unsigned long &crc)
{ crc = (crc >> 8) ^ crc_table[(crc & 0xFF) ^ c]; }

inline void add_CRC(const char *c, unsigned long &crc)
{
  for(; *c; add_CRC((unsigned char)*c++, crc));
  add_CRC((unsigned char)0, crc);
}

#endif
