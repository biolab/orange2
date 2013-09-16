#ifndef __CRC_H
#define __CRC_H

extern unsigned int crc_table[256];

#define INIT_CRC(x) (x) = 0xffffffff
#define FINISH_CRC(x) (x) = (x) ^ 0xffffffff

#define ADD_CRC \
  for(unsigned char const *b = (unsigned char const *)(&data), *e = b + sizeof(data); \
      b != e; \
      crc = (crc >> 8) ^ crc_table[(crc & 0xFF) ^ *(b++)]); \

inline void add_CRC(const int data, unsigned int &crc)
{ ADD_CRC }

inline void add_CRC(const unsigned long data, unsigned int &crc)
{ ADD_CRC }

inline void add_CRC(const float data, unsigned int &crc)
{ ADD_CRC }

inline void add_CRC(const unsigned char c, unsigned int &crc)
{ crc = (crc >> 8) ^ crc_table[(crc & 0xFF) ^ c]; }

inline void add_CRC(const char *c, unsigned int &crc)
{
  for(; *c; add_CRC((unsigned char)*c++, crc));
  add_CRC((unsigned char)0, crc);
}

#endif
