extern unsigned long crc_table[256];

#define INIT_CRC(x) (x) = 0xffffffff
#define FINISH_CRC(x) (x) = (x) ^ 0xffffffff

#define ADD_CRC \
  for(unsigned char *b = (unsigned char *)(&data), *e = b + sizeof(data); \
      b != e; \
      crc = (crc >> 8) ^ crc_table[(crc & 0xFF) ^ *(b++)]); \

inline void add_CRC(unsigned long &data, unsigned long &crc) 
{ ADD_CRC }

inline void add_CRC(float &data, unsigned long &crc) 
{ ADD_CRC }

inline void add_CRC(unsigned char c, unsigned long &crc)
{ crc = (crc >> 8) ^ crc_table[(crc & 0xFF) ^ c]; }
