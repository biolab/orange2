extern unsigned long crc32_table[256];

#define INIT_CRC(x) (x) = 0xffffffff
#define FINISH_CRC(x) (x) = (x) ^ 0xffffffff

inline void add_CRC4(unsigned char *buffer, unsigned long &crc) 
{ 
  crc = (crc >> 8) ^ crc32_table[(crc & 0xFF) ^ *(buffer++)]; 
  crc = (crc >> 8) ^ crc32_table[(crc & 0xFF) ^ *(buffer++)]; 
  crc = (crc >> 8) ^ crc32_table[(crc & 0xFF) ^ *(buffer++)]; 
  crc = (crc >> 8) ^ crc32_table[(crc & 0xFF) ^ *buffer]; 
}

inline void add_CRC(unsigned long &data, unsigned long &crc) 
{ add_CRC4((unsigned char*)(&data), crc); }

inline void add_CRC(float &data, unsigned long &crc) 
{ add_CRC4((unsigned char *)(&data), crc); }

inline void add_CRC(unsigned char c, unsigned long &crc)
{ crc = (crc >> 8) ^ crc32_table[(crc & 0xFF) ^ c]; }
