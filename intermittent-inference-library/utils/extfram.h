#ifndef JAPARILIB_EXTFRAM_H_
#define JAPARILIB_EXTFRAM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
	
#define EXTFRAM
#define FRAM_8Mb

#ifdef FRAM_16Mb
	#define EXTFRAM_SIZE  0x200000
#elif defined(FRAM_8Mb)
	#define EXTFRAM_SIZE  0x100000
#else
	#define EXTFRAM_SIZE  0x80000
#endif

#define EXTFRAM_USE_DMA

extern uint8_t controlTable[1024] __attribute__((aligned (1024)));
extern uint32_t curDMATransmitChannelNum, curDMAReceiveChannelNum;

typedef union Data {
    unsigned char byte[4];
    unsigned long L;
}SPI_ADDR;

void eraseFRAM(void);
int initSPI(void);
void SPI_READ(SPI_ADDR* A,uint8_t *dst, unsigned long len );
void SPI_WRITE(SPI_ADDR* A, const uint8_t *src, unsigned long len );
void SPI_FILL_Q15(SPI_ADDR* A, int16_t val, unsigned long len );
uint8_t testSPI (void);

#ifdef __cplusplus
}
#endif

#endif /* JAPARILIB_EXTFRAM_H_ */
