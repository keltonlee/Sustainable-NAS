/*
 * helper_funtions.c
 * misc helper functions
 */

#include "helper_functions.h"


// infinite loop
void wait_forever(void){
    while(1){__no_operation();}
}

// swap two buffer pointers
void swap(uint8_t **buff1_ptr, uint8_t **buff2_ptr){
  uint8_t *temp_ptr = *buff1_ptr;
  *buff1_ptr = *buff2_ptr;
  *buff2_ptr = temp_ptr;
}

// convert integer to ascii char string
void itoa(long unsigned int value, char* result, int base)   {
  // check that the base if valid
  if (base < 2 || base > 36) { *result = '\0';}

  char* ptr = result, *ptr1 = result, tmp_char;
  int tmp_value;

  do {
    tmp_value = value;
    value /= base;
    *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
  } while ( value );

  // Apply negative sign
  if (tmp_value < 0) *ptr++ = '-';
  *ptr-- = '\0';
  while(ptr1 < ptr) {
    tmp_char = *ptr;
    *ptr--= *ptr1;
    *ptr1++ = tmp_char;
  }

}
