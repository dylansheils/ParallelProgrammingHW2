/*********************************************************************/
/* cla.h ************************************************************/
/*********************************************************************/
//
// 01/14/2022: Revised Version that fixes buffer overruns & mem leaks
// Used Valgrind
//
/*********************************************************************/

#ifndef CLA_MAIN_C_H
#define CLA_MAIN_C_H

/*********************************************************************/
/* Include File ******************************************************/
/*********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "clockcycle.h"

/*********************************************************************/
/* Forward Inline Functions ******************************************/
/*********************************************************************/

int rand_in_range(int min, int max);
char hex_lookup(int in);
char* generate_random_hex(size_t len);
char* hex_to_binary_single(char theDigit);
char* hex_to_binary(char* hexString);
void reverse_string(char* str, size_t len);
int* grab_slice(int* input, int starti, int length);
char* grab_slice_char(char* input, int starti, int length);
int* string_to_int(char* str);
char* int_to_string(int* in, size_t len);
char binary_quad_to_hex_single(char *binary);
char* revbinary_to_hex(char* binaryLine,int len);
int* gen_formated_binary_from_hex(char *hex);
void print_chararrayln(char *in);

/*********************************************************************/
/* rand_in_range *****************************************************/
/*********************************************************************/

int rand_in_range(int min, int max)
{
    return (rand() % (max - min) + min);
}

/*********************************************************************/
/* hex_lookup ********************************************************/
/*********************************************************************/

char hex_lookup(int in)
{

    if(in < 10)
    {
        char buf[16];
        snprintf(buf, 16, "%i",in);
        return buf[0];
    }
    else
    {
        switch(in)
        {
            case 10: return 'A';
            case 11: return 'B';
            case 12: return 'C';
            case 13: return 'D';
            case 14: return 'E';
            case 15: return 'F';
            default: return '0';
        }
    }
}

/*********************************************************************/
/* generate_random_hex ***********************************************/
/*********************************************************************/

char* generate_random_hex(size_t len)
{
  char* output = (char *)calloc(len+1,sizeof(char));
    output[len] = '\0';
    for(int i = 0; i < len; i++)
    {
        output[i] = hex_lookup(rand_in_range(0,16));
    }
    return output;
}

/*********************************************************************/
/* preprend_non_sig_zero *********************************************/
/*********************************************************************/

char* prepend_non_sig_zero(char* str)
{
    size_t len = strlen(str);
    char* output = (char *)calloc(len+2,sizeof(char));
    output[0] = '0';
    output[len+1] = '\0';
    for(int i = 1; i < len+1; i++)
    {
        output[i] = str[i-1];
    }

    return output;
}

/*********************************************************************/
/* string_to_int *****************************************************/
/*********************************************************************/

int* string_to_int(char* str)
{
    size_t len = strlen(str);

    int* output = (int *)calloc(len+1,sizeof(int));

    for(int i = 0; i < len; i++)
    {
        output[i] = str[i] - '0';
    }

    return output;
}

/*********************************************************************/
/* int_to_string *****************************************************/
/*********************************************************************/

char* int_to_string(int* in, size_t len)
{
  char* output = (char *)calloc(len+1,sizeof(char));

    int buflen = 0;
    for(int i = 0; i < len; i++)
    {
        buflen += sprintf(output+buflen, "%i",in[i]);
    }

    return output;
}

/*********************************************************************/
/* hex_to_binary_single **********************************************/
/*********************************************************************/

char* hex_to_binary_single(char theDigit)
{
    char* binaryVersion;
    switch(theDigit)
    {
        case '0': binaryVersion = "0000"; break;
        case '1': binaryVersion = "0001"; break;
        case '2': binaryVersion = "0010"; break;
        case '3': binaryVersion = "0011"; break;
        case '4': binaryVersion = "0100"; break;
        case '5': binaryVersion = "0101"; break;
        case '6': binaryVersion = "0110"; break;
        case '7': binaryVersion = "0111"; break;
        case '8': binaryVersion = "1000"; break;
        case '9': binaryVersion = "1001"; break;
        case 'A': binaryVersion = "1010"; break;
        case 'B': binaryVersion = "1011"; break;
        case 'C': binaryVersion = "1100"; break;
        case 'D': binaryVersion = "1101"; break;
        case 'E': binaryVersion = "1110"; break;
        case 'F': binaryVersion = "1111"; break;
        default:  binaryVersion = "0000"; break;
    }
    return binaryVersion;
}

/*********************************************************************/
/* binary_quad_to_hex_single *****************************************/
/*********************************************************************/

char binary_quad_to_hex_single(char *binary)
{
    char hexVersion;
    char* buffer = (char *)calloc(5, sizeof(char));
    buffer[4] = '\0';

    long int hexInt = strtol(binary, NULL,2);

    sprintf(buffer,"%lX",hexInt);

    hexVersion = buffer[0];

    free(buffer);
    
    return hexVersion;

}

/*********************************************************************/
/* revbinary_to_hex **************************************************/
/*********************************************************************/

char* revbinary_to_hex(char* binaryLine,int len)
{
    reverse_string(binaryLine, strlen(binaryLine));

    size_t blen = (size_t)len/4 + (len%4);
    char* buffer = (char *)calloc(blen+1,sizeof(char));
    buffer[blen] = '\0';
    int bufferLength = 0;
    for(int j = 0; j < blen; j++)
    {
        int jstart = j*4;
        char* binquad = grab_slice_char(binaryLine, jstart,4);
        char hexDigit = binary_quad_to_hex_single(binquad);
        bufferLength += sprintf(buffer+bufferLength, "%c",hexDigit);
	free(binquad);
    }

    return buffer;
}

/*********************************************************************/
/* hex_to_binary *****************************************************/
/*********************************************************************/

char* hex_to_binary(char* hexString)
{
    size_t num = strlen(hexString);
    char* buffer = (char *)calloc(num*4+1, sizeof(char));
    buffer[num*4] = '\0';
    int len = 0;

    for(int i = 0; i < num; i++)
    {
        len += sprintf(buffer+len, "%s",hex_to_binary_single(hexString[i]));
    }
    return buffer;
}

/*********************************************************************/
/* reverse_string ****************************************************/
/*********************************************************************/

void reverse_string(char* str, size_t len)
{
    size_t i;
    char tmp;
    size_t j;

    for (i = 0, j = len - 1; i < (len / 2); i++, j--) {
        tmp = str[j];
        str[j] = str[i];
        str[i] = tmp;
    }
}

/*********************************************************************/
/* gen_formated_binary_from_hex **************************************/
/*********************************************************************/

int* gen_formated_binary_from_hex(char *hex)
{
    char* binChar = hex_to_binary(hex);
    size_t len = strlen(binChar);
    reverse_string(binChar,len);
    int* bin = string_to_int(binChar);
    free(binChar);
    
    return bin;
}

/*********************************************************************/
/* print_chararrayln *************************************************/
/*********************************************************************/

void print_chararrayln(char *in)
{
    size_t len = strlen(in);
    for(int i = 0; i<len; i++)
    {
        printf("%c",in[i]);
    }
    printf("\n");
}

/*********************************************************************/
/* grab_slice ********************************************************/
/*********************************************************************/

int* grab_slice(int* input, int starti, int length)
{
  int* output = (int *)calloc((size_t)length,sizeof(int));

    int i,j;
    for(i = 0, j = starti; i<length; i++,j++)
    {
        output[i] = input[j];
    }
    return output;
}

/*********************************************************************/
/* grab_slice_char ***************************************************/
/*********************************************************************/

char* grab_slice_char(char* input, int starti, int length)
{
  char* output = (char *)calloc((size_t)length+1,sizeof(char));

    int i,j;
    for(i = 0, j = starti; i<length; i++,j++)
    {
        output[i] = input[j];
    }
    output[length]= '\0';
    
    return output;
}

#endif //CLA_MAIN_C_H
