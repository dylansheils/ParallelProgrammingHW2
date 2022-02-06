/*********************************************************************/
//
// 02/01/2022: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 8388608 // hex digits 
#define block_size 32
#define verbose 0

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size
#define nsupersupersections nsupersections/block_size

//Global definitions of the various arrays used in steps for easy access
int gi[bits] = {0};
int pi[bits] = {0};
int ci[bits] = {0};

int ggj[ngroups] = {0};
int gpj[ngroups] = {0};
int gcj[ngroups] = {0};

int sgk[nsections] = {0};
int spk[nsections] = {0};
int sck[nsections] = {0};

int ssgl[nsupersections] = {0} ;
int sspl[nsupersections] = {0} ;
int sscl[nsupersections] = {0} ;

int sssgm[nsupersupersections] = {0} ;
int ssspm[nsupersupersections] = {0} ;
int ssscm[nsupersupersections] = {0} ;

int sumi[bits] = {0};

int sumrca[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input()
{
  char* in1 = (char *)calloc(input_size+1, sizeof(char));
  char* in2 = (char *)calloc(input_size+1, sizeof(char));

  if( 1 != scanf("%s", in1))
    {
      printf("Failed to read input 1\n");
      exit(-1);
    }
  if( 1 != scanf("%s", in2))
    {
      printf("Failed to read input 2\n");
      exit(-1);
    }
  
  hex1 = grab_slice_char(in1,0,input_size+1);
  hex2 = grab_slice_char(in2,0,input_size+1);
  
  free(in1);
  free(in2);
}

void compute_gp()
{
    for(int i = 0; i < bits; i++)
    {
        gi[i] = bin1[i] & bin2[i];
        pi[i] = bin1[i] | bin2[i];
    }
}

void compute_group_gp()
{
    for(int j = 0; j < ngroups; j++)
    {
        int jstart = j*block_size;
        int* ggj_group = grab_slice(gi,jstart,block_size);
        int* gpj_group = grab_slice(pi,jstart,block_size);

        int sum = 0;
        for(int i = 0; i < block_size; i++)
        {
            int mult = ggj_group[i]; //grabs the g_i term for the multiplication
            for(int ii = block_size-1; ii > i; ii--)
            {
                mult &= gpj_group[ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
            }
            sum |= mult; //sum up each of these things with an or
        }
        ggj[j] = sum;

        int mult = gpj_group[0];
        for(int i = 1; i < block_size; i++)
        {
            mult &= gpj_group[i];
        }
        gpj[j] = mult;

	// free from grab_slice allocation
	free(ggj_group);
	free(gpj_group);
    }
}

void compute_section_gp()
{
  for(int k = 0; k < nsections; k++)
    {
      int kstart = k*block_size;
      int* sgk_group = grab_slice(ggj,kstart,block_size);
      int* spk_group = grab_slice(gpj,kstart,block_size);
      
      int sum = 0;
      for(int i = 0; i < block_size; i++)
        {
	  int mult = sgk_group[i];
	  for(int ii = block_size-1; ii > i; ii--)
            {
	      mult &= spk_group[ii];
            }
	  sum |= mult;
        }
      sgk[k] = sum;
      
      int mult = spk_group[0];
      for(int i = 1; i < block_size; i++)
        {
	  mult &= spk_group[i];
        }
      spk[k] = mult;
      
      // free from grab_slice allocation
      free(sgk_group);
      free(spk_group);
    }
}

void compute_super_section_gp()
{
  for(int l = 0; l < nsupersections ; l++)
    {
      int lstart = l*block_size;
      int* ssgl_group = grab_slice(sgk,lstart,block_size);
      int* sspl_group = grab_slice(spk,lstart,block_size);
      
      int sum = 0;
      for(int i = 0; i < block_size; i++)
        {
	  int mult = ssgl_group[i];
	  for(int ii = block_size-1; ii > i; ii--)
            {
	      mult &= sspl_group[ii];
            }
	  sum |= mult;
        }
      ssgl[l] = sum;
      
      int mult = sspl_group[0];
      for(int i = 1; i < block_size; i++)
        {
	  mult &= sspl_group[i];
        }
      sspl[l] = mult;
      
      // free from grab_slice allocation
      free(ssgl_group);
      free(sspl_group);
    }
}

void compute_super_super_section_gp()
{
  for(int m = 0; m < nsupersupersections ; m++)
    {
      int mstart = m*block_size;
      int* sssgm_group = grab_slice(ssgl,mstart,block_size);
      int* ssspm_group = grab_slice(sspl,mstart,block_size);
      
      int sum = 0;
      for(int i = 0; i < block_size; i++)
        {
	  int mult = sssgm_group[i];
	  for(int ii = block_size-1; ii > i; ii--)
            {
	      mult &= ssspm_group[ii];
            }
	  sum |= mult;
        }
      sssgm[m] = sum;
      
      int mult = ssspm_group[0];
      for(int i = 1; i < block_size; i++)
        {
	  mult &= ssspm_group[i];
        }
      ssspm[m] = mult;
      
      // free from grab_slice allocation
      free(sssgm_group);
      free(ssspm_group);
    }
}

void compute_super_super_section_carry()
{
  for(int m = 0; m < nsupersupersections; m++)
    {
      int ssscmlast=0;
      if(m==0)
        {
	  ssscmlast = 0;
        }
      else
        {
	  ssscmlast = ssscm[m-1];
        }
      
      ssscm[m] = sssgm[m] | (ssspm[m]&ssscmlast);
    }
}

void compute_super_section_carry()
{
  for(int l = 0; l < nsupersections; l++)
    {
      int sscllast=0;
      if(l%block_size == block_size-1)
        {
	  sscllast = ssscm[l/block_size];
        }
      else if( l != 0 )
        {
	  sscllast = sscl[l-1];
        }
      
      sscl[l] = ssgl[l] | (sspl[l]&sscllast);
    }
}

void compute_section_carry()
{
  for(int k = 0; k < nsections; k++)
    {
      int scklast=0;
      if(k%block_size==block_size-1)
        {
	  scklast = sscl[k/block_size];
        }
      else if( k != 0 )
        {
	  scklast = sck[k-1];
        }
      
      sck[k] = sgk[k] | (spk[k]&scklast);
    }
}

void compute_group_carry()
{
  for(int j = 0; j < ngroups; j++)
    {
      int gcjlast=0;
      if(j%block_size==block_size-1)
        {
	  gcjlast = sck[j/block_size];
        }
      else if( j != 0 )
        {
	  gcjlast = gcj[j-1];
        }
      
      gcj[j] = ggj[j] | (gpj[j]&gcjlast);
    }
}

void compute_carry()
{
  for(int i = 0; i < bits; i++)
    {
      int clast=0;
      if(i%block_size==block_size-1)
        {
	  clast = gcj[i/block_size];
        }
      else if( i != 0 )
        {
	  clast = ci[i-1];
        }
      
      ci[i] = gi[i] | (pi[i]&clast);
    }
}

void compute_sum()
{
  for(int i = 0; i < bits; i++)
    {
      int clast=0;
      if(i==0)
        {
	  clast = 0;
        }
      else
        {
	  clast = ci[i-1];
        }
      sumi[i] = bin1[i] ^ bin2[i] ^ clast;
    }
}

void cla()
{
    compute_gp();
    compute_group_gp();
    compute_section_gp();
    compute_super_section_gp();
    compute_super_super_section_gp();
    compute_super_super_section_carry();
    compute_super_section_carry();
    compute_section_carry();
    compute_group_carry();
    compute_carry();
    compute_sum();
}

void ripple_carry_adder()
{
  int clast=0, cnext=0;

  for(int i = 0; i < bits; i++)
    {
      cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
      sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
      clast = cnext;
    }
}

void check_cla_rca()
{
  for(int i = 0; i < bits; i++)
    {
      if( sumrca[i] != sumi[i] )
	{
	  printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
		 i, sumrca[i], i, sumi[i]);
	  printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
		 i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
	  return;
	}
    }
  printf("Check Complete: CLA and RCA are equal\n");
}

int main(int argc, char *argv[])
{
  int randomGenerateFlag = 1;
  int deterministic_seed = (1<<30) - 1;
  char* hexa=NULL;
  char* hexb=NULL;
  char* hexSum=NULL;
  char* int2str_result=NULL;
  unsigned long long start_time=clock_now(); // dummy clock reads to init
  unsigned long long end_time=clock_now();   // dummy clock reads to init

  if( nsupersupersections != block_size )
    {
      printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
	     nsupersupersections, block_size );
      return(-1);
    }
  
  if (argc == 2) {
    if (strcmp(argv[1], "-r") == 0)
      randomGenerateFlag = 1;
  }
  
  if (randomGenerateFlag == 0)
    {
      read_input();
    }
  else
    {
      srand( deterministic_seed );
      hex1 = generate_random_hex(input_size);
      hex2 = generate_random_hex(input_size);
    }
  
  hexa = prepend_non_sig_zero(hex1);
  hexb = prepend_non_sig_zero(hex2);
  hexa[digits] = '\0'; //double checking
  hexb[digits] = '\0';
  
  bin1 = gen_formated_binary_from_hex(hexa);
  bin2 = gen_formated_binary_from_hex(hexb);

  start_time = clock_now();
  cla();
  end_time = clock_now();

  printf("CLA Completed in %llu cycles\n", (end_time - start_time));

  start_time = clock_now();
  ripple_carry_adder();
  end_time = clock_now();

  printf("RCA Completed in %llu cycles\n", (end_time - start_time));

  check_cla_rca();

  if( verbose==1 )
    {
      int2str_result = int_to_string(sumi,bits);
      hexSum = revbinary_to_hex( int2str_result,bits);
    }

  // free inputs fields allocated in read_input or gen random calls
  free(int2str_result);
  free(hex1);
  free(hex2);
  
  // free bin conversion of hex inputs
  free(bin1);
  free(bin2);
  
  if( verbose==1 )
    {
      printf("Hex Input\n");
      printf("a   ");
      print_chararrayln(hexa);
      printf("b   ");
      print_chararrayln(hexb);
    }
  
  if ( verbose==1 )
    {
      printf("Hex Return\n");
      printf("sum =  ");
    }
  
  // free memory from prepend call
  free(hexa);
  free(hexb);

  if( verbose == 1 )
    printf("%s\n",hexSum);
  
  free(hexSum);
  
  return 1;
}
