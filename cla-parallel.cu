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
int* gi; int* pi; int* ci; // Size: bits
int* ggj; int* gpj; int* gcj; // Size: ngroups
int* sgk; int* spk; int* sck; // Size: nsections
int* ssgl; int* sspl; int* sscl; // Size: nsupersupersections
int* sssgm; int* ssspm; int* ssscm; // Size: nsupersupersections
int* sumi; // Size: bits

int sumrca[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input()
{
    char* in1 = (char*)calloc(input_size + 1, sizeof(char));
    char* in2 = (char*)calloc(input_size + 1, sizeof(char));

    if (1 != scanf("%s", in1))
    {
        printf("Failed to read input 1\n");
        exit(-1);
    }
    if (1 != scanf("%s", in2))
    {
        printf("Failed to read input 2\n");
        exit(-1);
    }

    hex1 = grab_slice_char(in1, 0, input_size + 1);
    hex2 = grab_slice_char(in2, 0, input_size + 1);

    free(in1);
    free(in2);
}
void ripple_carry_adder() {
    int clast = 0, cnext = 0;
    for (int i = 0; i < bits; i++) {
        cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
        sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
        clast = cnext;
    }
}
void check_cla_rca() {
    for (int i = 0; i < bits; i++) {
        if (sumrca[i] != sumi[i]) {
            printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
                i, sumrca[i], i, sumi[i]);
            printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
                i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i - 1, ci[i - 1]);
            return;
        }
    }
    printf("Check Complete: CLA and RCA are equal\n");
}
void allocations() {
  cudaMallocManaged(&sumi, bits * sizeof(int));
  cudaMallocManaged(&gi, bits*sizeof(int)); cudaMallocManaged(&pi, bits*sizeof(int)); cudaMallocManaged(&ci, bits*sizeof(int));
  cudaMallocManaged(&ggj, ngroups*sizeof(int)); cudaMallocManaged(&gpj, ngroups*sizeof(int)); cudaMallocManaged(&gcj, ngroups*sizeof(int));
  cudaMallocManaged(&sgk, nsections*sizeof(int)); cudaMallocManaged(&spk, nsections*sizeof(int)); cudaMallocManaged(&sck, nsections*sizeof(int));
  cudaMallocManaged(&ssgl, nsupersections*sizeof(int)); cudaMallocManaged(&sspl, nsupersections*sizeof(int)); cudaMallocManaged(&sscl, nsupersections*sizeof(int));
  cudaMallocManaged(&sssgm, nsupersupersections*sizeof(int)); cudaMallocManaged(&ssspm, nsupersupersections*sizeof(int)); cudaMallocManaged(&ssscm, nsupersupersections*sizeof(int));
}
void deallocations() {
  cudaFree(gi); cudaFree(pi); cudaFree(ggj); cudaFree(sumi);
  cudaFree(ggj); cudaFree(gpj); cudaFree(gcj);
  cudaFree(sgk); cudaFree(spk); cudaFree(sck);
  cudaFree(ssgl); cudaFree(sspl); cudaFree(sscl);
  cudaFree(sssgm); cudaFree(ssspm); cudaFree(ssscm);
}

__global__ void compute_gp(int* gi, int* pi, int* bin1, int* bin2, int n) {
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < n) {
        gi[id] = bin1[id] & bin2[id];
        pi[id] = bin1[id] | bin2[id];
    }
}
__global__ void compute_forward(int sectionSize, int* f, int* s, int* t, int* fr) {
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (id < sectionSize) {
      int lstart = id * block_size;
      int sum = 0;
      for (int i = 0; i < block_size; i++) {
          int mult = f[i + lstart];
          for (int ii = block_size - 1; ii > i; ii--) {
              mult &= s[ii + lstart];
          }
          sum |= mult;
      }
      t[id] = sum;
      int mult = s[lstart];
      for (int i = 1; i < block_size; i++) {
          mult &= s[i + lstart];
      }
      fr[id] = mult;
  }
}
__global__ void compute_carry(int sectionSize, int* f, int* s, int* t, int* fr, int A, int B) {
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < sectionSize) {
        if (sectionSize == A) {
            int a = 0;
            if (id == 0) { a = 0; }
            else { a = f[id - 1]; }
            f[id] = s[id] | (t[id] & a);
        } else {
            int a = 0;
            if (id % B == B - 1) { a = f[id / B]; }
            else if (id != 0) { a = s[id - 1]; }
            s[id] = t[id] | (fr[id] & a);
        }
    }
}
__global__ void compute_sum(int n, int* ci, int* sumi, int* bin1, int* bin2) {
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < n) {
        int clast = 0;
        if (id == 0) { clast = 0; }
        else { clast = ci[id - 1]; }
        sumi[id] = bin1[id] ^ bin2[id] ^ clast;
    }
}

void cla() {
    int CUDAThreads = bits;
    int CUDABlock = 1024;
    int CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    cudaMallocManaged(&bin1, bits * sizeof(int)); 
    cudaMallocManaged(&bin2, bits * sizeof(int));

    allocations();

    compute_gp<<<CUDAGrid, CUDABlock>>>(gi, pi, bin1, bin2, bits);

    CUDAThreads = ngroups; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_forward<<<CUDAGrid, CUDABlock>>>(ngroups, gi, pi, ggj, gpj);
    
    CUDAThreads = nsections; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_forward << <CUDAGrid, CUDABlock >> > (nsections, ggj, gpj, sgk, spk);
    
    CUDAThreads = nsupersections; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_forward << <CUDAGrid, CUDABlock >> > (nsupersections, sgk, spk, ssgl, sspl);
    
    CUDAThreads = nsupersupersections; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_forward << <CUDAGrid, CUDABlock >> > (nsupersupersections, ssgl, sspl, sssgm, ssspm);
    compute_carry << <CUDAGrid, CUDABlock >> > (nsupersupersections, ssscm, sssgm, ssspm, NULL, nsupersupersections, block_size);
    
    CUDAThreads = nsupersections; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_carry << <CUDAGrid, CUDABlock >> > (nsupersections, ssscm, sscl, ssgl, sspl, nsupersupersections, block_size);
    
    CUDAThreads = nsections; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_carry << <CUDAGrid, CUDABlock >> > (nsections, sscl, sck, sgk, spk, nsupersupersections, block_size);
    
    CUDAThreads = ngroups; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_carry << <CUDAGrid, CUDABlock >> > (ngroups, sck, gcj, ggj, gpj, nsupersupersections, block_size);
    
    CUDAThreads = bits; CUDAGrid = (int)((CUDAThreads / CUDABlock) + ((CUDAThreads % CUDABlock) != 0));
    compute_carry << <CUDAGrid, CUDABlock >> > (bits, gcj, ci, gi, pi, nsupersupersections, block_size);
    compute_sum<<<CUDAGrid, CUDABlock>>>(bits, ci, sumi, bin1, bin2);

    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
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
  cudaFree(bin2);
  cudaFree(bin1);

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

  if( verbose==1 )
    printf("%s\n",hexSum);
  
  free(hexSum);

  deallocations();
  
  return 1;
}