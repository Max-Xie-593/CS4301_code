///////////////////////////////////////////////////////////////////////////////
// matmul.c
// Author: Dr. Richard A. Goodrum, Ph.D.
// Date: 16 September 2017
// Modification History
//				modified by Max Xie
//				When: 2 November 2018
//
// Procedures:
// main	    generates matrices and tests matmul with the cannon's algorithm
// matmul	basic, brute force matrix multiply
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>

///////////////////////////////////////////////////////////////////////////////
// int main( int argc, char *argv[] )
// Author: Dr. Richard A. Goodrum, Ph.D.
// Date:  16 September 2017
// Modification History
//				modified by Max Xie
//				When: 2 November 2018
// Description: Generates two matrices and then calls matmul to multiply them.
// It also uses the Cannon's algorithm to speedup computation
// 	Finally, it verifies that the results are correct.
//
// Parameters:
//	argc	I/P	int	   The number of arguments on the command line
//	argv	I/P	char *[]   The arguments on the command line
//	main	O/P	int	   Status code
///////////////////////////////////////////////////////////////////////////////

#define L (16*1024)
#define M (16*1024)
#define N (16*1024)

int matmul(int,int,int,float*,float*,float*);

int main(int argc,char *argv[])
{
  MPI_Init(&argc,&argv); // starts the MPI
  int p;  // number of processes
  MPI_Comm_size(MPI_COMM_WORLD,&p); // return the number of the processes in p
 
  int dimensions[2];
  dimensions[0] = (int)sqrt(p); // determine the dimensions of the processes
  dimensions[1] = (int)sqrt(p); 

  int period[2];
  period[0] = period[1] = 1; // makes the dimensions wrap around

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD,2,dimensions,period,1,&cart_comm); // creates the 2D grid

  int id;
  MPI_Comm_rank(cart_comm,&id); // gets the rank of the process
  
  int selfCoordinates[2];
  MPI_Cart_coords(cart_comm,id,2,selfCoordinates); // gets the coordinates of the process of the given rank

  int rowCoordinates = selfCoordinates[0]; // gets the coordinates of the process
  int colCoordinates = selfCoordinates[1];

  int rowA = rowCoordinates;                  // gets the shifted coordinates of matrix A sub-block
  int colA = colCoordinates + rowCoordinates;
  if(colA >= dimensions[1])
    colA -= dimensions[1];

  int rowB = rowCoordinates + colCoordinates; // gets the shifted coordinates of matrix B sub-block
  if(rowB >= dimensions[0])
    rowB -= dimensions[0];
  int colB = colCoordinates;
  
  int localL = L / dimensions[0]; // grabs the blocks of the matrix each process is going to create
  int localM = M / dimensions[1];
  int localN = N / dimensions[1];
  
  float* A = malloc(localL*localM*sizeof(float)); // allocates space for the sub-matrix
  float* B = malloc(localM*localN*sizeof(float));
  float* C = malloc(localL*localN*sizeof(float));

  int i, j, k, l;
  
  for(i=0;i<localL;i++)							// generates the A values of the sub-matrix as if it was skewed already
    for(j=0;j<localM;j++)
      A[i*localM+j] = (float) ((rowA*localL+i)*M+(colA*localM+j)+1);
  
  for(j=0;j<localM;j++)							// generates the B values of the sub-matrix as if it was skewed already
    for(k=0;k<localN;k++)
      if((rowB*localM+j) <= (colB*localN+k))
		B[j*localN+k] = 1.0;
      else
      	B[j*localN+k] = 0.0;
 
  for(i=0;i<localL;i++)							// initializes the C sub-matrix to 0
    for(k=0;k<localN;k++)
      C[i*localN+k] = 0.0;

  int leftNeighborCoordinates[2];
  int leftNeighborRank;

  leftNeighborCoordinates[0] = rowCoordinates; // find coordinates to determine the left neighbor's coordinates
  leftNeighborCoordinates[1] = colCoordinates - 1;
  if(leftNeighborCoordinates[1] < 0)
    leftNeighborCoordinates[1] += dimensions[1];
  MPI_Cart_rank(cart_comm,leftNeighborCoordinates,&leftNeighborRank); // gets the rank of the left neighbor

  int rightNeighborCoordinates[2];
  int rightNeighborRank;

  rightNeighborCoordinates[0] = rowCoordinates; // find coordinates to determine the right neighbor's coordinates
  rightNeighborCoordinates[1] = colCoordinates + 1;
  if(rightNeighborCoordinates[1] >= dimensions[1])
    rightNeighborCoordinates[1] -= dimensions[1];
  MPI_Cart_rank(cart_comm,rightNeighborCoordinates,&rightNeighborRank); // gets the rank of the right neighbor

  int upNeighborCoordinates[2];
  int upNeighborRank;
  
  upNeighborCoordinates[0] = rowCoordinates - 1; // find coordinates to determine the up neighbor's coordinates
  if(upNeighborCoordinates[0]<0)
    upNeighborCoordinates[0] += dimensions[0];
  upNeighborCoordinates[1] = colCoordinates;
  MPI_Cart_rank(cart_comm,upNeighborCoordinates,&upNeighborRank); // gets the rank of the up neighbor

  int downNeighborCoordinates[2];
  int downNeighborRank;

  downNeighborCoordinates[0] = rowCoordinates + 1; // find coordinates to determine the up neighbor's coordinates
  if(downNeighborCoordinates[0] >= dimensions[0])
    downNeighborCoordinates[0] -= dimensions[0];
  downNeighborCoordinates[1] = colCoordinates;
  MPI_Cart_rank(cart_comm,downNeighborCoordinates,&downNeighborRank); // gets the rank of the up neighbor

  struct timeval start, stop;
  
  float* tempA = malloc(localL*localM*sizeof(float)); // allocates space for the temporary arrays of A and B
  float* tempB = malloc(localM*localN*sizeof(float));
  
  gettimeofday(&start,NULL); // starts the timer
  for(l=0;l<dimensions[0];l++) 
    {    
      matmul(localL,localM,localN,A,B,C); // does the matrix multiply
      if(((id + rowCoordinates)%2) == 1)  // depending on the id of the process, it either sends first or receive first
		{
		  MPI_Send(A,localL*localM,MPI_FLOAT,leftNeighborRank,0,cart_comm); // sends A matrix first 
		  MPI_Recv(A,localL*localM,MPI_FLOAT,rightNeighborRank,0,cart_comm,MPI_STATUS_IGNORE); // receives A matrix from right neighbor
		  
		  MPI_Recv(tempB,localM*localN,MPI_FLOAT,downNeighborRank,0,cart_comm,MPI_STATUS_IGNORE); // receives B matrix into the temp array
		  MPI_Send(B,localM*localN,MPI_FLOAT,upNeighborRank,0,cart_comm); // sends the B matrix
		  
		  float* tempBP = B; // replaces the contents of B with tempB
		  B = tempB;
		  tempB = tempBP;	  
		}
      else
		{
		  MPI_Recv(tempA,localL*localM,MPI_FLOAT,rightNeighborRank,0,cart_comm,MPI_STATUS_IGNORE); // receives A matrix into the temp array
		  MPI_Send(A,localL*localM,MPI_FLOAT,leftNeighborRank,0,cart_comm); // sends the A matrix
		  
		  float* tempAP = A; // replaces the contents of A with tempA
		  A = tempA;
		  tempA = tempAP;        
		  
		  MPI_Send(B,localM*localN,MPI_FLOAT,upNeighborRank,0,cart_comm); // sends B matrix 
		  MPI_Recv(B,localM*localN,MPI_FLOAT,downNeighborRank,0,cart_comm,MPI_STATUS_IGNORE); // receives B matrix from down neighbor
		}
    }
  gettimeofday(&stop,NULL); // stops the timer   
  float elapsed = ( (stop.tv_sec-start.tv_sec) +
		    (stop.tv_usec-start.tv_usec)/(float)1000000 );
  float flops = ( 2 * (float)L * (float)M * (float)N ) / elapsed;
  MPI_Barrier(cart_comm); // waites for all processes to reach here
  if(id == 0) // only root process prints out its values
    {
		printf( "p=%d, L=%d, M=%d, N=%d, elapsed=%g, flops=%g\n", 
		p, L, M, N, elapsed, flops );
		fflush(stdout);
    }
#ifdef DEBUG // if want to print out contents of the array
  if(id != 0) // if not the root process, then send its array contents and coordinates to root process
    {
      MPI_Send(C,localL*localN,MPI_FLOAT,0,id*10,cart_comm); // tag is id * 10 to check for the array contents
      MPI_Send(selfCoordinates,2,MPI_INT,0,id*10+1,cart_comm); // tag is id * 10 + 1 to check for the coordinates contents
    }
  else // this is root process
    {
      float* bigC = malloc(L*N*sizeof(float));            // allocates space for the big array and temp array for the receive
      float* tempC = malloc(localL*localN*sizeof(float));
      int rankCoordinates[2]; // keeps coordinates
      for(i=0;i<p;i++) // loops through all the processes in the program
		if(i==0)  // root process just transfer its sub-matrix to the bigC
			for(j=0;j<localL;j++)
				for(k=0;k<localN;k++)    
					bigC[j*L+k] = C[j*localL+k];
		else  // all other processes
		{
			MPI_Recv(tempC,localL*localN,MPI_FLOAT,i,i*10,cart_comm,MPI_STATUS_IGNORE); // waits on the contents of the array
			MPI_Recv(rankCoordinates,2,MPI_INT,i,i*10+1,cart_comm,MPI_STATUS_IGNORE); // waits on the contents of the coordinates
			for(j=0;j<localL;j++) // transfer its sub-matrix to the bigC
				for(k=0;k<localN;k++)  
					bigC[(rankCoordinates[0]*N*localL)+(rankCoordinates[1]*localN)+(j*N)+k] = tempC[j*localL+k]; 
		}
      
      printf( "C:\n" ); // prints the big array
      for( i=0; i<L; i++ )
		{
		printf( "%g", bigC[i*N] );
		for( k=1; k<N; k++ )
			printf( " %g", bigC[i*N+k] );
		printf( "\n" );
		}
      fflush(stdout);
      
      free(bigC); // frees the malloc memory
      free(tempC);
    }  
#endif
  free(A); // frees the malloc memory
  free(B);
  free(tempA);
  free(tempB);
  free(C);
  MPI_Finalize();
}

///////////////////////////////////////////////////////////////////////////////
// int matmul( int l, int m, int n, float *A, float *B, float *C )
// Author: Dr. Richard A. Goodrum, Ph.D.
// Date:  16 September 2017
// Description: Generates two matrices and then calls matmul to multiply them.
// 	Finally, it verifies that the results are correct.
//
// Parameters:
//	l	I/P	int	The first dimension of A and C
//	m	I/P	int	The second dimension of A and  first of B
//	n	I/P	int	The second dimension of B and C
//	A	I/P	float *	The first input matrix
//	B	I/P	float *	The second input matrix
//	C	O/P	float *	The output matrix
//	matmul	O/P	int	Status code
///////////////////////////////////////////////////////////////////////////////
int matmul( int l, int m, int n, float *A, float *B, float *C )
{
  int i, j, k;  
  #pragma omp parallel for private (j,k)
  for( i=0; i<l; i++ )					// Loop over the rows of A and C.
    for( j=0; j<n; j++ )				// Loop over the columns of B and C
      for( k=0; k<m; k++ )				// Loop over the columns of A and C 
        C[i*n+k] += A[i*m+j] * B[j*n+k];// Compute the inner product 
}
