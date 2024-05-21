#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>

using namespace std;

//--------------------------------------------------------
const int P     = 100; //P=LDA
const int LWORK = 3*P-1;
double A[P][P], W[P], WORK[LWORK];
//--------------------------------------------------------
extern "C" void
dsyev_(const char&  JOBZ,const char& UPLO,
       const int &     N,
       double    A[P][P],const int &  LDA,
       double    W   [P],
       double    WORK[P],
       const int & LWORK,      int & INFO);
//--------------------------------------------------------
int main(){
  int    N;
  int    i,j;
  int    LDA,INFO;
  char   JOBZ,UPLO;
  string buf;
  //Define the **symmetric** matrix to be diagonalized
  //The subroutine uses the upper triangular part 
  //(UPLO='U') in the *FORTRAN* column-major mode, 
  //therefore in C++, we need to define its *lower*
  //triangular part!
  N = 4; // an N x N matrix
  A[0][0]=-7.7;
  A[1][0]= 2.1;A[1][1]= 8.3;
  A[2][0]=-3.7;A[2][1]=-16.;A[2][2]=-12.;
  A[3][0]= 4.4;A[3][1]= 4.6;A[3][2]=-1.04;A[3][3]=-3.7;
  
  //We print the matrix A before calling DSYEV since  
  //it is destroyed after the call.
  for(i=0;i<N;i++)
    for(j=0;j<=i;j++)
      cout << "A( "   << i+1 << " , " << j+1 << " )="
           << A[i][j] << endl;
  //We ask for eigenvalues AND eigenvectors (JOBZ='V')
  JOBZ='V'; UPLO='U';
  cout << "COMPUTING WITH DSYEV:"      << endl;
  // LDA: Leading dimension of A = number of rows of
  // full array
  LDA = P; 
  dsyev_(JOBZ,UPLO,N,A,LDA,W,WORK,LWORK,INFO);
  cout << "DSYEV: DONE. CHECKING NOW:" << endl;
  if(INFO != 0){cerr << "DSYEV failed. INFO= "
                     << INFO << endl;exit(1);}
  //Print results: W(I) has the eigenvalues:
  cout << "DSYEV: DONE.:"              << endl;
  cout << "EIGENVALUES  OF MATRIX:"    << endl;
  cout.precision(17);
  for(i=0;i<N;i++)
    cout << "LAMBDA("<< i+1
         << ")="     << W[i]           << endl;
  //Eigenvectors are in stored in the rows of A:
  cout << "EIGENVECTORS OF MATRIX:"    << endl;
  for(i=0;i<N;i++){
    cout << "EIGENVECTOR "     << i+1
         << " FOR EIGENVALUE " << W[i] << endl;
    for(j=0;j<N;j++)
      cout << "V_"  << i+1     << "("  << j+1
           << ")= " << A[i][j]         << endl; 
  }

}// main()
//  ---------------------------------------------------------------------
//  Copyright by Konstantinos N. Anagnostopoulos (2004-2014)
//  Physics Dept., National Technical University,
//  konstant@mail.ntua.gr, www.physics.ntua.gr/~konstant
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, version 3 of the License.
//  
//  This program is distributed in the hope that it will be useful, but
//  WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//  
//  You should have received a copy of the GNU General Public Liense along
//  with this program.  If not, see <http://www.gnu.org/licenses/>.
//  -----------------------------------------------------------------------
