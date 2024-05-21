#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>

using namespace std;

//--------------------------------------------------------
const int P     = 1000; //P=LDA
const int LWORK = 3*P-1;
int      DIM;
double   H[P][P], X[P][P], X4[P][P];
double   E[P], WORK[LWORK];
double   lambda;
//--------------------------------------------------------
extern "C" void
dsyev_(const char&  JOBZ,const char& UPLO,
       const int &     N,
       double    H[P][P],const int &  LDA,
       double    E   [P],
       double    WORK[P],
       const int & LWORK,      int & INFO);
//--------------------------------------------------------
void calculate_X4 ();
void calculate_evs();
void calculate_H  ();
//--------------------------------------------------------
int main(){
  string buf;

  cout << "# Enter Hilbert Space dimension:\n";
  cin  >> DIM;                          getline(cin,buf);
  cout << "# Enter lambda:\n";
  cin  >> lambda;                       getline(cin,buf);
  cout << "# lambda= " << lambda                 << endl;
  cout << "# ########################################\n";
  cout << "# Energy spectrum of anharmonic oscillator\n";
  cout << "# using matrix methods.\n";
  cout << "# Hilbert Space Dimension DIM = "<<DIM<< endl;
  cout << "# lambda coupling = " << lambda       << endl;
  cout << "# ########################################\n";
  cout << "# Output: DIM lambda E_0 E_1 .... E_{N-1} \n";
  cout << "# ----------------------------------------\n";
  
  cout.precision(15);
  //Calculate X^4 operator:
  calculate_X4();
  //Calculate eigenvalues:
  calculate_evs();
  cout.precision(17);
  cout << "EV " << DIM << " " << lambda << " ";
  for(int n=0;n<DIM;n++) cout << E[n] << " ";
  cout << endl;
}// main()
//--------------------------------------------------------
void calculate_evs(){
  int INFO;
  const char JOBZ='V',UPLO='U';

  calculate_H();
  dsyev_(JOBZ,UPLO,DIM,H,P,E,WORK,LWORK,INFO);
  if(INFO != 0){
    cerr << "dsyev failed. INFO= " << INFO << endl;
    exit(1);
  }
  cout << "# ***************** EVEC *****************\n";
  for(  int n=0;n<DIM;n++){
    cout   << "# EVEC " << lambda << " ";
    for(int m=0;m<DIM;m++)
      cout << H[n][m]   <<  " ";
    cout   <<              '\n';
  }
}//calculate_evs()
//--------------------------------------------------------
void calculate_H(){
  double X2[P][P];
  
  for(  int n =0;n<DIM;n++){
    for(int m =0;m<DIM;m++)
      H[n][m] = lambda*X4[n][m];
    H  [n][n]+= n+0.5;
  }

  cout << "# ***************** H    *****************\n";
  for(int n=0;n<DIM;n++){
    cout   << "# HH ";
    for(int m=0;m<DIM;m++)
      cout << H[n][m] << " ";
    cout   << '\n';
  }
  cout << "# ***************** H    *****************\n";
}//calculate_H()
//--------------------------------------------------------
void calculate_X4(){
  double X2[P][P];
  const double isqrt2=1.0/sqrt(2.0);

  for(  int n=0;n<DIM;n++)
    for(int m=0;m<DIM;m++)
      X[n][m]=0.0;

  for(  int n=0;n<DIM;n++){
    int m=n-1;
    if(m>=0) X[n][m] = isqrt2*sqrt(double(m+1));
    m    =n+1;
    if(m<DIM)X[n][m] = isqrt2*sqrt(double(m  ));
  }
  // X2 = X  . X
  for(    int n=0;n<DIM;n++)
    for(  int m=0;m<DIM;m++){
      X2  [n][m]  = 0.0;
      for(int k=0;k<DIM;k++)
        X2[n][m] += X [n][k]*X [k][m];
    }
  // X4 = X2 . X2
  for(    int n=0;n<DIM;n++)
    for(  int m=0;m<DIM;m++){
      X4  [n][m]  = 0.0;
      for(int k=0;k<DIM;k++)
        X4[n][m] += X2[n][k]*X2[k][m];
    }
}//calculate_X4()
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
