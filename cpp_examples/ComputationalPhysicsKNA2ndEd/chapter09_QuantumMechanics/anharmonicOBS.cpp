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
double   H [P][P], X [P][P], X4[P][P];
double   P2[P][P], X2[P][P];
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
void calculate_obs();
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
  //Calculate observables:
  calculate_obs();
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
}//calculate_evs()
//--------------------------------------------------------
void calculate_H(){
  double X2[P][P];
  
  for(  int n =0;n<DIM;n++){
    for(int m =0;m<DIM;m++)
      H[n][m] = lambda*X4[n][m];
    H  [n][n]+= n+0.5;
  }

}//calculate_H()
//--------------------------------------------------------
void calculate_X4(){
  double iP[P][P];
  const double isqrt2=1.0/sqrt(2.0);

  for(  int n=0;n<DIM;n++)
    for(int m=0;m<DIM;m++){
      X[n][m]=0.0;iP[n][m]=0.0;
    }

  for(  int n=0;n<DIM;n++){
    int m=n-1;
    if(m>=0 )X [n][m] = isqrt2*sqrt(double(m+1));
    if(m>=0 )iP[n][m] =-isqrt2*sqrt(double(m+1));
    m    =n+1;
    if(m<DIM)X [n][m] = isqrt2*sqrt(double(m  ));
    if(m<DIM)iP[n][m] = isqrt2*sqrt(double(m  ));
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
  // P2 =-iP . iP
  for(    int n=0;n<DIM;n++)
    for(  int m=0;m<DIM;m++){
      P2  [n][m]  = 0.0;
      for(int k=0;k<DIM;k++)
        P2[n][m] -= iP[n][k]*iP[k][m];
    }
}//calculate_X4()
//--------------------------------------------------------
void calculate_obs(){
  double avX2[P],avP2[P],avX4[P],DxDp[P];
  
  for(    int n=0;n<DIM;n++){
      avX2[n]    = 0.0; avP2[n] = 0.0;
      avX4[n]    = 0.0; DxDp[n] = 0.0;
    }
  for(    int n=0;n<DIM;n++)
    for(  int m=0;m<DIM;m++)
      for(int k=0;k<DIM;k++){
        avX2[n] += H[n][m]*H[n][k]*X2[m][k];
        avX4[n] += H[n][m]*H[n][k]*X4[m][k];
        avP2[n] += H[n][m]*H[n][k]*P2[m][k];
    }
  for(    int n=0;n<DIM;n++)
      DxDp[n]    = sqrt(avX2[n] * avP2[n]);

  cout << "avX2 " << DIM << " "   << lambda  << " ";
  for(    int n=0;n<DIM;n++) cout << avX2[n] << " ";
  cout << endl;
  cout << "avX4 " << DIM << " "   << lambda  << " ";
  for(    int n=0;n<DIM;n++) cout << avX4[n] << " ";
  cout << endl;
  cout << "avP2 " << DIM << " "   << lambda  << " ";
  for(    int n=0;n<DIM;n++) cout << avP2[n] << " ";
  cout << endl;
  cout << "DxDp " << DIM << " "   << lambda  << " ";
  for(    int n=0;n<DIM;n++) cout << DxDp[n] << " ";
  cout << endl;
  
}//calculate_obs()
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
