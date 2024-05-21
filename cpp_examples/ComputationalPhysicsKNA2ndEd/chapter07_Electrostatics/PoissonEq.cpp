//************************************************************
//set the boundary of a square to given potentials
//************************************************************
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
const  int L = 51;
bool   isConductor[L][L];
double V          [L][L];
double rho        [L][L];
//--------------------------------------------------------
void initialize_lattice(const double& V1,const double& V2,
                        const double& V3,const double& V4,
                        const double&  Q);
void laplace           (const double& epsilon);
void print_results     ();
//--------------------------------------------------------
int main(){
  string buf;
  double V1,V2,V3,V4,Q,epsilon;

  cout << "Enter V1,V2,V3,V4:"           << endl;
  cin  >> V1 >> V2 >> V3 >> V4; getline(cin,buf);
  cout << "Enter 4*PI*Q:";
  cin  >> Q;                    getline(cin,buf);
  cout << "Enter epsilon:"               << endl;
  cin  >> epsilon;              getline(cin,buf);
  cout << "Starting Laplace:"            << endl;
  cout << "Grid Size= "            << L  << endl;
  cout << "Conductors set at V1= " << V1
       << " V2= "                  << V2
       << " V3= "                  << V3
       << " V4= "                  << V4
       << " Q = "                  << Q  << endl;
  cout << "Relaxing with accuracy epsilon= "
       << epsilon << endl;

  initialize_lattice(V1,V2,V3,V4,Q);

  laplace(epsilon);

  print_results();

}//main()
//************************************************************
void initialize_lattice(const double& V1,const double& V2,
                        const double& V3,const double& V4,
                        const double&  Q ){
  int    L1,L2;
  double Area;
  //Initialize to 0 and .FALSE (default values 
  //for boundary and interior sites).
  for(int i=0;i<L;i++)
    for(int j=0;j<L;j++){
      V          [i][j  ] = 0.0;
      isConductor[i][j  ] = false;
      rho        [i][j  ] = 0.0;
    }
  //We set the boundary to be a conductor: (V=0 by default)
  for(int i=0;i<L;i++){
    isConductor[0  ][i  ] = true;
    isConductor[i  ][0  ] = true;
    isConductor[L-1][i  ] = true;
    isConductor[i  ][L-1] = true;
    V          [0  ][i  ] = V1;
    V          [i  ][L-1] = V2;
    V          [L-1][i  ] = V3;
    V          [i  ][0  ] = V4;
  }
  //We set the points with non-zero charge
  //A uniform distribution at a center square
  L1 = (L/2) - 5;
  L2 = (L/2) + 5;
  if(L1< 0){cerr <<"Array out of bounds. L1< 0\n";exit(1);}
  if(L2>=L){cerr <<"Array out of bounds. L2>=0\n";exit(1);}
  Area = (L2-L1+1)*(L2-L1+1);
  
  for(  int i=L1;i<=L2;i++)
    for(int j=L1;j<=L2;j++)
      rho[i][j] = Q/Area;
      
}//initialize_lattice()
//************************************************************
void laplace           (const double& epsilon){
  int    icount;
  double Vav,error,dV;

  icount = 0;
  while(icount < 10000){
    icount++;
    error = 0.0;
    for(  int i = 1;i<L-1;i++){
      for(int j = 1;j<L-1;j++){
        //We change V only for non conductors:
        if( ! isConductor[i][j]){
          Vav = 0.25*(V[i-1][j]+V[i+1][j]+V[i][j-1]+V[i][j+1]
                      +rho[i][j]);
          dV  = abs(V[i][j]-Vav);
          if(error < dV) error = dV; //maximum error
          V[i][j] = Vav;
        }//if( ! isConductor[i][j])
      }//for(int j = 1;j<L-1;j++)
    }//for(  int i = 1;i<L-1;i++)
    cout << icount << " err= " << error << endl ;
    if(error < epsilon) return;
  }//while(icount < 10000)
  cerr << "Warning: laplace did not converge.\n";
}//laplace()
//************************************************************
void print_results(){
  ofstream  myfile("data");
  myfile.precision(16);
  for(  int i = 0; i < L ; i++){
    for(int j = 0; j < L ; j++){
      myfile << i+1 << " " << j+1 << " " << V[i][j] << endl;
    }
    //print empty line for gnuplot, separate isolines:
    myfile   << " "                                 << endl;
  }
  myfile.close();
}//print_results()
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
