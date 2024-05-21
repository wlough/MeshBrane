//************************************************************
//PROGRAM LAPLACE_EM
//Computes the electrostatic potential around conductors.
//The computation is performed on a square lattice of linear
//dimension L. A relaxation method is used to converge to the
//solution of Laplace equation for the potential.
//DATA STRUCTURE:
//double V[L][L]: Value of the potential on the lattice sites
//bool isConductor[L][L]: If true  site has fixed potential
//                        If false site is empty space
//double epsilon: Determines the accuracy of the solution
//The maximum difference of the potential on each site
//between two consecutive sweeps should be less than epsilon.
//PROGRAM STRUCTURE
//main program: 
// . Data Input
// . call functions for initialization, computation and
//   printing of results
//   function initialize_lattice:
// . Initilization of V[L][L] and isConductor[L][L]
//   function laplace:
// . Solves laplace equation using a relaxation method
//   function print_results:
// . Prints results for V[L][L] in a file. Uses format 
//   compatible with splot of gnuplot.
//************************************************************
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
const  int L = 31;
bool   isConductor[L][L];
double V          [L][L];
//--------------------------------------------------------
void initialize_lattice(const double& V1,const double& V2);
void laplace           (const double& epsilon);
void print_results     ();
//--------------------------------------------------------
int main(){
  string buf;
  double V1,V2,epsilon;

  cout << "Enter V1,V2:"                 << endl;
  cin  >> V1 >> V2;             getline(cin,buf);
  cout << "Enter epsilon:"               << endl;
  cin  >> epsilon;              getline(cin,buf);
  cout << "Starting Laplace:"            << endl;
  cout << "Grid Size= "            << L  << endl;
  cout << "Conductors set at V1= " << V1
       << " V2= "                  << V2 << endl;
  cout << "Relaxing with accuracy epsilon= "
       << epsilon << endl;
  //The arrays V and isConductor are initialized
  initialize_lattice(V1,V2);
  //On entry, V,isConductor is initialized.
  //On exit the routine gives the solution V
  laplace(epsilon);
  //We print V in a file.
  print_results();
}//main()
//************************************************************
//function initialize_lattice
//Initializes arrays V[L][L] and isConductor[L][L].
//V[L][L]= 0.0  and isConductor[L][L]=  false  by default
//isConductor[i][j]= true on boundary of lattice where V=0
//isConductor[i][j]= true on sites with i=  L/3+1,5<=j<= L-5
//isConductor[i][j]= true on sites with i=2*L/3+1,5<=j<= L-5
//V[i][j] = V1 on all sites with i=  L/3+1,5<=j<= L-5
//V[i][j] = V2 on all sites with i=2*L/3+1,5<=j<= L-5
//V[i][j] = 0  on boundary (i=1,L and j=1,L)
//V[i][j] = 0  on interior sites with isConductor[i][j]=false 
//INPUT: 
//integer L: Linear size of lattice
//double V1,V2: Values of potential on interior conductors
//OUTPUT:
//double V[L][L]: Array provided by user. Values of potential
//bool   isConductor[L][L]: If true  site has fixed potential
//                          If false site is empty space
//************************************************************
void initialize_lattice(const double& V1,const double& V2){
  
  //Initialize to 0 and false (default values for 
  //boundary and interior sites).
  for(int i=0;i<L;i++)
    for(int j=0;j<L;j++){
      V          [i][j  ] = 0.0;
      isConductor[i][j  ] = false;
    }
  //We set the boundary to be a conductor: (V=0 by default)
  for(int i=0;i<L;i++){
    isConductor[0  ][i  ] = true;
    isConductor[i  ][0  ] = true;
    isConductor[L-1][i  ] = true;
    isConductor[i  ][L-1] = true;
  }
  //We set two conductors at given potential V1 and V2
  for(int i=4;i<L-5;i++){
    V          [  L/3][i] = V1;
    isConductor[  L/3][i] = true;
    V          [2*L/3][i] = V2;
    isConductor[2*L/3][i] = true;
  }
}//initialize_lattice()
//************************************************************
//function laplace
//Uses a relaxation method to compute the solution of the 
//Laplace equation for the electrostatic potential on a 
//2 dimensional squarelattice of linear size L. 
//At every sweep of the lattice we compute the average 
//Vav of the potential at each site (i,j) and we immediately
//update V[i][j]
//The computation continues until Max |Vav-V[i][j]| < epsilon
//INPUT:
//integer L: Linear size of lattice
//double V[L][L]: Value of the potential at each site
//bool   isConductor[L][L]: If  true   potential is fixed
//                          If  false  potential is updated
//double epsilon: if Max |Vav-V[i][j]| < epsilon return to
//callingprogram. 
//OUTPUT:
//double V[L][L]: The computed solution for the potential
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
          Vav = 0.25*(V[i-1][j]+V[i+1][j]+V[i][j-1]+V[i][j+1]);
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
//function  print_results
//Prints the array V[L][L] in file "data"
//The format of the output is appropriate for the splot 
//function of gnuplot: Each time i changes an empty line
// is printed.
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
