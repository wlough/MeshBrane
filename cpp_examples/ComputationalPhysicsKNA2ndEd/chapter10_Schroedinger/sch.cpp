//===========================================================
//
// File: sch.cpp
//
// Integrate 1d Schrodinger equation from xmin to xmax. 
// Determine energy eigenvalue and eigenfunction by matching 
// evolving solutions from xmin and from xmax at a point xm. 
// Mathing done by equating values of functions and their 
// derivatives at xm. The point xm chosen at the left most 
// turning point of the potential at any given value of the
// energy. The potential and boundary conditions chosen in 
// different file.
// ----------------------------------------------------------
// Input:  energy: Trial value of energy
//         de: energy step, if matching fails de -> e+de, if
//             logderivative changes sign     de -> -de/2
//         xmin, xmax, Nx
// ----------------------------------------------------------
// Output: Final value of energy, number of nodes of 
//    wavefunction in stdout
//    Final eigenfunction in file psi.dat
//    All trial functions and energies in file all.dat
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
const int P = 20001;
double energy;
//--------------------------------------------------------
double V  (const double& x                          );
double integrate(double* psi ,
           const double& dx  ,const int   & Nx      );
void
boundary  (const double& xmin,const double& xmax    ,
                 double& psixmin,   double& psipxmin,
                 double& psixmax,   double& psipxmax);
void
RKSTEP(          double& t      ,   double& x1,
                 double& x2     ,
           const double& dt                         );
//--------------------------------------------------------
int main(){
  int Nx,NxL,NxR;
  double psi[P],psip[P];
  double dx;
  double xmin,xmax,xm; //left/right/matching points
  double psixmin ,psipxmin ,psixmax ,psipxmax;
  double psileft ,psiright ,psistep ,psinorm;
  double psipleft,psipright,psipstep;
  double de,epsilon;
  double matchlogd,matchold,psiold,norm,x;
  int    iter,i,imatch,nodes;
  string buf;
  //Input:
  cout << "# Enter energy,de,xmin,xmax,Nx:\n";
  cin  >> energy>>de>>xmin>>xmax>>Nx;getline(cin,buf);
  //need even intervals for normalization integration:
  if(Nx   %2 == 0) Nx++;
  if(Nx   >P   ){cerr << "Error: Nx  > P   \n";exit(1);}
  if(xmin>=xmax){cerr << "Error: xmin>=xmax\n";exit(1);}
  dx      = (xmax-xmin)/(Nx-1);
  epsilon = 1.0e-6;
  boundary(xmin,xmax,psixmin,psipxmin,psixmax,psipxmax);
  cout <<"# #######################################\n";
  cout <<"# Estart= "<< energy <<" de= "<< de  << "\n";
  cout <<"# Nx=  " << Nx <<" eps= " << epsilon << "\n";
  cout <<"# xmin= "<< xmin << "xmax= "<< xmax
       <<" dx= "   << dx                       << "\n";
  cout <<"# psi(xmin)= "<< psixmin
       <<" psip(xmin)= "<< psipxmin            << "\n";
  cout <<"# psi(xmax)= "<< psixmax
       <<" psip(xmax)= "<< psipxmax            << "\n";
  cout <<"# #######################################\n";
  //Calculate:
  ofstream myfile("all.dat");
  myfile.precision(17);
  cout  .precision(17);
  matchold   = 0.0;
  for(iter=1;iter<=10000;iter++){
    //Determine matching point at turning point from the left:
    imatch=-1;
    for(i=0;i<Nx;i++){
      x = xmin + i * dx;
      if(imatch <  0 && (energy-V(x)) > 0.0) imatch = i;
    }
    if(imatch < 100 || imatch >= Nx-100) imatch = Nx/5-1;
    xm        = xmin + imatch*dx;
    NxL       = imatch+1;
    NxR       = Nx-imatch;
    //Evolve wavefunction from the left:
    psi  [0]  = psixmin;
    psip [0]  = psipxmin;
    psistep   = psixmin;
    psipstep  = psipxmin;
    for(i=1;i<NxL;i++){
      x       = xmin+(i-1)*dx;//this is x before the step
      RKSTEP(x,psistep,psipstep, dx);
      psi [i] = psistep;
      psip[i] = psipstep;
    }
    //use this to normalize eigenfunction to match at xm
    psinorm   = psistep;
    psipleft  = psipstep;
    //Evolve wavefunction from the right:
    psi [Nx-1]= psixmax;
    psip[Nx-1]= psipxmax;
    psistep   = psixmax;
    psipstep  = psipxmax;
    for(i=1;i<NxR;i++){
      x       = xmax-(i-1)*dx;
      RKSTEP(x,psistep,psipstep,-dx);
      psi [Nx-i-1] = psistep;
      psip[Nx-i-1] = psipstep;
    }
    psinorm   = psistep/psinorm;
    psipright = psipstep;
    //Renormalize psil so that psil(xm)=psir(xm)
    for(i=0;i<NxL-1;i++){
      psi [i] *= psinorm;
      psip[i] *= psinorm;
    }
    psipleft  *= psinorm;
    //print current solution:
    for(i=0;i<Nx;i++){
      x = xmin + i *dx;
      myfile << iter  <<" "<<energy<<" "
             << x     <<" "<<psi[i]<<" "
             <<psip[i]             <<"\n";
    }
    //matching using derivatives:
    //Careful: this can fail if psi'(xm) = 0
    //(use also |de|<1e-6 criterion)
    matchlogd =
      (psipright-psipleft)/(abs(psipright)+abs(psipleft));
    cout << "# iter,energy,de,xm,logd: "
         << iter      << " "
         << energy    << " "
         << de        << " "
         << xm        << " "
         << matchlogd << "\n";
    //break condition:
    if(abs(matchlogd)<=epsilon ||
       abs(de/energy)< 1.0e-12    ) break;
    if(matchlogd * matchold < 0.0 ) de = -0.5*de;
    energy   += de;
    matchold  = matchlogd;
  }//for(iter=1;iter<=10000;iter++)
  myfile.close();
  //------------------------------------------------------
  //Solution has been found and now it is stored:
  norm = integrate(psi,dx,Nx);
  norm = 1.0/sqrt(norm);
  //for(i=0;i<Nx;i++) psi[i] *= norm;
  //Count number of zeroes, add one and get energy level:
  nodes  = 1;
  psiold = psi[0];
  for(i=1;i<Nx-1;i++)
    if(abs(psi[i])     > epsilon){
      if(psiold*psi[i] < 0.0    ) nodes++;
      psiold = psi[i];
    }
  //Print final solution:
  myfile.open("psi.dat");
  cout << "Final result: E= " << energy
       << " n= "              << nodes
       << " norm= "           << norm   << endl;
  if(abs(matchlogd) > epsilon)
    cout << "Final result: SOS: logd>epsilon. logd= "
         << matchlogd                   << endl;
  myfile << "# E= "           << energy
         << " n= "            << nodes
         << " norm = "        << norm   << endl;
  for(i=0;i<Nx;i++){
    x = xmin + i * dx;
    myfile << x << " "       << psi[i]  << "\n";
  }
  myfile.close();
}//main()
//========================================================
//Simpson's rule to integrate psi(x)*psi(x) for proper
//normalization. For n intervals of width dx (n even)
//Simpson's rule is:
//int(f(x)dx) = 
//(dx/3)*(f(x_0)+4 f(x_1)+2 f(x_2)+...+4 f(x_{n-1})+f(x_n))
//
//Input:   Discrete values of function psi[Nx], Nx is odd
//         Integration step dx
//Returns: Integral(psi(x)psi(x) dx)
//========================================================
double integrate(double* psi,
           const double& dx ,const int& Nx){
  double Integral;
  int    i;
  //zeroth order point:
  i         = 0;
  Integral  = psi[i]*psi[i];
  //odd  order points:
  for(i=1;i<=Nx-2;i+=2) Integral += 4.0*psi[i]*psi[i];
  //even order points:
  for(i=2;i<=Nx-3;i+=2) Integral += 2.0*psi[i]*psi[i];
  //last point:
  i         = Nx-1;
  Integral += psi[i]*psi[i];
  //measure normalization:
  Integral *=dx/3.0;

  return Integral;
}//integrate()
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
