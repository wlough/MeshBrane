#include <iostream>
#include <fstream>
using namespace std;

int main(){
  const int    N  = 10;
  const double PI = 3.1415926535897932;
  double R[N];
  double area,perimeter;
  int    i;

  for(i=0;i<N;i++){
    cout << "Enter radius of circle: ";
    cin  >> R[i];
    cout << "i= " << (i+1) << " R(i)= " << R[i] << '\n';
  }

  ofstream myfile ("AREA.DAT");
  for(i=0;i<N;i++){
    perimeter = 2.0*PI*R[i];
    area      = PI*R[i]*R[i];
    myfile << (i+1) << ") R= " << R[i] << " perimeter= " << perimeter << '\n';
    myfile << (i+1) << ") R= " << R[i] << " area     = " << area      << '\n';
  }
    
  myfile.close();
  
}

