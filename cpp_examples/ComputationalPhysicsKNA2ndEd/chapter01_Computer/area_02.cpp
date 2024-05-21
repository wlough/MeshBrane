#include <iostream>
using namespace std;

int main(){
  double PI = 3.1415926535897932;
  double R[10];
  double area, perimeter;
  int i;
  
  R[0] = 2.28;
  for(i=1;i<10;i++){
    R[i] = R[i-1] + 1.0;
  }

  for(i=0;i<10;i++){
    perimeter = 2.0*PI*R[i];
    area      = PI*R[i]*R[i];
    cout << (i+1) << ") R= " << R[i] << " perimeter= " << perimeter << '\n';
    cout << (i+1) << ") R= " << R[i] << " area     = " << area      << '\n';
  }

}

      
