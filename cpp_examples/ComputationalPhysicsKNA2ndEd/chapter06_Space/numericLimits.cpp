// Compile with:
// g++ -std=c++11 numericLimits.cpp -o n ; ./n
#include <iostream>
#include <limits>

using namespace std;

int main(){
  double MCHEPS,DWARF;

  MCHEPS = numeric_limits<double>::epsilon();
  DWARF  = numeric_limits<double>::min    ();
  cout << "MCHEPS = " << MCHEPS/2.0 << endl;
  cout << "DWARF  = " << DWARF      << endl;
  
  // More:
  cout << "------------------------------------------------------------" << endl;
  cout << "double:"                                                      << endl;
  cout << "double: # Bytes      = " << sizeof        (double)            << endl;
  cout << "double: Maximum value= " << numeric_limits<double>::max    () << endl;
  cout << "double: Minimum value= " << numeric_limits<double>::min    () << endl;
  cout << "double: Lowest  value= " << numeric_limits<double>::lowest () << endl; // C++11
  cout << "double: epsilon      = " << numeric_limits<double>::epsilon() << endl;
  cout << "------------------------------------------------------------" << endl;
  cout << "float :"                                                      << endl;
  cout << "float : # Bytes      = " << sizeof        (float )            << endl;
  cout << "float : Maximum value= " << numeric_limits<float >::max    () << endl;
  cout << "float : Minimum value= " << numeric_limits<float >::min    () << endl;
  cout << "float : Lowest  value= " << numeric_limits<float >::lowest () << endl; // C++11
  cout << "float : epsilon      = " << numeric_limits<float >::epsilon() << endl;
  cout << "------------------------------------------------------------" << endl;
  cout << "integers:"                                                    << endl;
  cout << "int   : # Bytes      = " << sizeof        (int   )            << endl;
  cout << "int   : Maximum value= " << numeric_limits<int   >::max    () << endl;
  cout << "long  : # Bytes      = " << sizeof        (long  )            << endl;
  cout << "long  : Maximum value= " << numeric_limits<long  >::max    () << endl;
  cout << "short : # Bytes      = " << sizeof        (short )            << endl;
  cout << "short : Maximum value= " << numeric_limits<short >::max    () << endl;
  cout << "------------------------------------------------------------" << endl;
  cout << "long double:"                                                           << endl;
  cout << "long double: # Bytes      = " << sizeof        (long double)            << endl;
  cout << "long double: Maximum value= " << numeric_limits<long double>::max    () << endl;
  cout << "long double: Minimum value= " << numeric_limits<long double>::min    () << endl;
  cout << "long double: Lowest  value= " << numeric_limits<long double>::lowest () << endl; // C++11
  cout << "long double: epsilon      = " << numeric_limits<long double>::epsilon() << endl;
}
