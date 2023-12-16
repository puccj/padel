#include "padel.h"
#include <iostream>

int main() {
  Padel t(0);
  //Padel t(0);
  //t.calculateBackground();
  //t.loadBackground("bg.png");
  //t.showBackground();
  t.showTrackbars();
  t.process();
  
  std::cout << "FINE\n";
  return 0;
}