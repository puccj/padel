#include "padel.h"
#include <iostream>

int main() {
  //Padel t("videopadel3.mp4");
  Padel t(0);
  //t.calculateBackground();
  //t.loadBackground("bg.png");
  //t.showBackground();
  //t.showTrackbars();
  t.process();
  
  std::cout << "FINE\n";
  return 0;
}