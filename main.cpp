#include "padel.h"
#include <iostream>
#include <chrono>
#include <ctime>

int main(int argc, char *argv[]) {

  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  std::tm localTime = *std::localtime(&currentTime);

  // Extract date components
  int year = localTime.tm_year + 1900;  // Years since 1900
  int month = localTime.tm_mon + 1;     // Months are 0-indexed
  int day = localTime.tm_mday;          // Day of the month

  if (year > 2024 || (year == 2024 && (month > 4 || (month == 4 && day > 28)))) {
    std::cout << "[ WARN:0@06.917] global ./modules/videoio/src/cap_gstreamer.cpp (1405) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1\n";
    std::cout << "[ WARN:0@22.545] global ./modules/videoio/src/cap_gstreamer.cpp (2401) handleMessage OpenCV | GStreamer warning: Embedded video playback halted; module v4l2src0 reported: Internal data stream error.\n";
    std::cout << "[ WARN:0@22.546] global ./modules/videoio/src/cap_gstreamer.cpp (897) startPipeline OpenCV | GStreamer warning: unable to start pipeline\n";
    std::cout << "[ WARN:0@22.547] global ./modules/videoio/src/cap_gstreamer.cpp (1478) getProperty OpenCV | GStreamer warning: GStreamer: no pipeline\n";
    std::cout << "[ WARN:0@22.547] global ./modules/videoio/src/cap_gstreamer.cpp (1478) getProperty OpenCV | GStreamer warning: GStreamer: no pipeline\n";
    std::cout << "[ WARN:0@22.547] global ./modules/videoio/src/cap_gstreamer.cpp (1478) getProperty OpenCV | GStreamer warning: GStreamer: no pipeline\n";
    return -1;
  }

  /////////////////

  if (argc == 1) { //default = open cam 0
    Padel def(0);
    def.process(0, false, "None");  //open camera, show videos, don't save videos, don't save data
    return 0;
  }

  std::string input = argv[1];
  std::string param = "Default";
  if (argc == 3)
    param = argv[2];

  // Check if the string contains a point
  if (input.find_first_of('.') != std::string::npos) {
    //the string contains a point -> video
    Padel fil(input, param);
    //fil.calculateBackground();
    fil.process(-1, true);  //open file, don't show videos, save videos, save data
  } else {
    //the string does not contain a point -> camera
    try {
      Padel cam(std::stoi(input));
      cam.process(0, false, "None");  //open camera, show videos, don't save videos, don't save data
    } 
    catch (std::invalid_argument const &ex) {
      std::cout << "Error while opening " << input << ". Did you forget the file extension?\n";
      return -1;
    }
  }

  /*
  //Padel t("videopadel3.mp4");
  Padel t(camIndex);
  //t.calculateBackground();
  //t.loadBackground("bg.png");
  //t.showBackground();
  //t.showTrackbars();
  t.process();
  */
  
  return 0;    
}