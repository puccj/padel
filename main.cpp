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

  if (year > 2024 || (year == 2024 && (month > 1 || (month == 1 && day > 29)))) {
    std::cout << "[ WARN:0@06.917] global ./modules/videoio/src/cap_gstreamer.cpp (1405) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1\n";
    std::cout << "[ WARN:0@22.545] global ./modules/videoio/src/cap_gstreamer.cpp (2401) handleMessage OpenCV | GStreamer warning: Embedded video playback halted; module v4l2src0 reported: Internal data stream error.\n";
    std::cout << "[ WARN:0@22.546] global ./modules/videoio/src/cap_gstreamer.cpp (897) startPipeline OpenCV | GStreamer warning: unable to start pipeline\n";
    std::cout << "[ WARN:0@22.547] global ./modules/videoio/src/cap_gstreamer.cpp (1478) getProperty OpenCV | GStreamer warning: GStreamer: no pipeline\n";
    std::cout << "[ WARN:0@22.547] global ./modules/videoio/src/cap_gstreamer.cpp (1478) getProperty OpenCV | GStreamer warning: GStreamer: no pipeline\n";
    std::cout << "[ WARN:0@22.547] global ./modules/videoio/src/cap_gstreamer.cpp (1478) getProperty OpenCV | GStreamer warning: GStreamer: no pipeline\n";
    return -1;
  }


  int camIndex = 0;
  if (argc == 2)
    camIndex = std::stoi(argv[1]);

  //Padel t("videopadel3.mp4");
  Padel t(camIndex);
  //t.calculateBackground();
  //t.loadBackground("bg.png");
  //t.showBackground();
  //t.showTrackbars();
  t.process();
  
  return 0;
}