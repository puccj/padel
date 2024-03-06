#include "padel.h"
#include <iostream>
#include <chrono>
#include <ctime>

//Get parser options (return 0 if not find)
char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool stob(std::string input) {
  if (input == "false")
    return false;
  return true;
}

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

  // if (argc == 1) { //default = open cam 0
  //   Padel def(0);
  //   def.process(0, false, "None");  //open camera, show videos, don't save videos, don't save data    // TO DO: change that into save when 'S' pressed.
  //   return 0;
  // }

  std::string inputVideo;
  int camera = 0;
  bool cameraOpen; //true for cameras, false for input videos
  std::string param = "Default";
  bool showVideo = true;
  std::string output = "Default";
  std::string data = "Default";

  //check if arguments start with '-'
  bool optionPresent = false;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-') {
      optionPresent = true;
      break;
    }
  }

  try {
    if (!optionPresent) {
      if (argc == 1) {
        cameraOpen = true;
      }
      else if (argc == 2) {
        // Check if the string contains a point
        if (std::string(argv[1]).find_first_of('.') != std::string::npos) {
          inputVideo = argv[1];
          cameraOpen = false;
          showVideo = false;
        }
        else {
          camera = std::stoi(argv[1]);
          cameraOpen = true;
        }
      }
      else if (argc == 3) {  //if 3 arguments with no options
        if (std::string(argv[1]).find_first_of('.') != std::string::npos) {
          inputVideo = argv[1];
          cameraOpen = false;
        }
        else {
          camera = std::stoi(argv[1]);
          cameraOpen = true;
        }
        param = argv[2];
      }
    }
    else {      
      //Check parameters

      if (std::find(argv, argv+argc, std::string("-h")) != argv+argc ||
          std::find(argv, argv+argc, std::string("--help")) != argv+argc) 
      {
        //Print help message
        std::cout << "\n--Padel 0.2.0--\n\n";
        std::cout << "Usage:\n";
        std::cout << "  padel-0.2.0  (open default cam 0 with default parameters)   OR\n";
        std::cout << "  padel-0.2.0 <inputFile>                                     OR\n";
        std::cout << "  padel-0.2.0 <camIndex>                                      OR\n";
        std::cout << "  padel-0.2.0 <inputFile> <paramFile>                         OR\n";
        std::cout << "  padel-0.2.0 <camIndex> <paramFile>                          OR\n";
        std::cout << "  padel-0.2.0 [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help \t\t Show this help page\n";
        std::cout << "  -f, --file, <path> \t Path to the input file path to be processed.\n";
        std::cout << "  -c, --camera, <index>  Index of the camera device to stream. Default: 0\n";
        std::cout << "  -p, --param, <path> \t Path to the parameter file to use/create (if non-existent).\n";
        std::cout << "                      \t   Default: ./parameters/<inputPath>.dat\n";
        std::cout << "  --show <true/false> \t Whether to show video outputs or not. Default: false for videos, true for cameras\n";
        std::cout << "  --output <dir> \t Directory where to save video outputs. Use 'None' to avoid saving them.\n";
        std::cout << "                 \t   Default: ./ToBeUploaded/\n";
        std::cout << "  --data <dir> \t\t Directory where to save raw data. Use 'None' to avoid saving them.\n";
        std::cout << "               \t\t   Default: ./data/<inputPath>.dat\n\n";

        std::cout << "Notes:\n";
        std::cout << "  In order to ensure the corrent video and data saving, make sure that their respectively output folder(s)\n";
        std::cout << "    exist before running the executable.\n";
        std::cout << "  The output videos will be saved in the desired or default directory as [name]-2D, [name]-box, [name]-BW\n";
        std::cout << "    where [name] is <index> or <inputFile> with '/' and '\\' substituted with '-'.\n";
        return 0;
      }

      char* filename = getCmdOption(argv, argv+argc, "-f");
      if (!filename) {
        filename = getCmdOption(argv, argv+argc, "--file");
      }
      if (filename) {
        cameraOpen == false;
        inputVideo = filename;
      }

      char* cam_c = getCmdOption(argv, argv+argc, "-c");
      if (!cam_c) {
        cam_c = getCmdOption(argv, argv+argc, "--camera");
      }
      if (cam_c) {
        cameraOpen = true;
        camera = std::stoi(cam_c);
      }

      if (filename && cam_c) { //if both -f and -c options specified
        std::cout << "Error: You can't specify to open both a camera (-c) and a file (-f)\n";
        return -2;
      }
      if (!filename && !cam_c) {  //if neither -f nor -c are specified
        std::cout << "Warning: Neither camera (-c) nor file (-f) was specified. Using default camera index 0\n";
        cameraOpen = true;
        camera = 0;
      }

      char* param_c = getCmdOption(argv, argv+argc, "-p");
      if (!param_c) {
        param_c = getCmdOption(argv, argv+argc, "--param");
      }
      if (param_c)
        param = param_c;
    
      char* show_c = getCmdOption(argv, argv+argc, "--show");
      if (show_c)
        showVideo = stob(show_c);
      else
        showVideo = cameraOpen;   //default: show video for a camera, don't show it for file.
      
      char* output_c = getCmdOption(argv, argv+argc, "--output");
      if (output_c)
        output = output_c;

      char* data_c = getCmdOption(argv, argv+argc, "--data");
      if (data_c)
        data = data_c;
    }

    //Debug
    // std::cout << "inputVideo = " << inputVideo << '\n';
    // std::cout << "camera = " << camera << '\n';
    // std::cout << "cameraOpen = " << cameraOpen << '\n';
    // std::cout << "param = " << param << '\n';
    // std::cout << "showVideo = " << showVideo << '\n';
    // std::cout << "output = " << output << '\n';
    // std::cout << "data = " << data << '\n';
  }
  catch (std::invalid_argument const &e) {
    std::cout << "Error while opening camera (index is not a number). If you intended to open a video, remember to include the file extension\n";
    return -1;
  }

  int delay = -1;
  if (showVideo)
    delay = 0;

  if (cameraOpen) {
    Padel cam(camera, param);
    cam.process(delay, output, data);
  }
  else {
    Padel fil(inputVideo, param);
    fil.process(delay, output, data);
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