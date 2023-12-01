#ifndef PADEL_H
#define PADEL_H

#include <iostream>
#include <opencv2/videoio.hpp>

class Padel
{
  //parameters:
  int thresholdValue = 46;
  int dilationValue = 6;
  int minAreaValue = 50;
  int consecutiveValue = 1;
  double learningRateValue = -1;
  

  enum class bgSubMode{KNN, MOG2};
  struct Positions {
    //store the position of the players
    cv::Point2d pos[4];
  };

  cv::VideoCapture _cap;
  int _totalFrame;  //total number of frames in the video, 0 if a camera is opened (real time exec)
  Positions* _data;
  double _fps;
  cv::Mat _background;

 public:
  Padel(std::string filePath);  //open a video from a file
  Padel(int camIndex);          //open a camera

  void showTrackbars();

  //show the background for 'delay' milliseconds (0 = untill a key is pressed)
  void showBackground(int delay = 0, std::string winName = "Default");

  //calculate the starting background using given number of seconds
  //(0(default) = entire video lenght for a file, 5 seconds for a camera)
  void calculateBackground(double seconds = 0, double weight = 0.005);

  /// @brief Load an image to be uses as background
  /// @param filename Filepath of the input image
  /// @return false if an error occurs
  bool loadBackground(std::string filename);

  /// @brief Produces data of positions for the current frame and optionally show video with box around each person
  /// @param outputFile File where to save data
  /// @param delay Delay between each shown frame. If 0 (default), the video is shown with original fps. If negative the video is not shown at all
  /// @param mode Mode use to get the foreground Mask. Only used if background is not pre-set
  /// @param removeShadows Whether to consider shadows to create the box. Only used if background is not pre-set
  /// @return false if an error occurs
  bool process(std::string outputFile = "output.dat", int delay = 0, bgSubMode mode = bgSubMode::KNN, bool removeShadows = true);

  //save data in a file
  void saveData(std::string outputFile = "output.dat");

  //generates heatmap from the data
  void createHeatmap();

 private:
  void calculateFPS(bool file);

};


#endif //PADEL_H
