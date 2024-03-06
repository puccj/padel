#ifndef PADEL_H
#define PADEL_H

#include <iostream>
#include <opencv2/videoio.hpp>

class Padel
{
  static cv::Point2f mousePosition;
  static void onMouse(int event, int x, int y, int flag, void* param);

  //parameters:
  int thresholdValue = 46;
  int dilationValue = 6;
  int minAreaValue = 50;
  int consecutiveValue = 1;
  double learningRateValue = -1;

  enum class bgSubMode{KNN, MOG2};

  cv::VideoCapture _cap;
  bool _fileOpened;
  std::string _camname;
  cv::Mat _perspMat;  //perspective matrix
  double _fps;
  cv::Mat _background;
  cv::Point2f _lastPos[4];  //positions of the 4 player in the last frame

 public:
  /// @brief Open a video from a file
  /// @param filename Filename of the video to open
  /// @param paramPath Filepath containing the parameter of the camera. If the file does not exists, parameters will be re-calculated
  Padel(std::string filename, std::string paramPath = "Default"); //open video file
  Padel(int camIndex, std::string paramPath = "Default");         //open a camera

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

  /// @brief Analyze video stream producing data of player positions and/or show and/or save videos: boxed player, FGmask and 2D field graphics.
  /// @param delay Delay between each shown frame. If 0 (default), the video is shown with original fps. If negative the video is not shown at all.
  /// @param outputVideo Path where to save video. If set to "None", no output video will be produced
  /// @param outputData Filename where to save data. If set to "None", no output data will be produced.
  /// If set to "Default" (default), the video is saved as "<camIndex>-data.dat" or "<filename>-data.dat".
  /// @param mode Mode use to get the foreground Mask. Only used if background is not pre-set
  /// @param removeShadows Whether to consider shadows to create the box. Only used if background is not pre-set
  /// @return false if an error occurs
  bool process(int delay = 0, std::string outputVideo = "Default", std::string outputData = "Default", bgSubMode mode = bgSubMode::KNN, bool removeShadows = true);

  //save data in a file
  //void saveData(std::string outputFile = "output.dat");

  //generates heatmap from the data
  void createHeatmap();

 private:
  //Load parameters (perspective matrix and fps) from file if exists, calculate them otherwise
  void loadParam(const std::string& paramFile);
  bool calculatePerspMat(const std::string& filename);
  void calculateFPS();
  
  //draw a 2D scheme of a padel field
  void draw2DField(cv::Mat mat, int offset, int zoom);
  
  //Find the best n points among a vector, based on their proximity to given n points
  std::vector<cv::Point2f> findBest_Position(const std::vector<cv::Point2f>& points, const std::vector<cv::Point2f>& given);

  //find the best n  points among a vector, based on their content color
  void findBest_Color();
};


#endif //PADEL_H
