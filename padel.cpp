#include "padel.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <fstream>

Padel::Padel(std::string filePath) 
  : _data {new Positions[_totalFrame]}, 
    _cap{cv::VideoCapture(filePath)},
    _totalFrame{(int) _cap.get(cv::CAP_PROP_FRAME_COUNT)}
{
  if (!_cap.isOpened()) {
    std::cerr << "Error: could not open " << filePath << '\n';
    return;
  }

  //To do: Load parameters from file if exists, calculate them otherwise

  calculateFPS(true);
}

Padel::Padel(int camIndex)
  : _cap{cv::VideoCapture(camIndex)},
    _totalFrame{0}
{
  if (!_cap.isOpened()) {
    std::cerr << "Error while opening camera " << camIndex << '\n';
    return;
  }
  calculateFPS(false);
}

void Padel::showTrackbars() {
  std::string winName = "Trackbars";
  cv::namedWindow(winName, cv::WINDOW_NORMAL);
  cv::createTrackbar("Threshold", winName, &thresholdValue, 100);
  cv::createTrackbar("Dilation (detection)",winName, &dilationValue, 20);
  cv::createTrackbar("Min Area", winName, &minAreaValue, 400);
}

void Padel::showBackground(int delay, std::string winName) {
  if (winName == "Default")
    winName = "Background";
  cv::imshow(winName, _background);
  cv::waitKey(delay);
}

void Padel::calculateBackground(double seconds, double weight) {
  std::cout << "Getting color background... ";
  
  int frames = seconds * _fps;

  if (seconds == 0 && _totalFrame == 0) //if a camera is opened
    frames = 5 * _fps;   //5 second as default

  cv::Mat firstFrame;
  _cap.read(firstFrame);
  
  cv::Mat acc(firstFrame.size(), CV_32FC3);

  int takenFrames = 1;
  while (true) {
    cv::Mat thisFrame;
    _cap.read(thisFrame);
    ++takenFrames;
    if (thisFrame.empty())  //if video ends
      break;
    if (takenFrames > frames && frames != 0)
      break;

    cv::accumulateWeighted(thisFrame, acc, weight);

  }

  //if cap is a file, set to the beggining of the video
  if (_totalFrame != 0)
    _cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  std::cout << "     Done\n";
  _background = acc;

  cv::imshow("Background", _background/255);
}

bool Padel::loadBackground(std::string filename) {
  _background = cv::imread(filename);
  if (_background.empty()) {
    std::cerr << "Error while loading the background from " << filename << '\n';
    return false;
  }
  return true;
}

bool Padel::process(std::string outputFile, int delay, bgSubMode mode, bool removeShadows) {
  if (delay == 0)
    delay = 1000/_fps;
  
  cv::Mat frame, fgMask, bg;
  std::fstream fout(outputFile, std::ios::out);

  if (_background.empty()) {
    std::cout << "Debug: without BG\n";
    
    //create Background Subtractor objects
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    if (mode == bgSubMode::KNN)
      pBackSub = cv::createBackgroundSubtractorKNN();
    else if (mode == bgSubMode::MOG2)
      pBackSub = cv::createBackgroundSubtractorMOG2();
    else {
      std::cerr << "ERROR: process was called without a BG and an unknown mode for calculation has been given\n";
      fout.close();
      return false;
    }
  
    while (true) {
      _cap >> frame;;
      if (frame.empty()) {
        std::cout << "End of video\n";
        fout.close();
        return true;
      }
  
      //Foreground detection and update the background model
      pBackSub->apply(frame, fgMask, learningRateValue);
      if (removeShadows)
        cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);           //thresholding to remove shadows
      cv::dilate(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), dilationValue); //dilate the image (no inside dark regions)
      cv::erode(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), dilationValue);  //erode the image (make contours have the original size)


      //Contours drawing
      std::vector<cv::Mat> contours;
      cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      //if (drawContours)
      cv::drawContours(frame, contours, -1, {255,0,0}, 1);
    
      for (auto contour : contours) {
        //skip if area of contour is too little (it's probably noise)
        if (cv::contourArea(contour) < minAreaValue)
          continue;
      
        //otherwise draw a rectangle
        cv::Rect r = cv::boundingRect(contour);
        //if (drawRects)
        cv::rectangle(frame, r, {0,255,0}, 2);

        //Find position of feet and save on file
        cv::Point feet = {r.x + r.width/2, r.y + r.height};
        cv::circle(frame, feet, 1, {255,0,255}, 3, cv::LINE_AA);
        fout << feet << ' ';
      }

      if (delay > 0) { //Only show progress if delay >= 0
        //pBackSub->getBackgroundImage(bg);
        imshow("Frame", frame);
        imshow("FG Mask", fgMask);
        // imshow("BG", bg);

        int k = cv::waitKey();
        if (k == 'q' || k == 27) {
          fout.close();
          return true;
        }
      }

      //New frame -> new line on file
      fout << '\n';
    }
  }
  else { //BG not empty 
    std::cout << "Debug: With BG\n";

    cv::cvtColor(_background, bg, cv::COLOR_BGR2GRAY);
    bg.convertTo(bg, CV_8U);    //needed to confront it with frame

    while (true) { 
      _cap.read(frame);
      if (frame.empty()) {
        std::cout << "End of video\n";
        return true;
      }

      cv::Mat fgMask(frame.size(), CV_32FC1);  //gray frame
      cv::cvtColor(frame, fgMask, cv::COLOR_BGR2GRAY);

      //Foreground detection
      cv::absdiff(fgMask, bg, fgMask);                                        //subtract background from image
      cv::threshold(fgMask, fgMask, thresholdValue, 255, cv::THRESH_BINARY);  //thresholding to convert to binary
      cv::dilate(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), dilationValue); //dilate the image (no inside dark regions)
      cv::erode(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), dilationValue);  //erode the image (make contours have the original size)
    
      //Contours drawing
      std::vector<cv::Mat> contours;
      cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      //if (drawContours)
      cv::drawContours(frame, contours, -1, {255,0,0}, 1);
    
      for (auto contour : contours) {
        //skip if area of contour is too little (it's probably noise)
        if (cv::contourArea(contour) < minAreaValue)
          continue;
      
        //otherwise draw a rectangle
        cv::Rect r = cv::boundingRect(contour);
        //if (drawRects)
        cv::rectangle(frame, r, {0,255,0}, 2);

        //Find position of feet and save on file
        cv::Point feet = {r.x + r.width/2, r.y + r.height};
        cv::circle(frame, feet, 1, {255,0,255}, 3, cv::LINE_AA);
        fout << feet << ' ';
      }

      if (delay > 0) { //Only show progress if delay >= 0
        //pBackSub->getBackgroundImage(bg);
        imshow("Frame", frame);
        imshow("FG Mask", fgMask);
        // imshow("BG", bg);

        int k = cv::waitKey();
        if (k == 'q' || k == 27) {
          fout.close();
          return true;
        }
      }
      
      //New frame -> new line on file
      fout << '\n';
    }
  } //end else
  
  fout.close();
  return true;
}


// -- Private methods -- //


void Padel::calculateFPS(bool file) {
  _fps = _cap.get(cv::CAP_PROP_FPS);

  //if the property is zero, try to calculate it in a different way
  if (_fps != 0)
    return;

  std::cout << "Calculating fps...";
  int num_frames = 60; //number of frames to capture
  time_t start, end;
  cv::Mat frame;
  
  time(&start);
  for(int i = 0; i < num_frames; i++)
    _cap >> frame;
  time(&end);

  double seconds = difftime (end, start);
  std::cout << "     Done (" << num_frames/seconds << " fps)\n";

  if (file)
    _cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  _fps = num_frames / seconds;
}

