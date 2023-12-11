#include "padel.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <fstream>

cv::Point2f Padel::mousePosition = {-1,-1};

void Padel::onMouse(int event, int x, int y, int flag, void *param) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    mousePosition.x = x;
    mousePosition.y = y;
  }
  else if (event == cv::EVENT_RBUTTONDOWN) {
    mousePosition.x = -2;
  }
}

Padel::Padel(std::string filePath)
    : _cap{cv::VideoCapture(filePath)},
      _fileOpened(true)
{
  if (!_cap.isOpened()) {
    std::cerr << "Error: could not open " << filePath << '\n';
    return;
  }

  //Load parameters from file if exists, calculate them otherwise
  std::string paramFile = filePath + "-param.dat";
  std::fstream fin(paramFile, std::ios::in);
  if (fin.is_open()) {
    //TO DO: read from file
  }
  else {
    std::cout << "Debug: Paramfile not present, recalculating\n";
    calculatePerspMat(paramFile);
  }

  calculateFPS(true);
}

Padel::Padel(int camIndex)
  : _cap{cv::VideoCapture(camIndex)},
    _fileOpened{false}
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

  if (seconds == 0 && !_fileOpened) //if a camera is opened
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
  if (_fileOpened)
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
  if (delay == 0) {
    if (_fileOpened)
      delay = 1000/_fps;
    else
      delay = 18;   //higher number means less lag, but less percieved fluidity
  }
  
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

        int k = cv::waitKey(delay);
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

bool Padel::calculatePerspMat(std::string filename) {
  std::string winName = "Click on points indicated in green. Use WASD to move last cross. Right click to remove it";
  cv::namedWindow(winName);
  cv::setMouseCallback(winName, onMouse);
  
  cv::Point2f angles[4]; //angles of the field

  if (_fileOpened) {
    std::cout << "Click on the 4 indicated points. Press 'n' to show another (random) frame. Press 'space' to confirm.\n";
    std::srand(time(0));
    int totalFrame = _cap.get(cv::CAP_PROP_FRAME_COUNT) -2;
    int key;
    int count = 0;
    do {
      key = -1;
      cv::Mat original;
      _cap.set(cv::CAP_PROP_POS_FRAMES, rand() % totalFrame);
      _cap.read(original);

      do {
        cv::Mat frame = original.clone();

        if (mousePosition.x == -2) {  //right click on mouse
          if (count > 0)
            --count;
          mousePosition = {-1,-1};
        }
        else if (mousePosition.x != -1) {  //if mouse has been clicked
          if (count >= 4) {   //if user is trying to add fifht point, break
            std::cout << "Degub: Fifth point -> Breaking\n";
            key = ' ';
            break;
          }
          angles[count] = mousePosition;
          ++count;
          mousePosition = {-1,-1};
        }
        
        /*  //This uses arrows to move the cross around, but its "implementation specific and depends on used backend"
        key = cv::waitKeyEx(5);

        if (count != 0) {
          //move the last cross
          if (key == 2424832)       //left
            angles[count-1].x--;
          else if (key == 2555904)  //right
            angles[count-1].x++;
          else if (key == 2490368)  //up
            angles[count-1].y--;
          else if (key == 2621440)  //down
            angles[count-1].y++;
        }
        */

        //Instead I'll use WASD to move the cross
        key = cv::waitKey(5);

        if (count > 0) {
          //move the last cross
          if (key == 'a')       //left
            angles[count-1].x--;
          else if (key == 'd')  //right
            angles[count-1].x++;
          else if (key == 'w')  //up
            angles[count-1].y--;
          else if (key == 's')  //down
            angles[count-1].y++;
        }

        //draw crosses on frame
        for (int i = 0; i < count; ++i) {
            cv::drawMarker(frame, angles[i], {0,0,255}, cv::MARKER_TILTED_CROSS, 20, 1);
        }

        int scale = 10;
        int offset = 20;
        cv::rectangle(frame, cv::Rect(cv::Point{offset, offset}, cv::Point{10*scale+offset, 20*scale+offset}), {129,94,61}, -1);
        cv::line(frame, {offset        ,      3  *scale+offset }, {10*scale+offset,        3  *scale+offset }, {255,255,255}, 1); //horizontal
        cv::line(frame, {offset        ,     17  *scale+offset }, {10*scale+offset,       17  *scale+offset }, {255,255,255}, 1); //horizontal
        cv::line(frame, {5*scale+offset,(int)(2.5*scale+offset)}, { 5*scale+offset, (int)(17.5*scale+offset)}, {255,255,255}, 1); //vertical
        cv::line(frame, {offset        ,     10  *scale+offset }, {10*scale+offset,       10  *scale+offset }, {0  ,0  ,255}, 2); //net
        cv::drawMarker(frame, {offset, offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
        cv::drawMarker(frame, {10*scale+offset, offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
        cv::drawMarker(frame, {offset, 17*scale+offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
        cv::drawMarker(frame, {10*scale+offset, 17*scale+offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
        //cv::putText(frame, "Select points indicated in green", {offset/2, offset/2}, 0, 0.5, {0,255,0}, 1);

        cv::imshow(winName, frame);
      }
      while (key != 'n' && (key != ' ' || count < 4));
    }
    while (key != ' ');

    cv::destroyWindow(winName);

    //sort angle points
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3-i; ++j) {
        if (angles[j].y > angles[j+1].y)
          std::swap(angles[j], angles[j+1]);
      }
    }
    if (angles[0].x > angles[1].x)
      std::swap(angles[0], angles[1]);
    if (angles[2].x < angles[3].x)
      std::swap(angles[2], angles[3]);
    
    std::cout << "Debug: Angles = \n";
    for (int i = 0; i < 4; ++i) {
      std::cout << angles[i] << "  -  ";
    }
    std::cout << '\n';

    //calculate matrix
    float zoom = 50;
    cv::Point2f rect[4] = { {0,0}, {10*zoom,0}, {10*zoom,17*zoom}, {0,17*zoom} };
    _perspMat = cv::getPerspectiveTransform(angles, rect);

    //save matrix to file
    std::fstream fout(filename, std::ios::out);
    fout << _perspMat;
    fout.close();

    if (_fileOpened)
      _cap.set(cv::CAP_PROP_POS_FRAMES, 0);
  }

  return false;
}

void Padel::calculateFPS(bool file)
{
  _fps = _cap.get(cv::CAP_PROP_FPS);

  if (_fps != 0)
    return;

  //if the property is zero, try to calculate it in a different way
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
