#include "padel.h"
#include "padel-utils.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <fstream>
#include <algorithm>

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

Padel::Padel(std::string filename, std::string paramPath)
    : _cap{cv::VideoCapture(filename)},
      _fileOpened{true},
      _camname{filename.substr(0, filename.find_last_of('.'))},  // Remove the file extension
      _lastPos{{5,10},{5,10},{5,10},{5,10}}
{
  if (!_cap.isOpened()) {
    throw std::runtime_error{"PADEL ERROR 01: could not open " + filename};
    return;
  }

  std::replace(_camname.begin(), _camname.end(), '/', '-');
  std::replace(_camname.begin(), _camname.end(), '\\', '-');
  
  if (paramPath == "Default")
    paramPath = "./parameters/" + _camname + ".dat";
  loadParam(paramPath);
}


Padel::Padel(int camIndex, std::string paramPath)
    : _cap{cv::VideoCapture(camIndex)},
      _fileOpened{false},
      _camname{std::to_string(camIndex)},
      _lastPos{{5,10},{5,10},{5,10},{5,10}}
{
  if (!_cap.isOpened()) {
    throw std::runtime_error{"PADEL ERROR 02 while opening camera " + std::to_string(camIndex)};
    return;
  }
  _cap.set(cv::CAP_PROP_FPS, 20);

  if (paramPath == "Default")
    paramPath = "./parameters/" + _camname + ".dat";
  loadParam(".//parameters/" + _camname + ".dat");
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
}

bool Padel::loadBackground(std::string filename) {
  _background = cv::imread(filename);
  if (_background.empty()) {
    std::cerr << "PADEL WARNING 99: Could not load the background image " << filename << ". FGmask will be calculated without using a background \n";
    return false;
  }
  return true;
}

bool Padel::process(int delay, bool saveVideo, std::string outputFile, bgSubMode mode, bool removeShadows) {
  if (outputFile == "Default")
    outputFile = "./data/ " + _camname + "-data.dat";

  if (delay == 0) {
    if (_fileOpened)
      delay = 1000/_fps;
    else
      delay = 15;   //higher number means less lag, but less percieved fluidity
  }
  else if (delay < 0)
    std::cout << "Analyzing...";
  
  //Needed Mats
  int zoom = 40;
  int offset = 2*zoom;
  cv::Mat field(20*zoom + 2*offset, 10*zoom + 2*offset, CV_8UC3, {209,186,138});  //2D field graphics
  cv::Mat frame, fgMask, bg, original;
  cv::Ptr<cv::BackgroundSubtractor> pBackSub; //BG subtractor
  
  if (_background.empty()) {
    //std::cout << "Debug: without BG\n";
    if (mode == bgSubMode::KNN)
      pBackSub = cv::createBackgroundSubtractorKNN();
    else if (mode == bgSubMode::MOG2)
      pBackSub = cv::createBackgroundSubtractorMOG2();
    else {
      throw std::runtime_error{"PADEL ERROR 03: process was called without a BG and an unknown mode for calculation has been given."};
      return false;
    }
  }
  else {
    //std::cout << "Debug: With BG\n";
    cv::cvtColor(_background, bg, cv::COLOR_BGR2GRAY);
    bg.convertTo(bg, CV_8U);    //needed to confront it with frame
  }
  
  //Objects to save data and videos
  std::fstream fout;
  if (outputFile != "None")
    fout.open(outputFile, std::ios::out);

  _cap >> frame;
  //cv::VideoWriter wtrOriginal(_camname + "original.avi", cv::VideoWriter::fourcc('M','J','P','G'), _fps, frame.size(), true);
  cv::VideoWriter wtrFrame("./ToBeUploaded/" + _camname + "-box.mp4", cv::VideoWriter::fourcc('m','p','4','v'), _fps, frame.size(), true);
  cv::VideoWriter wtrMask ("./ToBeUploaded/" + _camname + "-BW.mp4",  cv::VideoWriter::fourcc('m','p','4','v'), _fps, frame.size(), false);
  cv::VideoWriter wtrField("./ToBeUploaded/" + _camname + "-2D.mp4",  cv::VideoWriter::fourcc('m','p','4','v'), _fps, field.size(), true);


  /////////// ----- Main loop (1 iteration per frame) ----- ///////////
  while (true) {
    time_t start, end;  //to know how long each frame take to be analyzed
    time(&start);

    _cap >> frame;
    if (frame.empty()) {
      if (delay > 0)
        std::cout << "End of video\n";
      break;
    }

    //store founded points for this frame
    int maxPoint = 8;   //maybe I'll change it
    std::vector<cv::Point2f> thisPos;

    original = frame.clone(); //save original frame to create output video

    //Foreground detection and update the background model
    if (_background.empty()) {
      pBackSub->apply(frame, fgMask, learningRateValue);
      if (removeShadows)
        cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);           //thresholding to remove shadows
    }
    else {
      fgMask = cv::Mat(frame.size(), CV_32FC1);  //gray frame
      cv::cvtColor(frame, fgMask, cv::COLOR_BGR2GRAY);
      cv::absdiff(fgMask, bg, fgMask);                                        //subtract background from image
      cv::threshold(fgMask, fgMask, thresholdValue, 255, cv::THRESH_BINARY);  //thresholding to convert to binary
    }
    cv::dilate(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), dilationValue); //dilate the image (no inside dark regions)
    cv::erode(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), dilationValue);  //erode the image (make contours have the original size)

    //draw the 2D field graphics
    field = cv::Scalar{209, 186, 138};
    cv::rectangle(field, cv::Rect(cv::Point{offset, offset}, cv::Point{10*zoom+offset, 20*zoom+offset}), {129,94,61}, -1);
    cv::line(field, {offset        ,     3  *zoom+offset }, {10*zoom+offset,        3  *zoom+offset }, {255,255,255}, 1); //horizontal
    cv::line(field, {offset        ,    17  *zoom+offset }, {10*zoom+offset,       17  *zoom+offset }, {255,255,255}, 1); //horizontal
    cv::line(field, {5*zoom+offset,(int)(2.7*zoom+offset)}, { 5*zoom+offset, (int)(17.3*zoom+offset)}, {255,255,255}, 1); //vertical
    cv::line(field, {offset        ,    10  *zoom+offset }, {10*zoom+offset,       10  *zoom+offset }, {0  ,0  ,255}, 2); //net

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

      //Find position of feet
      cv::Point2f feet = {(float)(r.x + r.width/2), (float)(r.y + r.height)};
      cv::circle(frame, feet, 1, {255,0,255}, 3, cv::LINE_AA);

      //calculate perspective of point and save them
      cv::Point3f homogeneous; //= warp.inv() * point;
      cv::perspectiveTransform(cv::Mat(1, 1, CV_32FC2, &feet), cv::Mat(1, 1, CV_32FC2, &homogeneous), _perspMat);
      cv::Point2f result(homogeneous.x, homogeneous.y);  // Drop the z=1 to get out of homogeneous coordinates
      //result is between (0,0) and (10,20)
      result;

      //Draw all points in the graphic and output all of them
      circle(field, result*zoom+offset, 1, {185,185,185}, 3, cv::LINE_AA);
      if (outputFile != "None")
        fout << result << ' ';

      //Calculate and draw with a different color only the best 4 ones
      // 1. Don't add point if it's outside of field
      if (result.x < 0 || result.x > 10 || result.y < 0 || result.y > 20) {
        //std::cout << "Debug: point not added (" << result << ")\n";
        continue;
      }

      // 2. Don't add point if it's too far from the previous 4 ones
      // bool near = false;
      // for (int i = 0; i < 4; ++i) {
      //   if (cv::norm(result - _lastPos[i]) < 5) {
      //     near = true;
      //     break;
      //   }
      // }
      // if (!near) {
      //   continue;
      // }

      thisPos.push_back(result);
      //std::cout << "Debug: " << thisPos.size() << ". Point " << result << "added\n";
    }
    
    // std::cout << "\nDebug: old points: ";
    // for (int i = 0; i < 4; ++i)
    //   std::cout << _lastPos[i] << "  -  ";
    // std::cout << "\nDebug: new points : ";
    // for (auto p : thisPos)
    //   std::cout << p << "  -  ";
    // std::cout << '\n';

    // //Degug: draw old points:
    // for (int i = 0; i < 4; ++i)
    //   cv::circle(field, _lastPos[i]*zoom+offset, 1, {0,0,0}, 3, cv::LINE_AA);


    auto nPoints = thisPos.size();
    // 3. If the points are enough
    if (nPoints >= 4 && nPoints < maxPoint) {  //(This works even if the detected points are equal to 4)
      //consider all possible combinations of 4 points.
      auto combinations = generateCombinations(thisPos, 4);
      double minDistance = 99999;
      //cv::Point2f temp[4];  //TO DO: maybe I need this temp if I want to keep thisPos to do something  (see C1, C2)

      //c is a single combination (a vector of 4 points)
      for (auto c : combinations) {   //for each combination of 4 points..
        //..calculate every possible pair (permutation among the elements) and take the min distance

        do {
          //get the sum of distance between old (_lastPos) and current (c) 4 points
          double tot = 0;
          for (int i = 0; i < 4; ++i) {
            /*
            double d = cv::norm(_lastPos[i] - c[i]);
            double d2 = distance(_lastPos[i], c[i]);
            double d3 = distance(c[i], _lastPos[i]);

            if (d - d2 > 0.001 || d2 - d3 > 0.001 || d - d3 > 0.001)
              std::cout << "AAAAAAAAAAA  d = " << d << "  -  d2 = " << d2 << '\n';
            */

            tot += cv::norm(_lastPos[i] - c[i]);
          }

          //if a smaller distance is found, save it and the 4 points (keeping the order!)
          if (tot < minDistance) {
            minDistance = tot;
            for (int i = 0; i < 4; ++i) {
              thisPos[i] = c[i];
              //temp[i] = c[i];   // (C1)
            }
          }

        } while (std::next_permutation(c.begin(), c.end(), comparePoints<cv::Point2f>));
      }

      // Update last points
      for (int i = 0; i < 4; ++i)
        _lastPos[i] = thisPos[i];

      // (C2) TO DO: if 2 points are very near (define how much) and there are more than 4 points and using the mean between these 2 points I find a smaller distance,
      //then it probably means that a point/person has been split in 2 in this particular frame. I can use the mean point between the 2 as the real value.
    }
    else if (nPoints > 0 && nPoints < 4) {
      // 4. If there are less then 4 point, keep 1 (or more) point from _lastPos. Which one(s)? The farther(s) from the 3 (or more) points in thisPos
      
      // 4a. nPoints is the number of found points. Check which 'nPoints' points out of the 4 _lastPos are closer to found points.
      
      // auto combinations = generateCombinations(std::vector<cv::Point2f>(_lastPos, _lastPos + 4), nPoints);
      auto combinations = generateCombinations(std::vector<int>{0,1,2,3}, nPoints);
      double minDistance = 99999;

      int indexes[nPoints]; //will contain the indexes of the nPoints _lastPos which are closer to the found ones

      //c is a single combination (a vector of nPoints indexes)
      for (auto c : combinations) {   //for each combination of 4 points..
        //..calculate every possible pair (permutation among the elements) and take the min distance

        do {
          //get the sum of distance between the found (thisPos) and selected (c) nPoints points
          double tot = 0;
          for (int i = 0; i < nPoints; ++i) {
            tot += cv::norm(thisPos[i] - _lastPos[c[i]]);
          }

          //if a smaller distance is found, save it and the points indexes (keeping the order!)
          if (tot < minDistance) {
            minDistance = tot;
            for (int i = 0; i < 4; ++i) {
              indexes[i] = c[i];
              //temp[i] = c[i];   // (C1)
            }
          }

        } while (std::next_permutation(c.begin(), c.end()));
      }

      // 4b. Then, update (set to thisPos) those and keep (set to _lastPos) the others
      
      // Update last points
      for (int i = 0; i < nPoints; ++i) {
        _lastPos[indexes[i]] = thisPos[i];
      }
      //just a temp solution
      // while(thisPos.size() < 4) {
      //   std::cout << "Few points, adding another point\n";
      //   thisPos.push_back({5,10});
      // }
      //end of temp solution

    } //end cases on number of points

    // 6. Draw selected 4 points (with different color)
    cv::circle(field, _lastPos[0]*zoom+offset, 1, {0, 0, 255}, 3, cv::LINE_AA);
    cv::circle(field, _lastPos[1]*zoom+offset, 1, {255, 0, 0}, 3, cv::LINE_AA);
    cv::circle(field, _lastPos[2]*zoom+offset, 1, {0, 255, 0}, 3, cv::LINE_AA);
    cv::circle(field, _lastPos[3]*zoom+offset, 1, {255, 255, 0}, 3, cv::LINE_AA);
      
      
      /* Other thing to try: color.
       * 
       *
       */


    //invert the fgMask before saving and showing (Marco likes it that way)
    cv::bitwise_not(fgMask, fgMask);

    if (saveVideo) {
      //Save videos
      wtrFrame.write(frame);
      wtrField.write(field);
      wtrMask.write(fgMask);
    }

    if (delay > 0) { //Only show progress if delay >= 0
      //pBackSub->getBackgroundImage(bg);
      cv::imshow("Frame", frame);
      cv::imshow("Field", field);
      cv::imshow("FG Mask", fgMask);
      //cv::imshow("Transformed", transformed);
      // imshow("BG", bg);

      time(&end);
      if (cv::waitKey(delay - difftime(end, start)) == 'q')
        break;
    }
    
    //New frame -> new line on file
    if (outputFile != "None")
      fout << '\n';

  } //end of main loop
  
  wtrFrame.release();
  wtrField.release();
  wtrMask.release();
  fout.close();   //closing if no file is opened simply has no effect. Safe to leave this line un-iffed

  if (delay < 0)
    std::cout << "  Done!\n";
  return true;
}


// -- Private methods -- //

void Padel::loadParam(std::string paramFile)
{
  std::fstream fin(paramFile, std::ios::in);
  if (fin.is_open()) {    
    _perspMat = cv::Mat(3, 3, CV_64FC1);
    
    fin >> _fps;

    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        fin >> _perspMat.at<double>(i,j);
  }
  else {
    calculatePerspMat(paramFile);
    calculateFPS();
    
    //save data
    std::fstream fout(paramFile, std::ios::out);
    fout << _fps << '\n';
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
        fout << _perspMat.at<double>(i,j) << ' ';
      fout << '\n';
    }

    fout.close();
  }
}

bool Padel::calculatePerspMat(std::string filename) {
  std::string winName = "Click on points indicated in green. Use WASD to move last cross. Right click to remove it. Press space to confirm.";
  cv::namedWindow(winName);
  cv::setMouseCallback(winName, onMouse);
  
  cv::Point2f angles[4]; //angles of the field

  if (_fileOpened)
    std::cout << "--- Calculating perspective ---\nClick on the 4 indicated points. Press 'n' to show another (random) frame. Press 'space' to confirm. Press 'q' to abort\n";
  else
    std::cout << "--- Calculating perspective ---\nClick on the 4 indicated points. Press 'space' to confirm. Press 'q to abort\n";

  std::srand(time(0));
  int totalFrame = _cap.get(cv::CAP_PROP_FRAME_COUNT) -2;
  int key;
  int count = 0;
  do {
    key = -1;
    cv::Mat original;
    if (_fileOpened)
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
      if (key == 'q')
        throw std::runtime_error{"PADEL NOTE: execution aborted by user."};

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

      // Draw little field to show where to put crosses
      int scale = 10;
      int offset = 20;
      cv::rectangle(frame, cv::Rect(cv::Point{offset, offset}, cv::Point{10*scale+offset, 20*scale+offset}), {129,94,61}, -1);
      cv::line(frame, {offset        ,      3  *scale+offset }, {10*scale+offset,        3  *scale+offset }, {255,255,255}, 1); //horizontal
      cv::line(frame, {offset        ,     17  *scale+offset }, {10*scale+offset,       17  *scale+offset }, {255,255,255}, 1); //horizontal
      cv::line(frame, {5*scale+offset,(int)(2.7*scale+offset)}, { 5*scale+offset, (int)(17.3*scale+offset)}, {255,255,255}, 1); //vertical
      cv::line(frame, {offset        ,     10  *scale+offset }, {10*scale+offset,       10  *scale+offset }, {0  ,0  ,255}, 2); //net
      cv::drawMarker(frame, {offset, offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
      cv::drawMarker(frame, {10*scale+offset, offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
      cv::drawMarker(frame, {offset, 17*scale+offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
      cv::drawMarker(frame, {10*scale+offset, 17*scale+offset}, {0,255,0}, cv::MARKER_TILTED_CROSS, scale, 2);
      //cv::putText(frame, "Select points indicated in green", {offset/2, offset/2}, 0, 0.5, {0,255,0}, 1);

      cv::imshow(winName, frame);
    }
    while (_fileOpened && key != 'n' && (key != ' ' || count < 4));
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

  //calculate matrix
  cv::Point2f rect[4] = { {0,0}, {10,0}, {10,17}, {0,17} };
  _perspMat = cv::getPerspectiveTransform(angles, rect);

  //reset video back to the beginning
  if (_fileOpened)
    _cap.set(cv::CAP_PROP_POS_FRAMES, 0);
  
  return true;
}

void Padel::calculateFPS() {
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
  double seconds = difftime(end, start);

  if (_fileOpened)
    _cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  _fps = num_frames / seconds;
  std::cout << "     Done (" << _fps << " fps)\n";
}