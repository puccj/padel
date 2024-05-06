#ifndef PADEL_UTILS_H
#define PADEL_UTILS_H

#include <vector>
#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//Generate a single combination
template<typename T>
void generate(int index, int remaining, const std::vector<T> &elements, std::vector<T> &combination, std::vector<std::vector<T>> &result) {
  // If the combination is complete, add it to the result
  if (remaining == 0) {
    result.push_back(combination);
    return;
  }

  // Generate combinations using recursion
  for (int i = index; i < elements.size(); ++i) {
    combination.push_back(elements[i]);
    generate(i + 1, remaining - 1, elements, combination, result);
    combination.pop_back();
  }
}

//generate all combinations given a set of elements
template<typename T>
std::vector<std::vector<T>> generateCombinations(const std::vector<T>& elements, int r) {
  std::vector<std::vector<T>> result;
  std::vector<T> combination;
  //void generate(int index, int remaining, const std::vector<T>& elements, std::vector<T>& combination, std::vector<std::vector<T>>& result);
  generate(0, r, elements, combination, result);
  return result;
}

//draw a 2D scheme of a padel field
void draw2DField(cv::Mat mat, int offset, int zoom) {
  mat = cv::Scalar{209, 186, 138};
  cv::rectangle(mat, cv::Rect(cv::Point{offset, offset}, cv::Point{10*zoom+offset, 20*zoom+offset}), {129,94,61}, -1);
  cv::line(mat, {offset        ,     3  *zoom+offset }, {10*zoom+offset,        3  *zoom+offset }, {255,255,255}, 1); //horizontal
  cv::line(mat, {offset        ,    17  *zoom+offset }, {10*zoom+offset,       17  *zoom+offset }, {255,255,255}, 1); //horizontal
  cv::line(mat, {5*zoom+offset,(int)(2.7*zoom+offset)}, { 5*zoom+offset, (int)(17.3*zoom+offset)}, {255,255,255}, 1); //vertical
  cv::line(mat, {offset        ,    10  *zoom+offset }, {10*zoom+offset,       10  *zoom+offset }, {0  ,0  ,255}, 2); //net
}


//return the most present color inside the contour
cv::Scalar findBestColor(const cv::Mat& image, const std::vector<cv::Point>& contour) {
  // Create a mask for the contour region
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  std::vector<std::vector<cv::Point>> contours = {contour};
  cv::drawContours(mask, contours, 0, cv::Scalar(255), cv::FILLED);

  // Extract pixels within the contour region using the mask
  cv::Mat region;
  image.copyTo(region, mask);

  // Reshape the ROI to a 2D array of pixels
  cv::Mat reshaped = region.reshape(1, region.rows * region.cols);
  reshaped.convertTo(reshaped, CV_32F);

  // Define the number of clusters (colors) you want to find
  int numClusters = 3; // Adjust as needed

  // Apply k-means clustering
  cv::Mat labels, centers;
  kmeans(reshaped, numClusters, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, centers);

  //get colors and their percentage
  cv::Mat predominant_color;
  double max = 0;
  for (int i = 0; i < numClusters; ++i) {
    cv::Mat thisColor = centers.row(i);
    int n = cv::countNonZero( labels == i );
    double p = 100.0 * double(n) / (image.rows * image.cols);

    if (p > 90) //skip if found color for 90% of the image (it's the black border)
      continue;
    if (max < p) {
      max = p;
      predominant_color = thisColor;
    }
  }
  // Convert back to uchar
  predominant_color.convertTo(predominant_color, CV_8U);
  //std::cout << "Predominant color: " << predominant_color << ' ' << max << " %\n";

  /*
  // Replace each pixel with the color of the nearest cluster center
  cv::Mat simplified_image(image.size(), image.type());

  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      int label = labels.at<int>(i * image.cols + j);
      simplified_image.at<cv::Vec3b>(i, j) = centers.at<cv::Vec3f>(label);
    }
  }

  // Convert back to uchar
  simplified_image.convertTo(simplified_image, CV_8U);

  // Display the original and simplified images
  cv::imshow("Original Image", region);
  cv::imshow("Simplified Image", simplified_image);
  //cv::waitKey(0);
  */

  cv::Scalar color(predominant_color.at<uchar>(0),predominant_color.at<uchar>(1),predominant_color.at<uchar>(2));
  return {color};
}


//calculate the median color of the region inside the contour
cv::Scalar medianColor(const cv::Mat& image, const std::vector<cv::Point>& contour) {
  // Create a mask for the contour region
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  std::vector<std::vector<cv::Point>> contours = {contour};
  cv::drawContours(mask, contours, 0, cv::Scalar(255), cv::FILLED);

  // Extract pixels within the contour region using the mask
  cv::Mat region;
  image.copyTo(region, mask);

  std::vector<cv::Mat> channels(3);
  cv::split(region, channels);

  // cv::imshow("Image", image);
  // cv::imshow("Mask", mask);
  // cv::imshow("Region", region);
  // cv::imshow("Channel0", channels[0]);
  // cv::imshow("Channel1", channels[1]);
  // cv::imshow("Channel2", channels[2]);

  cv::Scalar median;
  for (int i = 0; i < 3; ++i) {
    // Convert the image to a 1D vector
    std::vector<uchar> pixels;
    pixels.assign(channels[i].data, channels[i].data + channels[i].total());

    // Sort the vector
    sort(pixels.begin(), pixels.end());

    // Calculate the median value
    int size = pixels.size();

    // if (size % 2 == 0)
    //   median = (pixels[size / 2 - 1] + pixels[size / 2]) / 2.0;
    // else
        median[i] = pixels[size / 2];
  }

  return median;


  // Flatten the region to a 1D array
  cv::Mat flatRegion = region.reshape(1, 1);
  

  cv::Scalar medianColor;
  for (int i = 0; i < 3; ++i) {
    int imageType = channels[i].type();
    std::cout << "type = " << imageType << '\n';
    // Interpret the data type
    int depth = imageType & CV_MAT_DEPTH_MASK;
    int Nchannels = 1 + (imageType >> CV_CN_SHIFT);

    // Print the information
    std::cout << "Image Depth: ";
    switch (depth) {
      case CV_8U: std::cout << "8-bit unsigned"; break;
      case CV_8S: std::cout << "8-bit signed"; break;
      case CV_16U: std::cout << "16-bit unsigned"; break;
      case CV_16S: std::cout << "16-bit signed"; break;
      case CV_32S: std::cout << "32-bit signed"; break;
      case CV_32F: std::cout << "32-bit float"; break;
      case CV_64F: std::cout << "64-bit float"; break;
      default: std::cout << "Unknown depth"; break;
    }
    std::cout << ", Channels: " << Nchannels << std::endl;
    cv::sort(channels[i], channels[i], cv::SORT_ASCENDING);
    medianColor[i] = channels[i].at<uchar>(flatRegion.cols/2);
  }
  
  return medianColor;
}


/*
cv::Scalar medianColor(const cv::Mat& image) {
  int nPixels = image.rows * image.cols;
  cv::Mat reshaped = image.reshape(1, nPixels);

  std::cout << "Debug: channel = " << image.channels() << '\n';


  // // Check the data type of the image
  // int imageType = image.type();

  // // Interpret the data type
  // int depth = imageType & CV_MAT_DEPTH_MASK;
  // int Nchannels = 1 + (imageType >> CV_CN_SHIFT);

  // // Print the information
  // std::cout << "Image Depth: ";
  // switch (depth) {
  //   case CV_8U: std::cout << "8-bit unsigned"; break;
  //   case CV_8S: std::cout << "8-bit signed"; break;
  //   case CV_16U: std::cout << "16-bit unsigned"; break;
  //   case CV_16S: std::cout << "16-bit signed"; break;
  //   case CV_32S: std::cout << "32-bit signed"; break;
  //   case CV_32F: std::cout << "32-bit float"; break;
  //   case CV_64F: std::cout << "64-bit float"; break;
  //   default: std::cout << "Unknown depth"; break;
  // }
  // std::cout << ", Channels: " << Nchannels << std::endl;
  //


  std::vector<cv::Mat> channels(3);
  cv::split(reshaped, channels);
  
  //calculate median value for each channel
  std::cout << 1;
  cv::Scalar medianColor;
  for (int i = 0; i < 3; ++i) {
    std::cout << 2;
    cv::sort(channels[i], channels[i], cv::SORT_ASCENDING);
    std::cout << 3;
    medianColor[i] = channels[i].at<int>(nPixels/2);
    std::cout << 4;
  }
  std::cout << 5;

  // Convert back to 8-bit representation
  // medianColor.convertTo(medianColor, CV_8U);

  return medianColor;
}
*/


// template<typename P>
// double distance(P p1, P p2) {
//   return std::sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
// }

template<typename P>
bool comparePoints(P p1, P p2) {
  if (p1.x < p2.x)
    return true;
  if (p1.x == p2.x && p1.y < p2.y)
    return true;
  return false;
}

template <typename P>
P operator+(P lhs, int rhs) {
  return P{lhs.x + rhs, lhs.y + rhs};
}

std::vector<cv::Point> operator-(std::vector<cv::Point> lhs, cv::Rect const& rhs) {
  for (int i = 0; i < lhs.size(); ++i)
    lhs[i] -= {rhs.x, rhs.y};
  return lhs;
}

#endif //PADEL_UTILS_H