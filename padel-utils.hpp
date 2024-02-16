#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <opencv2/core.hpp>

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


template<typename P>
double distance(P p1, P p2) {
  return std::sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

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


#endif //UTILS_H