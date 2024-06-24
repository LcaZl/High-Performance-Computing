#ifndef HTS_h
#define HTS_H

#include "utils.h"
#include "HTs_evaluation.h"

/**************
 *   SERIAL   *
***************/

/**
 * Performs the standard sequential Hough Transform to detect lines in the given image.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs the Progressive Probabilistic Hough Transform to detect segments in the given image.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/********************
 *  PARALLEL - MPI  *
*********************/
/**
 * Performs the parallelized version of the Hough Transform using MPI.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs the parallelized version of the Progressive Probabilistic Hough Transform using MPI.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_MPI(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/********************
 *  PARALLEL - OMP  *
*********************/

/**
 * Performs the parallelized version of the Hough Transform or the Probabilistic Hough Transform using OpenMP.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs the parallelized version of the Progressive Probabilistic Hough Transform using OpenMP.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters);


std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters);
#endif