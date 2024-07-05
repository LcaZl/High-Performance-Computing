#ifndef HTS_h
#define HTS_H

#include "utils.h"
#include "HTs_evaluation.h"

/**************
 *   SERIAL   *
***************/

/**
 * Performs the standard sequential Hough Transform (HT) or Probabilistic Hough Transform (PHT) to detect lines in the given image, sequential version.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A pair composed of: 2D vector representing the accumulator array, segments detected.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs the Progressive Probabilistic Hough Transform (PPHT) to detect segments in the given image, sequential version.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A pair composed of: 2D vector representing the accumulator array, segments detected.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/********************
 *  PARALLEL - MPI  *
*********************/
/**
 * Performs the parallelized version of the HT or PHT using MPI.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A pair composed of: 2D vector representing the accumulator array, segments detected.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI(Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs the parallelized version of the PPHT using MPI.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A pair composed of: 2D vector representing the accumulator array, segments detected.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_MPI(Image& image, std::unordered_map<std::string, std::string>& parameters);

/********************
 *  PARALLEL - OMP  *
*********************/

/**
 * Performs the parallelized version of HT or PHT using OpenMP.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A pair composed of: 2D vector representing the accumulator array, segments detected.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs the parallelized version of the PPHT using OpenMP.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A pair composed of: 2D vector representing the accumulator array, segments detected.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/***********************
 *  PARALLEL - Hybrid  *
************************/

/**
 * Performs the parallelized version of the HT using an Hybrid approach with MPI and openMP.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A pair composed of: 2D vector representing the accumulator array, segments detected.
 */
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI_OMP(Image& image, std::unordered_map<std::string, std::string>& parameters);
#endif