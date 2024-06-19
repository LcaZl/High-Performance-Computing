#ifndef HTS_h
#define HTS_H

#include "utils.h"
#include "HTs_evaluation.h"


std::vector<std::vector<int>> houghTransformParallel_MPI(const Image& image, std::unordered_map<std::string, std::string>& parameters);
std::vector<Segment> linesExtractionParallel_MPI(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters);
std::vector<Segment> linesProgressiveExtractionParallel_MPI(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters);
/**************
 *   SERIAL   *
***************/

/**
 * Performs the Hough Transform to detect lines in the given image. This function calculates the parameter space 
 * (rho and theta) for potential lines in the image and populates an accumulator array based on pixel values.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::vector<std::vector<int>> houghTransform(const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Extracts line segments from a Hough Transform (HT) accumulator.
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Extracts line segments using a progressive approach to exactly identify them.
 * The basi HT version for this approach is a PHT.
 * [REFERENCE]
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesProgressiveExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**************
 *  PARALLEL  *
***************/

/**
 * Performs the parallelized version of the Hough Transform using OpenMP.
 * Similar to the houghTransform function, but optimized for performance by distributing the computation across multiple
 * threads specified by the user.
 *
 * @param image Reference to the image object to analyze.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return A 2D vector representing the accumulator array.
 */
std::vector<std::vector<int>> houghTransformParallel_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters);


/**
 * Parallel version.
 * 
 * Extracts line segments from a Hough Transform (HT) accumulator.
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesExtractionParallel_OMP(const std::vector<std::vector<int>> &accumulator, const Image &image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Parallel version.
 * 
 * Extracts line segments from a Probabilistic Progressive Hough Transform (PPHT) accumulator.
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param parameters A map containing parameters for the Hough Transform.
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesProgressiveExtractionParallel_OMP(const std::vector<std::vector<int>> &accumulator, const Image &image, std::unordered_map<std::string, std::string>& parameters);

#endif