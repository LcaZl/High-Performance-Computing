#ifndef HTS_h
#define HTS_H

#include "utils.h"
#include "HTs_evaluation.h"

/**************
 *   SERIAL   *
***************/

/**
 * Performs the Hough Transform to detect lines in the given image. This function calculates the parameter space 
 * (rho and theta) for potential lines in the image and populates an accumulator array based on pixel values.
 *
 * @param image Reference to the image object to analyze.
 * @param thetaResolution The resolution in degree for the theta dimension in the accumulator array.
 * @param probabilistic Flag to determine if a probabilistic approach should be used.
 * @param samplingRate Determines how frequently pixels are sampled in the probabilistic approach (probablistic=true).
 * @return A 2D vector representing the accumulator array.
 */
std::vector<std::vector<int>> houghTransform(const Image& image, int thetaResolution, bool probabilistic, int samplingRate);

/**
 * Extracts line segments from a Hough Transform (HT) accumulator.
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param voteTreshold Minimum number of votes required to consider a line.
 * @param thetaResolution Resolution used for theta axis in the accumulator (degree).
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, int voteTreshold, int thetaResolution);

/**
 * Extracts line segments using a progressive approach to exactly identify them.
 * The basi HT version for this approach is a PHT.
 * [REFERENCE]
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param voteTreshold Minimum number of votes required to consider a line.
 * @param thetaResolution Resolution used for theta axis in the accumulator (degree).
 * @param line_gap Maximum allowed gap between points on the same line.
 * @param line_length Minimum number of points to form a valid line segment.
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesProgressiveExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, int voteTreshold, int thetaResolution, int line_gap, int line_length);

/**************
 *  PARALLEL  *
***************/

/**
 * Performs the parallelized version of the Hough Transform using OpenMP.
 * Similar to the houghTransform function, but optimized for performance by distributing the computation across multiple
 * threads specified by the user.
 *
 * @param image Reference to the image object to analyze.
 * @param thetaResolution The resolution in degree for the theta dimension in the accumulator array.
 * @param probabilistic Flag to determine if a probabilistic approach should be used.
 * @param samplingRate Determines how frequently pixels are sampled in the probabilistic approach (probablistic=true).
 * @param threadCount The number of threads to use in the parallel computation.
 * @return A 2D vector representing the accumulator array.
 */
std::vector<std::vector<int>> parallelHoughTransform(const Image& image, int thetaResolution, bool probabilistic, int samplingRate, int threadCount);


/**
 * Parallel version.
 * 
 * Extracts line segments from a Hough Transform (HT) accumulator.
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param voteTreshold Minimum number of votes required to consider a line.
 * @param thetaResolution Resolution used for theta axis in the accumulator (degree).
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesExtractionParallel(const std::vector<std::vector<int>> &accumulator, const Image &image, int voteThreshold, int thetaResolution, int threadCount);

/**
 * Parallel version.
 * 
 * Extracts line segments from a Probabilistic Progressive Hough Transform (PPHT) accumulator.
 *
 * @param accumulator 2D vector representing the HT accumulator space.
 * @param image Reference to the image from which lines are to be extracted.
 * @param voteTreshold Minimum number of votes required to consider a line.
 * @param thetaResolution Resolution used for theta axis in the accumulator (degree).
 * @param line_gap Maximum allowed gap between points on the same line.
 * @param line_length Minimum number of points to form a valid line segment.
 * @return Vector of extracted line segments.
 */
std::vector<Segment> linesProgressiveExtractionParallel(const std::vector<std::vector<int>> &accumulator, const Image &image, int voteThreshold, int thetaResolution, int lineGap, int lineLength, int threadCount);

#endif