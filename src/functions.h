
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "HTs.h"
#include "HTs_evaluation.h"

/**
 * Print and store the values of some environment variables intialized in PBS script.
 * Print and store values of omp.h environment resources.
 * 
 * @param parameters A map containing the parameters for the environment settings.
*/
void environmentInfo(std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs a series of preprocessing steps on the provided image based on the specified parameters.
 * Available options: grayscale conversion, histogram equalization, Gaussian blurring and Sobel edge detection (with binary conversion).
 *
 * Has a parallel version: Sobel edge detection & Gaussian blurring.
 *
 * @param img The image to be processed.
 * @param parameters A map containing the parameters for each preprocessing step.
 */
void preprocessImage(Image &img, std::unordered_map<std::string, std::string>& parameters);

/**
 * Applies the requested Hough Transform version to the provided image. 
 * Serial versions: HT, PHT, PPHT
 * Parallel MPI versions: HT, PHT, PPHT
 * Parallel OMP versions: HT, PHT, PPHT
 * Parallel Hybrid versions: HT, PHT
 *
 * @param img The image on which to perform the Hough Transform.
 * @param parameters A map containing parameters for the Hough Transform.
 * @param gtLines A vector of ground truth lines for the input image, if available. 
 */
std::vector<Segment> HoughTransformation(Image &img, std::unordered_map<std::string, std::string>& parameters, std::vector<Segment> gtSegments);

#endif