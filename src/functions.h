
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "HTs.h"
#include "HTs_evaluation.h"

/**
 * Print the values of some environment variables intialized in PBS script and in parameters.
 * 
 * @param parameters A map containing the parameters related to the environment.
*/
void environmentInfo(std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs a series of preprocessing steps on the provided image based on the specified parameters.
 * Available options: grayscale conversion (mandatory), histogram equalization, Gaussian blurring and Sobel edge detection (SED) (with binary conversion mandatory after SED).
 *
 * Has a parallel version: SED & Gaussian blurring.
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
 * @param gtLines A vector of ground truth lines for the input image (Available only with synthetic samples).
 */
std::vector<Segment> HoughTransformation(Image &img, std::unordered_map<std::string, std::string>& parameters, std::vector<Segment> gtSegments);

#endif