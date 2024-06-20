
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "HTs.h"
#include "HTs_evaluation.h"

/**
 * Print and store the values of some environment variables intialized in PBS script.
 * Print and store values of omp.h environment resources.
 * 
 * @param parameters A map containing the parameters for each preprocessing step.
*/
void environmentInfo(std::unordered_map<std::string, std::string>& parameters);

/**
 * Performs a series of preprocessing steps on the provided image based on the specified parameters.
 * Available options: grayscale conversion, histogram equalization, Gaussian blurring, color range restriction and Sobel edge detection (with binary conversion, optional).
 *
 * Has a parallel version: Sobel edge detection & Gaussian blurring.
 *
 * @param img The image to be processed.
 * @param parameters A map containing the parameters for each preprocessing step.
 * @param verbose Boolean parameter, if true show console messages, otherwise not.
 */
void preprocessImage(Image &img, std::unordered_map<std::string, std::string>& parameters, bool verbose);

/**
 * Applies the requested Hough Transform to the provided image. 
 * Serial versions: HT, PHT, PPHT
 * Parallel versions: HT, PHT, PPHT
 *
 * @param img The image on which to perform the Hough Transform.
 * @param parameters A map containing parameters for the Hough Transform.
 * @param verbose Boolean parameter, if true show console messages, otherwise not.
 */
std::vector<Segment> HoughTransformation(Image &img, std::unordered_map<std::string, std::string>& parameters, bool verbose);

/**
 * Process each image of the dataset with the same specified methodology.
 * Compute metrics: precision & recall.
 * 
 * @param parameters A map containing parameters for dataset processing.
*/
void processDataset(std::unordered_map<std::string, std::string>& parameters);

#endif