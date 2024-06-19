#ifndef UTILS_H
#define UTILS_H

#include "structures.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

/*********************
 * Utility functions *
**********************/

/**
 * Mean value of a vector. 
*/
double calculateMean(const std::vector<double> &recalls);

/**
 * Convert degree to radiants.
*/
double degreeToRadiant(double degree_values);

/**
 * Computes the Euclidean distance between two points.
 *
 * @param p1 First point.
 * @param p2 Second point.
 * @return The Euclidean distance between points p1 and p2.
 */
double euclideanDistance(const Point& p1, const Point& p2);

/**************************
 * Presentation functions *
 ***************************/
/**
 * Prints information about the given image.
 *
 * @param img The Image object to print information for.
 */
void imageInfo(const Image &img);

/**
 * Prints the Gaussian kernel to the console.
 *
 * @param kernel The Gaussian kernel represented as a 2D vector of floats.
 */
void printGaussianKernel(const std::vector<std::vector<float>> &kernel);


/**
 * Print line segment object attributes.
 * 
 * @param segments Vector of line segments.
*/
void printSegmentsInfo(const std::vector<Segment> &segments);

/***************************
 * Input/Output management *
****************************/
/**
 * Processes command line inputs for configuring the execution mode and paths related to image processing.
 * Validates the number of arguments, execution mode, existence, and format of the image file, 
 * and parses the parameters file to populate a parameters map.
 * 
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @param parameters Map to store parameters extracted from the parameters file.
 * @return Returns true if inputs are processed successfully, false otherwise.
 */
bool processInputs(int argc, char *argv[], std::unordered_map<std::string, std::string> &parameters);

/**
 * Reads an image from a file in PNM format (P5 for grayscale, P6 for color).
 * 
 * @param image_path Path to the image file.
 * @return Image object containing the image data. Returns an empty Image object on failure.
 */
Image readImage(const std::string& imagePath);

void savePerformance(const std::unordered_map<std::string, std::string> &parameters);

/**
 * Saves an image to a file in PNM format (P5 for grayscale, P6 for color).
 *
 * @param img Image object containing the image data to be saved.
 * @param outputPath Path where the image file should be saved.
 */
void saveImage(const Image& img, const std::string& outputPath);

/**
 * Converts all images in the given directory, or a single image, to the specified output format.
 * 
 * @param path Path to the image or directory containing images.
 * @param outputFormat Desired output image format.
 * @param parameters Map to store parameters extracted from the parameters file.
 */
void convertImages(const std::string &directoryPath, const std::string& outputFormat, std::unordered_map<std::string, std::string>& parameters);

/**
 * Creates a new directory at the specified path, or empties it if it already exists.
 * 
 * @param path Path of the directory to create or empty.
 */
void createOrEmptyDirectory(const std::string &path);

/**
 * Recursively removes all files and directories within the specified path.
 * 
 * @param path Path of the directory to empty.
 */
void removeContents(const std::string &path);

/**
 * Valdiate a path to a file or a directory, respectively.
 * 
 * @param path Path of the directory/file to validate.
*/
bool fileExists(const std::string &path);
bool pathExists(const std::string &path);

#endif
