#ifndef UTILS_H
#define UTILS_H

#include "structures.h"


/*********************
 * Utility functions *
**********************/

/**
 * Mean value of a vector. 
 * @param values Values list.
 * @return The mean value of the input series.
*/
double calculateMean(const std::vector<double> &values);

/**
 * Convert degree to radiants.
 * @param degreeValues Velue in degree to convert to radiant.
 * @return Input value in radiant
*/
double degreeToRadiant(double degreeValues);

/**
 * Computes the Euclidean distance between two points.
 *
 * @param p1 First point.
 * @param p2 Second point.
 * @return The Euclidean distance between points p1 and p2.
 */
double euclideanDistance(const Point& p1, const Point& p2);

/**
 * Calculates the Euclidean distance between the midpoints of two line segments defined by their endpoints.
 *
 * @param aStart The starting point of the first segment.
 * @param aEnd The ending point of the first segment.
 * @param bStart The starting point of the second segment.
 * @param bEnd The ending point of the second segment.
 * @return The distance between the midpoints of the two line segments as a double.
 */
double midpointDistance(const Point &aStart, const Point &aEnd, const Point &bStart, const Point &bEnd);

/**
 * Calculates the length of a segment using the Euclidean distance formula.
 *
 * @param segment The segment for which the length is to be calculated.
 * @return The Euclidean length of the segment as a double.
 */
double segmentLength(const Segment &segment);

/**
 * Flattens a 2D vector of integers into a single 1D vector.
 *
 * @param matrix The 2D vector of integers to be flattened.
 * @return A 1D vector containing all elements of the 2D matrix in row-major order.
 */std::vector<int> flatten(const std::vector<std::vector<int>> &matrix);

/**
 * Reshapes a 1D vector of integers into a 2D vector with specified number of rows and columns.
 * The elements are filled into the 2D vector in row-major order based on the sequence in the 1D vector.
 *
 * @param flat The 1D vector of integers to be reshaped into a 2D matrix.
 * @param rows The number of rows the resulting matrix should have.
 * @param cols The number of columns the resulting matrix should have.
 * @return A 2D vector of integers structured into the specified rows and columns.
 * 
 */std::vector<std::vector<int>> reshape(const std::vector<int> &flat, int rows, int cols);


/**************************
 * Presentation functions *
 ***************************/
/**
 * Print the program external fixed parameters and also the internal generated.
 *
 * @param parameters Parameters of the program.
 */
void printParameters(const std::unordered_map<std::string, std::string>& parameters);


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

/**
 * Print the parameters of the Hough Transform.
 * 
 * @param parameters A map containing the parameters of the Hough Transform.
 */
void houghTransformInfo(std::unordered_map<std::string, std::string>& parameters);

/***************************
 * Input/Output management *
****************************/
/**
 * Processes command line inputs for configuring the execution mode and paths related to image processing.
 * Validates the number of arguments, execution mode, existence, format of the image file 
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
 * @return Image structure containing the image data and information. Returns an empty Image object on failure.
 */
Image readImage(const std::string& imagePath);

/**
 * Load the ground truth data from the specified path.
 * 
 * @param gtPath File system path where the ground_truth.csv file is located.
 * @return List of segments data structures specified in the file at the given path.
 */
std::unordered_map<std::string, std::vector<Segment>> loadGroundTruthData(const std::string &gtPath);

/**
 * Save all the performance metrics into a structured .csv file for later inspection.
 * 
 * @param parameters map that contain the output path and the metrics to save.
 */
void savePerformance(const std::unordered_map<std::string, std::string> &parameters);

/**
 * Saves an image to a file in PNM format (P5 for grayscale, P6 for color).
 *
 * @param img Image object containing the image data to be saved.
 * @param outputPath Path where the image file should be saved.
 */
void saveImage(const Image& img, const std::string& outputPath);

/**
 * Converts all images in the given directory, or a single image, to the specified output format using an external python script.
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
