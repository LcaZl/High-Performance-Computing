#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include <sys/stat.h> // For stat()
#include <cstdlib> // For std::system
#include <iomanip> // Include for std::setprecision
#include <chrono> // Include for std::chrono
#include <dirent.h> // for I/O
#include <unistd.h> //for I/O
#include <random>

// Image "container"

/*************************
 * Images rapresentation *
**************************/

struct Image {
    std::vector<unsigned char> data; 
    int width, height;
    bool isColor; // true per PPM, false per PGM
};

struct ImagePart {
    std::vector<unsigned char> data; // Stores the pixel data for this part
    int startRow; // The starting row of this part in the original image
    int width; // The width of the image (same for all parts)
    int height; // The height of this part including overlap
    int overlapTop; // The overlap rows on the top of the part
    int overlapBottom; // The overlap rows on the bottom of the part
};

/***************************
 * Input/Output management *
****************************/

/**
 * Processes command line inputs for configuring the execution mode and paths related to image processing.
 * Validates the number of arguments, execution mode, existence, and format of the image file, 
 * and parses the configuration file to populate an output configuration map.
 * 
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @param outConfig Map to store configuration settings extracted from the command line and the configuration file.
 * @return Returns true if inputs are processed successfully, false otherwise.
 */
bool processInputs(int argc, char *argv[], std::unordered_map<std::string, std::string> &outConfig);

/**
 * Reads an image from a file in PNM format (P5 for grayscale, P6 for color).
 * 
 * @param imagePath Path to the image file.
 * @return Image object containing the image data. Returns an empty Image object on failure.
 */
Image readImage(const std::string& imagePath);

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
 */
void convertImages(const std::string &directoryPath, const std::string& outputFormat);

/**
 * Creates a new directory at the specified path, or empties it if it already exists.
 * 
 * @param path Path of the directory to create or empty.
 */
void createOrEmptyDirectory(const std::string &path);
void removeContents(const std::string &path);

bool fileExists(const std::string &path);
bool pathExists(const std::string &path);

/**************************
 * Presentation functions *
 ***************************/

/**
 * Prints information about the given Image object to the console.
 *
 * @param img The Image object to print information for.
 */
void printImageInfo(const Image &img);

/**
 * Prints the Gaussian kernel to the console.
 *
 * @param kernel The Gaussian kernel represented as a 2D vector of floats.
 */
void printGaussianKernel(const std::vector<std::vector<float>> &kernel);


#endif
