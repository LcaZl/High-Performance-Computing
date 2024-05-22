#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <sys/stat.h> // For stat()
#include <cstdlib> // For std::system
#include <iomanip> // Include for std::setprecision
#include <chrono> // Include for std::chrono
#include <dirent.h> // for I/O
#include <unistd.h> //for I/O
#include <random>
#include <cstdlib>
#include <tuple>
#include <mpi.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
/************************
 *  Internal Structure  *
*************************/

/**
 * 
 * Used to store an image and relative information. 
 * Has two methods to retrieve or set pixel's value at input coordinates of the image.
*/
struct Image {
    std::vector<unsigned char> data; 
    int width, height;
    bool isColor; // true per PPM, false per PGM

    // Function to get pixel value at (x, y)
    unsigned char at(int x, int y) const {
        if (x >= 0 && x < width && y >= 0 && y < height)
            return data[y * width + x];
        return 0;  // Return 0 for out of bounds
    }

    // Function to set pixel value at (x, y)
    void setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int index = (y * width + x) * 3;
            data[index] = r;
            data[index + 1] = g;
            data[index + 2] = b;
        }
    }
};

struct Point {
    int x, y;
    Point(int px, int py) : x(px), y(py) {}
    Point() : x(0), y(0) {}
};

/**
 * Used to store segments information. 
 * Used for segment generated with different HT versions (first constructor).
 * Also used for ground truth data (<inter>.. attributes and second constructor).
*/
struct Segment {
    Point start;
    Point end;
    double rho;        // Distance from the origin to the line
    double thetaRad;   // Angle in radians
    double thetaDeg;   // Angle in degrees (-180, 180)
    int votes;
    Point intersectionStart;
    Point intersectionEnd;
    double interRho;
    double interThetaRad;
    double interThetaDeg;

    Segment(Point s, Point e, double r, double tr, double td, int v) :
        start(s), end(e), rho(r), thetaRad(tr), thetaDeg(td), votes(v), intersectionStart(Point(0,0)), intersectionEnd(Point(0,0)), interRho(0), interThetaRad(0), interThetaDeg(0){}

    Segment(Point s, Point e, double r, double tr, double td, Point i1, Point i2, double interR, double interTr, double interTd) :
        start(s), end(e), rho(r), thetaRad(tr), thetaDeg(td), votes(0), intersectionStart(i1), intersectionEnd(i2), interRho(interR), interThetaRad(interTr), interThetaDeg(interTd){}
};

/**
 * Store information about a part of an image with organized informazion about the necessary outbound area (overlap).
 * Each image part is an horizontal split of an image.
 * 
*/
struct ImagePart {
    std::vector<unsigned char> data; // Stores the pixel data for this part
    int startRow; // The starting row of this part in the original image
    int width; // The width of the image (same for all parts)
    int height; // The height of this part including overlap
    int overlapTop; // The overlap rows on the top of the part
    int overlapBottom; // The overlap rows on the bottom of the part
};


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
void printImageInfo(const Image &img);

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
 * Splits an image into several parts, each with the necessary overlap to allow the kernel to co
 *
 * @param img The image to be split.
 * @param parts The number of parts to split the image.
 * @param overlap The number of rows that should overlap between consecutive parts, half the kernel size.
 * @return Returns a vector of ImagePart, where each ImagePart represents an horizontal portion of the original image.
 */
std::vector<ImagePart> splitImage(const Image &img, int parts, int overlap);

/**
 * Recombines split image parts back into a single image, considering overlaps.
 * The counterpart of the previous function.
 *
 * @param parts A vector of ImagePart, each representing a segment of an image.
 * @return Returns a single Image object that represents the recombined image.
 */
Image recombineImage(const std::vector<ImagePart> &parts);

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
