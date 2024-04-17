// image_processing.h

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "utils.h"

/******************
 * Main functions *
*******************/

/**
 * Performs a series of preprocessing steps on the provided image based on the specified configuration.
 * Available options: grayscale conversion, histogram equalization, Gaussian blurring and Sobel edge detection.
 * The function also supports parallel processing for some of these steps. 
 * Parallel support: Sobel edge detection & Gaussian blurring
 *
 * @param img The image to be processed.
 * @param config A map containing the configuration options for each processing step, including whether to enable
 *               each step, parameters for Gaussian blur and Sobel edge detection, and parallel processing options.
 */
void preprocessImage(Image &img, std::unordered_map<std::string, std::string> config);

/**
 * Applies the Hough Transform to the provided image, supporting both parallel and sequential execution. 
 * This function is called after preprocessing steps to effectively run the transform with the specified parameters.
 *
 * @param img The image on which to perform the Hough Transform.
 * @param config A map containing configuration options for the Hough Transform.
 * @return A 2D accumulator array representing the parameter space (rho, theta) with voting counts, indicating
 *         the presence and strength of lines in the image.
 */
std::vector<std::vector<int>> HoughTransformation(Image &img, std::unordered_map<std::string, std::string> config);

/********************
 * Image processing *
*********************/

std::vector<std::vector<int>> applyProbabilisticHoughTransform(const Image &image, int voteThreshold, int thetaResolution, int samplingRate);
void restrictColorScale(Image &img, unsigned char newMin, unsigned char newMax);

/**
 * Converts the Sobel filter output to a binary image where all non-black pixels are set to white.
 * The function modifies the input Image object in place.
 *
 * @param img Reference to the Image object whose data has been processed with the Sobel filter.
 */
void sobelToBinary(Image &img);
/**
 * Converts a color image to grayscale.
 * The function modifies the input Image object in place.
 *
 * @param img Reference to the Image object to be converted.
 */
void convertToGrayscale(Image& image);

/**
 * Enhances the contrast of a grayscale image using histogram equalization.
 * The function modifies the input Image object in place.
 * Assumes the image is already converted to grayscale.
 * 
 * @param img Reference to the Image object to be enhanced.
 */
void equalizeHistogram(Image &img);

/**
 * Calculates and returns a Gaussian kernel.
 * 
 * @param kernelSize The size of the kernel (must be odd).
 * @param sigma The standard deviation of the Gaussian function.
 * @return A 2D vector representing the Gaussian kernel.
 */
std::vector<std::vector<float>> calculateGaussianKernel(int kernelSize, float sigma);

/**
 * Draws a line on an image using Bresenham's line algorithm.
 * 
 * @param x0 Starting x-coordinate of the line.
 * @param y0 Starting y-coordinate of the line.
 * @param x1 Ending x-coordinate of the line.
 * @param y1 Ending y-coordinate of the line.
 * @param rgbData Reference to the vector containing the image's RGB data.
 * @param width The width of the image.
 * @param height The height of the image.
 */
void draw_lines_bresenham(int x0, int y0, int x1, int y1, std::vector<unsigned char> &rgbData, int width, int height);

/**
 * Draws lines detected by the Hough Transform on an image.
 * 
 * @param accumulator The accumulator array from the Hough Transform, indicating detected lines.
 * @param image Reference to the Image object to draw lines on.
 * @param threshold The threshold of votes a line must have to be drawn.
 */
void draw_hough_lines(const std::vector<std::vector<int>> &accumulator, Image &image, std::unordered_map<std::string, std::string> config);

/**
 * Applies Gaussian blur to an image.
 * 
 * @param img Reference to the Image object to be blurred.
 * @param kernelSize Size of the Gaussian kernel to use for blurring.
 * @param sigma Standard deviation for the Gaussian kernel.
 */
void applyGaussianBlur(Image &image, int kernelSize, float sigma);

/**
 * Applies Sobel edge detection on a grayscale image.
 * The function modifies the input Image object in place, emphasizing edges based on the Sobel operator.
 * 
 * @param img Reference to the Image object to be processed. Must be in grayscale.
 * @param threshold Intensity threshold for edge detection. Pixels below this threshold will be set to 0.
 * @param scaleFactor Factor to scale the intensity of the edges in the final image.
 */
void applySobelEdgeDetection(Image &img, int threshold, float scaleFactor);

/**
 * Applies the Hough Transform to detect lines in a binary edge-detected image.
 * 
 * @param image The image object containing edge-detected binary image data.
 * @param voteThreshold The minimum number of votes in the accumulator for a potential line to be considered a real line.
 * @param thetaResolution The resolution of the theta axis in the accumulator array, determining the granularity of possible line orientations.
 * @return A 2D accumulator array representing the parameter space (rho, theta) with voting counts for each possible line.
 */
std::vector<std::vector<int>> applyHoughTransform(const Image& image, int voteThreshold, int thetaResolution);


/**
 * Applies Sobel edge detection on a grayscale image.
 * 
 * The function modifies the input Image object in place, emphasizing edges based on the Sobel operator.
 * @param img Reference to the Image object to be processed. Must be in grayscale.
 * @param threshold Intensity threshold for edge detection. Pixels below this threshold will be set to 0.
 * @param scaleFactor Factor to scale the intensity of the edges in the final image.
 */
void applySobelEdgeDetectionParallel(Image &img, int threshold, float scaleFactor);

/**
 * Applies the Hough Transform in parallel to detect lines in edge-detected image.
 * 
 * @param image The image object containing edge-detected binary image data.
 * @param voteThreshold The minimum number of votes needed for a line to be considered significant.
 * @param thetaResolution The resolution of the theta axis in the accumulator array.
 * @return A 2D accumulator array representing the parameter space (rho, theta) with voting counts.
 */
std::vector<std::vector<int>> applyHoughTransformParallel(const Image &image, int voteThreshold, int thetaResolution);

/********************
 * Images splitting *
*********************/

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
 * TO FIX
*/
void parallelGaussianBlur(Image &img, int kernelSize, float sigma, int numParts);
void applyGaussianBlurToPart(ImagePart &imgPart, const std::vector<std::vector<float>> &kernel, int overlap);


#endif