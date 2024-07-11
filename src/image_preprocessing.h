// image_processing.h

#ifndef IMAGE_PREPROCESSING_H
#define IMAGE_PREPROCESSING_H

#include "utils.h"

/**********************
 * Image manipulation *
***********************/

/**
 * Converts a color image to grayscale.
 *
 * @param image Reference to the Image object to be modified.
 */
void convertToGrayscale(Image &image);

/**
 * Enhances the contrast of an image using histogram equalization.
 * Assumes the image is already converted to grayscale.
 *
 * @param img Reference to the Image object to be modi9fied.
 */
void equalizeHistogram(Image &img);

/**
 * Converts the Sobel filter output to a binary image where all non-black pixels are set to white.
 *
 * @param img Reference to the Image object whose data has been processed with the Sobel filter.
 */
void toBinary(Image &img);


/**************************************************************
 * Image manipulation - Gaussian blurring (Serial & Parallel) *
***************************************************************/

/**
 * Calculates and returns a Gaussian kernel. Matrix of kernelSize x kernelSize.
 *
 * @param kernelSize The size of the kernel (must be odd).
 * @param sigma The standard deviation of the Gaussian function.
 * @return A 2D vector representing the Gaussian kernel.
 */
std::vector<std::vector<float>> calculateGaussianKernel(int kernelSize, float sigma);

/**
 * Applies Gaussian blur to an image.
 *
 * @param img Reference to the Image object to be blurred.
 * @param kernelSize Size of the Gaussian kernel to use for blurring.
 * @param sigma Standard deviation for the Gaussian kernel.
 * @param verbose True for console output.
 */
void gaussianBlur(Image &image, int kernelSize, float sigma, bool verbose);

/**
 * Applies Gaussian blur to an image in parallel.
 *
 * @param img Reference to the image to be blurred.
 * @param kernelSize Size of the Gaussian kernel, must be odd.
 * @param sigma Standard deviation of the Gaussian kernel.
 * @param verbose True for console output.
 * @param numThreads Number of threads for openMP parallelization.
 */
void gaussianBlurParallel(Image &img, int kernelSize, float sigma, bool verbose, int numThreads);

/**
 * Applies Gaussian blur to a single pixel in an image. Uses a convolution kernel.
 *
 * @param img Reference to the image to be blurred.
 * @param output Reference to the Image object where the blurred result will be stored.
 * @param kernel 2D vector representing the Gaussian kernel used for blurring.
 * @param x Horizontal coordinate of the pixel to be blurred.
 * @param y Vertical coordinate of the pixel to be blurred.
 */
void gaussianBlurPixel(const Image& img, Image& output, const std::vector<std::vector<float>>& kernel, int x, int y);

/*********************************************************************
 * Image manipulation - Sobel for edge detection (Serial & Parallel) *
**********************************************************************/

/**
 * Applies Sobel edge detection. Serial version.
 *
 * @param img Reference to the Image object to be processed. Must be in grayscale.
 * @param threshold Intensity threshold for edge detection. Pixels below this threshold will be set to 0.
 * @param scaleFactor Factor to scale the intensity of the edges in the final image.
 */
void sobelEdgeDetection(Image &img, int threshold, float scaleFactor);

/**
 * Applies Sobel edge detection. Parallel version.
 *
 * @param img Reference to the Image object to be processed. Must be in grayscale.
 * @param threshold Intensity threshold for edge detection. Pixels below this threshold will be set to 0.
 * @param scaleFactor Factor to scale the intensity of the edges in the final image.
 * @param numThreads Number of threads for openMP parallelization.
 */
void sobelEdgeDetectionParallel(Image &img, int threshold, float scaleFactor, int numThreads);

#endif