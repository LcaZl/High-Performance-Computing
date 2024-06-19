
#include "image_preprocessing.h"

/**********************
 * Image manipulation *
***********************/

void convertToGrayscale(Image& img) {

    // If the image is already in grayscale, do nothing.
    if (!img.isColor) return;

    std::vector<unsigned char> grayscaleData;
    // Pre-allocate memory for grayscale data.
    grayscaleData.reserve(img.width * img.height);

    for (size_t i = 0; i < img.data.size(); i += 3) {
        // Calculate the luminance of the pixel.
        unsigned char r = img.data[i];
        unsigned char g = img.data[i + 1];
        unsigned char b = img.data[i + 2];
        unsigned char gray = static_cast<unsigned char>(0.2126 * r + 0.7152 * g + 0.0722 * b);
        grayscaleData.push_back(gray);
    }

    // Update the img object to reflect the conversion.
    img.data = grayscaleData;
    img.isColor = false; // The image is now in grayscale.
}

void equalizeHistogram(Image& img) {

    // Ensure the image is in grayscale.
    if (img.isColor) {
        std::cerr << "Image is not in grayscale. Please convert it first." << std::endl;
        return;
    }

    const size_t imageSize = img.width * img.height;
    std::vector<unsigned int> histogram(256, 0);

    // Calculate histogram
    for (size_t i = 0; i < imageSize; ++i) {
        histogram[img.data[i]]++;
    }

    // Calculate Cumulative Distribution Function
    std::vector<float> cdf(256, 0);
    cdf[0] = histogram[0];
    for (size_t i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Normalize the CDF
    float cdfMin = cdf[0];
    float cdfMax = cdf[255];
    for (size_t i = 0; i < 256; ++i) {
        cdf[i] = ((cdf[i] - cdfMin) / (cdfMax - cdfMin)) * 255;
    }

    for (size_t i = 0; i < imageSize; ++i) {
        img.data[i] = static_cast<unsigned char>(cdf[img.data[i]]);
    }
}

void toBinary(Image& img) {
    if (img.data.empty()) return; // Do nothing if the image data is empty.

    for (unsigned char& pixel : img.data) {
        if (pixel != 0) {
            pixel = 255;
        }
    }
}

/**************************************************************
 * Image manipulation - Gaussian blurring (Serial & Parallel) *
***************************************************************/

std::vector<std::vector<float>> calculateGaussianKernel(int kernelSize, float sigma) {
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    float sum = 0.0; // Sum of all elements for normalization
    int edge = kernelSize / 2; // Calculate the edge to center the kernel

    // Generate Gaussian kernel
    for (int i = -edge; i <= edge; i++) {
        for (int j = -edge; j <= edge; j++) {
            float value = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[i + edge][j + edge] = value;
            sum += value; // Add to the normalization sum
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

void gaussianBlur(Image& img, int kernelSize, float sigma, bool verbose) {
    std::vector<std::vector<float>> kernel = calculateGaussianKernel(kernelSize, sigma);
    if (verbose){
        std::cout << "-> Gaussian Kernel (" << kernelSize << "x" << kernelSize << "):\n" << std::endl;
        printGaussianKernel(kernel);
    }

    std::vector<unsigned char> output(img.data.size(), 0); // Prepare output image

    int edge = kernelSize / 2; // Edge offset for kernel application

    // Apply Gaussian kernel to each pixel of the image
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            float blurredPixel = 0.0; // Accumulator for the blurred pixel value

            // Convolve the kernel over the pixel
            for (int ky = -edge; ky <= edge; ky++) {
                for (int kx = -edge; kx <= edge; kx++) {
                    int px = std::min(std::max(x + kx, 0), img.width - 1);
                    int py = std::min(std::max(y + ky, 0), img.height - 1);
                    blurredPixel += img.data[py * img.width + px] * kernel[ky + edge][kx + edge];
                }
            }

            // Assign the blurred value to the output image
            output[y * img.width + x] = static_cast<unsigned char>(std::min(std::max(int(blurredPixel), 0), 255));
        }
    }

    img.data = output; // Update the original image with the blurred one
}

void gaussianBlurPixel(const Image& img, Image& output, const std::vector<std::vector<float>>& kernel, int x, int y) {
    int edge = kernel.size() / 2;
    float blurredPixel = 0.0;

    for (int ky = -edge; ky <= edge; ky++) {
        for (int kx = -edge; kx <= edge; kx++) {
            // Calculate actual index accounting for boundary conditions
            int realY = std::max(0, std::min(y + ky, img.height - 1));
            int realX = std::max(0, std::min(x + kx, img.width - 1));
            // Accumulate the blurred value
            blurredPixel += img.data[realY * img.width + realX] * kernel[ky + edge][kx + edge];
        }
    }
    // Assign the blurred pixel, clamping to valid byte range
    output.data[y * output.width + x] = static_cast<unsigned char>(std::min(std::max(int(blurredPixel), 0), 255));
}

void gaussianBlurParallel(Image& img, int kernelSize, float sigma, bool verbose, int numThreads) {
    // Calculate Gaussian kernel
    std::vector<std::vector<float>> kernel = calculateGaussianKernel(kernelSize, sigma);
    if (verbose) {
        std::cout << "-> Gaussian Kernel (" << kernelSize << "x" << kernelSize << "):\n" << std::endl;
        printGaussianKernel(kernel);
    }

    Image output = img; // Create a copy of the original image for the output

    // Parallel region with specified number of threads
    #pragma omp parallel num_threads(numThreads) default(none) shared(img, output, kernel, kernelSize)
    {
        // Parallelize the nested loops with collapse to flatten nested loops into a single parallel loop
        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                gaussianBlurPixel(img, output, kernel, x, y);
            }
        }
    }

    // Update the original image data with the blurred result
    img.data = output.data;
}



/*********************************************************************
 * Image manipulation - Sobel for edge detection (Serial & Parallel) *
**********************************************************************/

void sobelEdgeDetection(Image& img, int threshold, float scaleFactor) {

    if (img.isColor) {
        std::cerr << "Sobel Edge Detection should be applied on grayscale images." << std::endl;
        return;
    }

    std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; // Sobel operator for X gradient
    std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; // Sobel operator for Y gradient

    std::vector<unsigned char> result(img.data.size(), 0); // Initialize result image with zeros
    int maxMagnitude = 0; // Used for normalization

    // First pass: Calculate the gradient magnitude and identify the maximum value
    for (int y = 1; y < img.height - 1; ++y) {
        for (int x = 1; x < img.width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int pixel = img.data[(y + i - 1) * img.width + (x + j - 1)];
                    sumX += pixel * gx[i * 3 + j];
                    sumY += pixel * gy[i * 3 + j];
                }
            }

            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            if (magnitude > maxMagnitude) maxMagnitude = magnitude; // Update maximum value
        }
    }

    // Second pass: Apply threshold and scaling
    for (int y = 1; y < img.height - 1; ++y) {
        for (int x = 1; x < img.width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int pixel = img.data[(y + i - 1) * img.width + (x + j - 1)];
                    sumX += pixel * gx[i * 3 + j];
                    sumY += pixel * gy[i * 3 + j];
                }
            }

            int magnitude = static_cast<int>((std::sqrt(sumX * sumX + sumY * sumY) / maxMagnitude) * 255 * scaleFactor);
            magnitude = (magnitude > threshold) ? magnitude : 0; // Apply threshold
            result[y * img.width + x] = std::min(std::max(magnitude, 0), 255);
        }
    }

    img.data = result;
}

void sobelEdgeDetectionParallel(Image& img, int threshold, float scaleFactor, int numThreads) {
    if (img.isColor) {
        std::cerr << "Sobel Edge Detection should be applied on grayscale images." << std::endl;
        return;
    }

    std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; // Sobel operator for X gradient
    std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; // Sobel operator for Y gradient

    std::vector<unsigned char> result(img.data.size(), 0);
    int maxMagnitude = 0;

    // First pass (Parallel): Calculate the gradient magnitude and identify the maximum value
    #pragma omp parallel for reduction(max:maxMagnitude) num_threads(numThreads) default(none) shared(img, gx, gy, result)
    for (int y = 1; y < img.height - 1; ++y) {
        for (int x = 1; x < img.width - 1; ++x) {
            int sumX = 0, sumY = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int pixel = img.data[(y + i - 1) * img.width + (x + j - 1)];
                    sumX += pixel * gx[i * 3 + j];
                    sumY += pixel * gy[i * 3 + j];
                }
            }

            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            if (magnitude > maxMagnitude) maxMagnitude = magnitude;
        }
    }

    // Ensure maxMagnitude is shared among threads before proceeding
    #pragma omp barrier

    // Second pass (Parallel): Apply threshold and scaling
    #pragma omp parallel for num_threads(numThreads) default(none) shared(img, gx, gy, result, maxMagnitude, threshold, scaleFactor)
    for (int y = 1; y < img.height - 1; ++y) {
        for (int x = 1; x < img.width - 1; ++x) {
            int sumX = 0, sumY = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int pixel = img.data[(y + i - 1) * img.width + (x + j - 1)];
                    sumX += pixel * gx[i * 3 + j];
                    sumY += pixel * gy[i * 3 + j];
                }
            }

            int magnitude = static_cast<int>((std::sqrt(sumX * sumX + sumY * sumY) / maxMagnitude) * 255 * scaleFactor);
            magnitude = (magnitude > threshold) ? magnitude : 0;
            result[y * img.width + x] = std::min(std::max(magnitude, 0), 255);
        }
    }

    img.data = result;
}
