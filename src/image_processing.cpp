
#include "image_processing.h"

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
void preprocessImage(Image& img, std::unordered_map<std::string, std::string> config) {
    
    std::cout << "\n ------ Pre-processing image ------ \n" << std::endl;
    
    // Paths and File Names
    std::string const input_folder = config["input_folder"];
    std::string const image_name = config["image_name"];
    std::string const image_format = config["image_format"];
    std::string const input_image = input_folder + image_name + image_format;
    std::string const output_folder = config["output_folder"] + image_name + "/";
    std::string const output_format = config["output_format"];
    std::cout << "Input Image: " << input_image << std::endl;
    std::cout << "Output Folder: " << output_folder << std::endl;
    std::cout << "Output Format: " << output_format << std::endl;
    
    // Parallel Processing
    const bool parallel = config["parallel"] == "true";
    const int thread_count = std::stoi(config["thread_count"]);
    std::cout << "Parallel Processing: " << (parallel ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Thread Count: " << thread_count << std::endl;
    
    // Gaussian Blur
    const bool gaussian_blur = config["gaussian_blur"] == "true";
    const int gb_kernel_size = std::stoi(config["gb_kernel_size"]);
    const float gb_sigma = std::stof(config["gb_sigma"]);
    std::cout << "Gaussian Blur: " << (gaussian_blur ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Kernel Size: " << gb_kernel_size << ", Sigma: " << gb_sigma << std::endl;
    
    // Sobel Edge Detection
    const bool sobel_edge_detection = config["sobel_edge_detection"] == "true";
    const int sed_threshold = std::stoi(config["sed_threshold"]);
    const float sed_scale_factor = std::stof(config["sed_scale_factor"]);
    std::cout << "Sobel Edge Detection: " << (sobel_edge_detection ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Threshold: " << sed_threshold << ", Scale Factor: " << sed_scale_factor << std::endl;
    
    // Histogram eq. & greyscale
    const bool histogram_equalization = config["histogram_equalization"] == "true";
    const bool greyscale_conversion = config["greyscale_conversion"] == "true";
    const bool sed_to_binary = config["sed_to_binary"] == "true";
    const bool restrict_colors = config["restrict_colors"] == "true";
    const int new_min = std::stoi(config["new_min"]);
    const int new_max = std::stoi(config["new_max"]);

    std::cout << "Histogram Equalization: " << (histogram_equalization ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Greyscale Conversion: " << (greyscale_conversion ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Sobel to Binary: " << (sed_to_binary ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Restrict colors: " << (restrict_colors ? "Enabled" : "Disabled") << std::endl;

    if (greyscale_conversion){

        std::cout << "\n[1] Converting the image to grayscale..." << std::endl;
        convertToGrayscale(img);
        saveImage(img, output_folder + image_name + "-1_greyscale" + image_format);

    }

    if (histogram_equalization){

        std::cout << "\n[2] Image (histogram) equalization ..." << std::endl;
        equalizeHistogram(img);
        saveImage(img, output_folder + image_name + "-2_equalized" + image_format);
    }

    if (gaussian_blur){

        std::cout << "\n[3] Blurring using a Gaussian filter ..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        if (parallel)
            parallelGaussianBlur(img, gb_kernel_size, gb_sigma, thread_count);
        else
            applyGaussianBlur(img, gb_kernel_size, gb_sigma);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
        saveImage(img, output_folder + image_name + "-3_blur" + image_format);

    }

    if (restrict_colors){
        std::cout << "\n[3.1] Restrict colors scale ..." << std::endl;
        restrictColorScale(img, new_min, new_max);
        saveImage(img, output_folder + image_name + "-3.1_restricted" + image_format);
    }

    if (sobel_edge_detection){

        std::cout << "\n[4] Applying Sobel for edge detection ..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        if (parallel)
            applySobelEdgeDetectionParallel(img, sed_threshold, sed_scale_factor);
        else
            applySobelEdgeDetection(img, sed_threshold, sed_scale_factor);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
        saveImage(img, output_folder + image_name + "-4_sobel" + image_format);
    }

    if (sed_to_binary){

        std::cout << "\n[4.1] Image to binary after Sobel ..." << std::endl;
        sobelToBinary(img);
        saveImage(img, output_folder + image_name + "-4.1_binary" + image_format);
    }
}


void restrictColorScale(Image& img, unsigned char newMin, unsigned char newMax) {
    if (newMin >= newMax) {
        std::cerr << "Invalid range: newMin must be less than newMax." << std::endl;
        return;
    }

    // Determine the current min and max values in the image
    auto minmax = std::minmax_element(img.data.begin(), img.data.end());
    unsigned char currentMin = *minmax.first;
    unsigned char currentMax = *minmax.second;

    std::cout << "Current Min: " << static_cast<int>(currentMin) << ", Current Max: " << static_cast<int>(currentMax) << std::endl;

    // Scale and shift the image data to the new range [newMin, newMax]
    for (unsigned char& pixel : img.data) {
        if (currentMax != currentMin) { // Prevent division by zero
            pixel = static_cast<unsigned char>((static_cast<float>(pixel - currentMin) / (currentMax - currentMin)) * (newMax - newMin) + newMin);
        } else {
            pixel = newMin; // All pixels are the same
        }
    }

    std::cout << "Adjustment completed. New range set from " << static_cast<int>(newMin) << " to " << static_cast<int>(newMax) << "." << std::endl;
}

/**
 * Applies the Hough Transform to the provided image, supporting both parallel and sequential execution. 
 * This function is called after preprocessing steps to effectively run the transform with the specified parameters.
 *
 * @param img The image on which to perform the Hough Transform.
 * @param config A map containing configuration options for the Hough Transform.
 * @return A 2D accumulator array representing the parameter space (rho, theta) with voting counts, indicating
 *         the presence and strength of lines in the image.
 */
std::vector<std::vector<int>> HoughTransformation(Image& img, std::unordered_map<std::string, std::string> config){

    std::cout << "\n ------ Hough Transform ------ " << std::endl;
    const int hough_vote_threshold = std::stoi(config["hough_vote_threshold"]); // Modificato in std::stoi
    const int hough_theta = std::stoi(config["hough_theta"]); // Modificato in std::sto
    const bool parallel = config["parallel"] == "true";
    const std::string version = config["HT_version"];
    const int samplig_rate = std::stoi(config["samplig_rate"]);

    std::cout << " - Version       :" << version << std::endl;
    std::cout << " - Parallel      :" << parallel << std::endl;
    std::cout << " - Vote Threshold:" << hough_vote_threshold << std::endl;
    std::cout << " - Theta         :" << hough_theta << std::endl;
    std::cout << " - Samplig Rate  :" << samplig_rate << " (only for probabilistic version)" <<std::endl;


    std::cout << "\n[5] Applying Hough Transform for line detection ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> accumulator;

    if (parallel){
        std::cout << "version:" + version;
        if (version == "basic")
            accumulator = applyHoughTransformParallel(img, hough_vote_threshold, hough_theta);
        
        else{
            throw std::invalid_argument("Hough transform version (" + version + ") not valid. Available versions: basic, probabilistic.");
        }
    }
    else{
        printf("in else");
        if (version == "basic"){
            accumulator = applyHoughTransform(img, hough_vote_threshold, hough_theta);
        }
        if (version == "probabilistic"){
            accumulator = applyProbabilisticHoughTransform(img, hough_vote_threshold, hough_theta, samplig_rate);
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;


    return accumulator;
}

/********************
 * Image processing *
*********************/

void sobelToBinary(Image& img) {
    if (img.data.empty()) return; // Do nothing if the image data is empty.

    for (unsigned char& pixel : img.data) {
        // Set all non-black pixels to white, keep black pixels unchanged.
        if (pixel != 0) {
            pixel = 255; // White
        }
        // If pixel == 0, it remains black.
    }

    std::cout << "-> Sobel output converted to binary image." << std::endl;
}

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
    std::cout << "-> Conversion to grayscale completed." << std::endl;
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

    // Calculate Cumulative Distribution Function (CDF)
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

    // Update image data using the equalized histogram
    for (size_t i = 0; i < imageSize; ++i) {
        img.data[i] = static_cast<unsigned char>(cdf[img.data[i]]);
    }
}

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

    // Print the kernel after creation
    std::cout << "-> Gaussian Kernel (" << kernelSize << "x" << kernelSize << "):\n" << std::endl;
    printGaussianKernel(kernel);

    return kernel;
}

void draw_lines_bresenham(int x0, int y0, int x1, int y1, std::vector<unsigned char>& rgbData, int width, int height) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            int idx = (y0 * width + x0) * 3; // Index for RGB data
            rgbData[idx] = 255;     // R (red)
            rgbData[idx + 1] = 0;   // G (green)
            rgbData[idx + 2] = 0;   // B (blue)
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_hough_lines(const std::vector<std::vector<int>>& accumulator, Image& image, std::unordered_map<std::string, std::string> config) {
    std::cout << "\n[6] Drawing Hough detected lines ..." << std::endl;

    convertToGrayscale(image);
    const int threshold = std::stoi(config["hough_vote_threshold"]); // Modificato in std::stoi

    std::vector<unsigned char> rgbData(image.width * image.height * 3);
    for (int i = 0; i < image.width * image.height; ++i) {
        rgbData[i * 3] = image.data[i];     // R
        rgbData[i * 3 + 1] = image.data[i]; // G
        rgbData[i * 3 + 2] = image.data[i]; // B
    }

    int maxRho = std::sqrt(image.width * image.width + image.height * image.height);

    // Iterate through the accumulator to find lines exceeding the vote threshold
    for (size_t rhoIndex = 0; rhoIndex < accumulator.size(); ++rhoIndex) {
        for (int theta = 0; theta < 180; ++theta) {
            if (accumulator[rhoIndex][theta] > threshold) {
                double rad = theta * M_PI / 180.0;
                double sinTheta = std::sin(rad);
                double cosTheta = std::cos(rad);
                double rho = rhoIndex - maxRho;

                int x1, y1, x2, y2;

                // Calculate the line endpoints
                if (sinTheta != 0) {
                    x1 = 0;
                    y1 = (rho - (x1 * cosTheta)) / sinTheta;
                    x2 = image.width - 1;
                    y2 = (rho - (x2 * cosTheta)) / sinTheta;
                } else {
                    x1 = rho / cosTheta;
                    y1 = 0;
                    x2 = x1;
                    y2 = image.height - 1;
                }

                // Draw the line
                draw_lines_bresenham(x1, y1, x2, y2, rgbData, image.width, image.height);
            }
        }
    }

    // Update the original image with modified RGB data
    image.data = std::move(rgbData);
    image.isColor = true;
}


// PARALLEL

void parallelGaussianBlur(Image& img, int kernelSize, float sigma, int numParts) {
    std::vector<std::vector<float>> kernel = calculateGaussianKernel(kernelSize, sigma);
    int overlap = kernelSize / 2; // Calculate the overlap based on half the kernel size

    // Split the image into parts with appropriate overlap
    std::vector<ImagePart> parts = splitImage(img, numParts, kernelSize);

    // Apply the Gaussian filter to each part in parallel
    int optimalThreads = std::min(numParts, omp_get_max_threads());
    omp_set_num_threads(optimalThreads);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numParts; i++) {
        applyGaussianBlurToPart(parts[i], kernel, overlap);
    }

    // Recombine the parts into the resulting image
    Image result = recombineImage(parts);

    // Update the original image with the result
    img.data = result.data;
    img.width = result.width;
    img.height = result.height;
}

void applyGaussianBlurToPart(ImagePart& imgPart, const std::vector<std::vector<float>>& kernel, int overlap) {
    int edge = kernel.size() / 2;
    std::vector<unsigned char> output(imgPart.width * imgPart.height, 0);

    for (int y = 0; y < imgPart.height; y++) {
        for (int x = 0; x < imgPart.width; x++) {
            float blurredPixel = 0.0;
            for (int ky = -edge; ky <= edge; ky++) {
                for (int kx = -edge; kx <= edge; kx++) {
                    int realY = std::max(0, std::min(y + ky, imgPart.height - 1));
                    int realX = std::max(0, std::min(x + kx, imgPart.width - 1));
                    blurredPixel += imgPart.data[realY * imgPart.width + realX] * kernel[ky + edge][kx + edge];
                }
            }
            output[y * imgPart.width + x] = static_cast<unsigned char>(std::min(std::max(int(blurredPixel), 0), 255));
        }
    }

    imgPart.data.swap(output);
}

void applySobelEdgeDetectionParallel(Image& img, int threshold, float scaleFactor) {
    if (img.isColor) {
        std::cerr << "Sobel Edge Detection should be applied on grayscale images." << std::endl;
        return;
    }

    std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; // Sobel operator for X gradient
    std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; // Sobel operator for Y gradient

    std::vector<unsigned char> result(img.data.size(), 0);
    int maxMagnitude = 0;

    // First pass (Parallel): Calculate the gradient magnitude and identify the maximum value
    #pragma omp parallel for reduction(max:maxMagnitude)
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
    #pragma omp parallel for
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

std::vector<std::vector<int>> applyHoughTransformParallel(const Image& image, int voteThreshold, int thetaResolution) {
    // Calculate the maximum possible value of rho based on image dimensions
    const double maxRho = std::sqrt(image.width * image.width + image.height * image.height);
    // Initialize the accumulator array with zero votes
    const int rhoSize = 2 * static_cast<int>(maxRho) + 1; // +1 to include the zero rho
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    // Parallelize the loop over image pixels with OpenMP, using collapse to merge two loops
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if (image.data[y * image.width + x] > 0) { // Check if the pixel is part of an edge
                // Iterate over possible theta values
                for (int theta = 0; theta < thetaResolution; ++theta) {
                    double rad = theta * M_PI / thetaResolution; // Convert theta to radians
                    int rho = static_cast<int>((x * cos(rad) + y * sin(rad)) + maxRho); // Calculate rho
                    #pragma omp atomic // Ensure thread safety when updating the accumulator
                    accumulator[rho][theta]++;
                }
            }
        }
    }

    int lineCount = 0;
    int maxVotes = 0;
    // Sequentially analyze the accumulator to count lines and find the max votes
    for (const auto &rhoRow : accumulator) {
        for (int vote : rhoRow) {
            if (vote > voteThreshold) {
                lineCount++;
                maxVotes = std::max(maxVotes, vote); // Update max votes if current is larger
            }
        }
    }

    std::cout << "-> Number of lines found: " << lineCount << std::endl;
    std::cout << "-> Maximum votes for a line: " << maxVotes << std::endl;

    return accumulator;
}

/********************
 * Images splitting *
*********************/

std::vector<ImagePart> splitImage(const Image& img, int parts, int overlap) {
    std::vector<ImagePart> imageParts;
    int basePartHeight = img.height / parts; // Calculate base height for each part
    int excess = img.height % parts; // Calculate excess rows to distribute among parts evenly

    for (int part = 0; part < parts; part++) {
        // Calculate starting row for the current part
        int startRow = part * basePartHeight + std::min(part, excess);
        // Calculate height of the current part, adding an extra row if this part includes excess
        int height = basePartHeight + (part < excess ? 1 : 0);
        // Calculate the actual start and end rows, considering the overlap
        int actualStartRow = std::max(0, startRow - overlap);
        int actualEndRow = std::min(img.height, startRow + height + overlap);
        // Calculate overlap at the top and bottom of the current part
        int overlapTop = startRow - actualStartRow;
        int overlapBottom = actualEndRow - (startRow + height);

        // Initialize the current ImagePart
        ImagePart imgPart;
        imgPart.startRow = startRow;
        imgPart.width = img.width;
        imgPart.height = actualEndRow - actualStartRow;
        imgPart.overlapTop = overlapTop;
        imgPart.overlapBottom = overlapBottom;
        imgPart.data.resize(imgPart.width * imgPart.height);

        // Copy the relevant part of the image data
        for (int row = 0; row < imgPart.height; row++) {
            std::copy_n(img.data.begin() + ((actualStartRow + row) * img.width),
                        img.width,
                        imgPart.data.begin() + (row * imgPart.width));
        }

        imageParts.push_back(imgPart);
    }

    return imageParts;
}

Image recombineImage(const std::vector<ImagePart>& parts) {
    if (parts.empty()) return Image(); // Handle empty input

    // Calculate the total width and height of the recombined image
    int width = parts.front().width;
    int height = std::accumulate(parts.begin(), parts.end(), 0,
                                 [&](int acc, const ImagePart& part) {
                                     return acc + part.height - part.overlapTop - part.overlapBottom;
                                 });

    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);

    int currentRow = 0;
    for (const auto& part : parts) {
        // Copy each part's data, skipping the overlapped rows
        for (int row = part.overlapTop; row < part.height - part.overlapBottom; row++) {
            std::copy_n(part.data.begin() + (row * part.width),
                        part.width,
                        img.data.begin() + ((currentRow++) * width));
        }
    }

    return img;
}

// SERIAL

void applyGaussianBlur(Image& img, int kernelSize, float sigma) {
    std::vector<std::vector<float>> kernel = calculateGaussianKernel(kernelSize, sigma);
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

void applySobelEdgeDetection(Image& img, int threshold, float scaleFactor) {

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

    // Second pass: Apply threshold and scaling (optional normalization)
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

    img.data = result; // Update the image data with the result
}

// STANDARD
std::vector<std::vector<int>> applyHoughTransform(const Image& image, int voteThreshold, int thetaResolution) {
    // Calculate the maximum distance (rho) possible within the image, diagonal distance
    const int maxRho = std::sqrt(image.width * image.width + image.height * image.height);
    // Initialize the accumulator with zeros, size determined by the possible range of rho values and theta resolution
    const int rhoSize = 2 * maxRho + 1; // +1 to include the center line (rho=0)
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    // Initialize variables for statistics
    int lineCount = 0;
    int maxVotes = 0;

    // Loop over every pixel in the image
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            // Process only edge pixels (assumed value > 0)
            if (image.data[y * image.width + x] > 0) {
                // For each edge pixel, consider every possible theta
                for (int theta = 0; theta < thetaResolution; ++theta) {
                    // Convert theta index to radians
                    double rad = theta * M_PI / thetaResolution;
                    // Calculate rho for this theta, offset by maxRho to ensure positive indices
                    int rho = static_cast<int>((x * cos(rad) + y * sin(rad)) + maxRho);
                    // Vote in the accumulator
                    accumulator[rho][theta]++;
                }
            }
        }
    }

    // After voting, count the number of potential lines that surpass the voteThreshold
    for (const auto &rhoRow : accumulator) {
        for (int vote : rhoRow) {
            if (vote > voteThreshold) {
                lineCount++; // Count potential lines
                maxVotes = std::max(maxVotes, vote); // Track the maximum vote count
            }
        }
    }

    // Output statistics about detected lines
    std::cout << "-> Number of lines found: " << lineCount << std::endl;
    std::cout << "-> Maximum votes for a line: " << maxVotes << std::endl;

    return accumulator;
}

std::vector<std::vector<int>> applyProbabilisticHoughTransform(const Image& image, int voteThreshold, int thetaResolution, int samplingRate) {
    // Calculate the maximum distance (rho) possible within the image, diagonal distance
    const int maxRho = std::sqrt(image.width * image.width + image.height * image.height);
    // Initialize the accumulator with zeros, size determined by the possible range of rho values and theta resolution
    const int rhoSize = 2 * maxRho + 1; // +1 to include the center line (rho=0)
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    // Initialize variables for statistics
    int lineCount = 0;
    int maxVotes = 0;

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, samplingRate - 1);

    // Loop over every pixel in the image
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            // Process only edge pixels (assumed value > 0) and sample pixels randomly
            if (image.data[y * image.width + x] > 0 && dis(gen) == 0) {
                // For each randomly selected edge pixel, consider a random set of theta
                for (int t = 0; t < thetaResolution; t += dis(gen) + 1) { // Random step in theta to reduce computation
                    // Convert theta index to radians
                    double rad = t * M_PI / thetaResolution;
                    // Calculate rho for this theta, offset by maxRho to ensure positive indices
                    int rho = static_cast<int>((x * cos(rad) + y * sin(rad)) + maxRho);
                    // Vote in the accumulator
                    accumulator[rho][t]++;
                }
            }
        }
    }

    // After voting, count the number of potential lines that surpass the voteThreshold
    for (const auto &rhoRow : accumulator) {
        for (int vote : rhoRow) {
            if (vote > voteThreshold) {
                lineCount++; // Count potential lines
                maxVotes = std::max(maxVotes, vote); // Track the maximum vote count
            }
        }
    }

    // Output statistics about detected lines
    std::cout << "-> Number of lines found: " << lineCount << std::endl;
    std::cout << "-> Maximum votes for a line: " << maxVotes << std::endl;

    return accumulator;
}
