#include "HTs.h"

/**************
 *   SERIAL   *
***************/

std::vector<std::vector<int>> houghTransform(const Image& image, int thetaResolution, bool probabilistic, int samplingRate) {
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    std::random_device rd; // Seed with a non-deterministic value if available
    std::mt19937 gen(rd());
    // Distribution to control the sampling frequency of pixels
    std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if ((!probabilistic && image.data[y * image.width + x] > 0) ||
                (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)
            ) {
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    // thetaResolution is in degree, from parameters file.
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double rho = (x - centerX) * cos(thetaRad) + (y - centerY) * sin(thetaRad);
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) { // Ensure index is within bounds
                        accumulator[rhoIndex][thetaIndex]++;
                    }
                }
            }
        }
    }

    return accumulator;
}

std::vector<Segment> linesExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, int voteThreshold, int thetaResolution) {
    std::vector<Segment> lines;
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);

    for (int rhoIndex = 0; rhoIndex < static_cast<int>(accumulator.size()); ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;

                Point start, end;
                std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);
                
                lines.push_back(Segment(start, end, rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
            }
        }
    }

    return lines;
}

std::vector<Segment> linesProgressiveExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, int voteThreshold, int thetaResolution, int lineGap, int lineLength) {
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    std::vector<Segment> segments;
    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;

    for (size_t rhoIndex = 0; rhoIndex < accumulator.size(); ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;

                double sinTheta = std::sin(thetaRad);
                double cosTheta = std::cos(thetaRad);
                std::vector<Point> linePoints;

                // Scan through the image to collect points that are on the calculated line
                for (int x = 0; x < image.width; ++x) {
                    for (int y = 0; y < image.height; ++y) {
                        if (image.data[y * image.width + x] > 0) {
                            double calculatedRho = static_cast<double>((x - centerX) * cosTheta + (y - centerY) * sinTheta);
                            if (std::abs(calculatedRho - rho) < 2) {  // Consider a tolerance for rho
                                if (!linePoints.empty() && (std::abs(linePoints.back().x - x) > lineGap || std::abs(linePoints.back().y - y) > lineGap)) {
                                    if (static_cast<int>(linePoints.size()) >= lineLength) {
                                        // Save the segment if it's long enough
                                        segments.push_back(Segment(linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                                    }
                                    linePoints.clear();
                                }
                                linePoints.push_back(Point(x, y));
                            }
                        }
                    }
                }

                // Check if there is a remaining valid segment
                if (static_cast<int>(linePoints.size()) >= lineLength) {
                    segments.push_back(Segment(linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                }
            }
        }
    }

    return segments;
}

/**************
 *  PARALLEL  *
***************/

std::vector<std::vector<int>> parallelHoughTransform(const Image& image, int thetaResolution, bool probabilistic, int samplingRate, int threadCount) {
    const double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    const int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    std::random_device rd; // Seed with a non-deterministic value if available
    std::mt19937 gen(rd());
    // Distribution to control the sampling frequency of pixels
    std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);

    // Set the number of threads in the OpenMP environment.
    omp_set_num_threads(std::min(threadCount, omp_get_max_threads()));

    // Parallelize both the outer and inner loops with collapse to flatten nested loops into a single parallel loop
    #pragma omp parallel for collapse(2)

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            // Process only edge pixels (non-zero values)
            if ((!probabilistic && image.data[y * image.width + x] > 0) ||
                (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)
            ) {  // Probabilistic sampling
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double rho = (x - centerX) * cos(thetaRad) + (y - centerY) * sin(thetaRad);
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) { // Ensure index is within bounds
                        #pragma omp atomic
                        accumulator[rhoIndex][thetaIndex]++;
                    }                
                }
            }
        }
    }

    return accumulator;
}

std::vector<Segment> linesExtractionParallel(const std::vector<std::vector<int>>& accumulator, const Image& image, int voteThreshold, int thetaResolution, int threadCount) {
    std::vector<Segment> lines;
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);

    omp_set_num_threads(std::min(threadCount, omp_get_max_threads()));

    #pragma omp parallel
    {
        std::vector<Segment> local_lines;

        #pragma omp for nowait
        for (size_t rhoIndex = 0; rhoIndex < accumulator.size(); ++rhoIndex) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double thetaDeg = thetaRad * (180.0 / M_PI);
                    double rho = rhoIndex - rhoMax;

                    Point start, end;
                    std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);
                    local_lines.push_back(Segment(start, end, rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                }
            }
        }

        #pragma omp critical
        lines.insert(lines.end(), local_lines.begin(), local_lines.end());
    }

    return lines;
}



std::vector<Segment> linesProgressiveExtractionParallel(const std::vector<std::vector<int>>& accumulator, const Image& image, int voteThreshold, int thetaResolution, int lineGap, int lineLength, int threadCount) {
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    std::vector<Segment> segments;

    omp_set_num_threads(std::min(threadCount, omp_get_max_threads()));

    #pragma omp parallel
    {
        std::vector<Segment> local_segments;

        #pragma omp for collapse(2) nowait
        for (size_t rhoIndex = 0; rhoIndex < accumulator.size(); ++rhoIndex) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {

                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double thetaDeg = thetaRad * (180.0 / M_PI);
                    double rho = rhoIndex - rhoMax;
                    double sinTheta = sin(thetaRad);
                    double cosTheta = cos(thetaRad);
                    std::vector<Point> linePoints;

                    // Scan through the image to collect points that are on the calculated line
                    for (int x = 0; x < image.width; ++x) {
                        for (int y = 0; y < image.height; ++y) {
                            if (image.data[y * image.width + x] > 0) {
                                int calculatedRho = static_cast<int>((x - image.width / 2) * cosTheta + (y - image.height / 2) * sinTheta);
                                if (std::abs(calculatedRho - rho) < 2) {  // Tolerance for rho
                                    if (!linePoints.empty() && (std::abs(linePoints.back().x - x) > lineGap || std::abs(linePoints.back().y - y) > lineGap)) {
                                        if (linePoints.size() >= static_cast<size_t>(lineLength)) {
                                            local_segments.push_back(Segment(linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                                        }
                                        linePoints.clear();
                                    }
                                    linePoints.push_back(Point(x, y));
                                }
                            }
                        }
                    }

                    if (linePoints.size() >= static_cast<size_t>(lineLength)) {
                        local_segments.push_back(Segment(linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                    }
                }
            }
        }

        #pragma omp critical
        segments.insert(segments.end(), local_segments.begin(), local_segments.end());
    }

    return segments;
}


