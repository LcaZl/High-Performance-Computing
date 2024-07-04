#include "HTs.h"

/**************
 *   SERIAL   *
***************/
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    
    bool probabilistic = (parameters["HT_version"] == "PHT");
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);

    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<Segment> segments;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100);

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;

    std::vector<int> points;
    int totalPoints = 0;
    int sampledPoints = 0;

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            if (image.data[y * image.width + x] > 0) {
                totalPoints++;
                if (!probabilistic || dis(gen) <= samplingRate){
                    points.push_back(y * image.width + x);
                    sampledPoints++;
                }
            }
        }
    }
    std::cout << "Total points: " << totalPoints << ", Sampled points: " << sampledPoints << ", Sampling rate: " << samplingRate << std::endl;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }


    // Accumulate votes
    for (int point : points) {
        int x = point % image.width;
        int y = point / image.width;
        for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
            double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
            int rhoIndex = static_cast<int>(rho + rhoMax);
            if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                accumulator[rhoIndex][thetaIndex]++;
            }
        }
    }


    for (int rhoIndex = 0; rhoIndex < rhoSize; rhoIndex++) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
            if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;
                Point start, end;
                std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);
                segments.push_back(Segment(start, end, rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
            }
        }
    }

    return {accumulator, segments};
}

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT(const Image& image, std::unordered_map<std::string, std::string>& parameters) {

    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);
    
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<Segment> segments;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100);

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;

    std::vector<int> points;
    int totalPoints = 0;
    int sampledPoints = 0;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            if (image.data[y * image.width + x] > 0) {
                totalPoints++;
                if (dis(gen) <= samplingRate){
                    points.push_back(y * image.width + x);
                    sampledPoints++;
                }
            }
        }
    }
    std::cout << "Total points: " << totalPoints << ", Sampled points: " << sampledPoints << ", Sampling rate: " << samplingRate << std::endl;
    
    std::vector<bool> processed(points.size(), false); // To mark points as processed
    std::uniform_int_distribution<> point_dis(0, points.size() - 1);
    
    while (!points.empty()) {
        int randomIndex = point_dis(gen);
        int randomPoint = points[randomIndex];
        int x = randomPoint % image.width;
        int y = randomPoint / image.width;

        if (!processed[randomIndex]) {
            processed[randomIndex] = true;
            for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
                double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                int rhoIndex = static_cast<int>(rho + rhoMax);
                if (rhoIndex >= 0 && rhoIndex < static_cast<int>(accumulator.size())) {
                    accumulator[rhoIndex][thetaIndex]++;
                    if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                        double thetaRad = thetaIndex * (M_PI / thetaResolution);
                        double thetaDeg = thetaRad * (180.0 / M_PI);
                        double sinThetaValue = sinTheta[thetaIndex];
                        double cosThetaValue = cosTheta[thetaIndex];
                        std::vector<Point> linePoints;

                        // Check points along the line
                        for (int yi = std::max(0, y - lineGap); yi < std::min(image.height, y + lineGap); yi++) {
                            for (int xi = std::max(0, x - lineGap); xi < std::min(image.width, x + lineGap); xi++) {
                                if (image.data[yi * image.width + xi] > 0) {
                                    double calculatedRho = static_cast<double>((xi - centerX) * cosThetaValue + (yi - centerY) * sinThetaValue);
                                    if (std::abs(calculatedRho - rho) < 5) {
                                        if (!linePoints.empty() && (std::abs(linePoints.back().x - xi) > lineGap || std::abs(linePoints.back().y - yi) > lineGap)) {
                                            if (static_cast<int>(linePoints.size()) >= lineLength) {
                                                segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]});
                                            }
                                            linePoints.clear();
                                        }
                                        linePoints.push_back(Point{xi, yi});
                                    }
                                }
                            }
                        }

                        if (static_cast<int>(linePoints.size()) >= lineLength) {
                            segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]});
                        }

                        // Unvote the points
                        for (const auto& point : linePoints) {
                            int unvoteX = point.x;
                            int unvoteY = point.y;
                            double unvoteRho = (unvoteX - centerX) * cosThetaValue + (unvoteY - centerY) * sinThetaValue;
                            int unvoteRhoIndex = static_cast<int>(unvoteRho + rhoMax);
                            if (unvoteRhoIndex >= 0 && unvoteRhoIndex < static_cast<int>(accumulator.size())) {
                                accumulator[unvoteRhoIndex][thetaIndex]--;
                            }
                        }
                    }
                }
            }
        }

        // Remove the processed point
        points[randomIndex] = points.back();
        points.pop_back();
    }

    return {accumulator, segments};
}


/********************
 *  PARALLEL - MPI  *
*********************/

// Function to flatten a 2D vector into a 1D vector
std::vector<int> flatten(const std::vector<std::vector<int>>& matrix) {
    std::vector<int> flat;
    for (const auto& row : matrix) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

// Function to reshape a 1D vector back into a 2D vector
std::vector<std::vector<int>> reshape(const std::vector<int>& flat, int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
    return matrix;
}

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI(Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool probabilistic = (parameters["HT_version"] == "PHT");
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);

    int imageWidth, imageHeight;
    std::vector<int> edgePoints;

    if (rank == 0) {
        imageWidth = image.width;
        imageHeight = image.height;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 100);

        for (int y = 0; y < image.height; y++) {
            for (int x = 0; x < image.width; x++) {
                if (image.data[y * image.width + x] > 0) {
                    if (!probabilistic || dis(gen) <= samplingRate) {
                        edgePoints.push_back(y * image.width + x);
                    }
                }
            }
        }
    }

    MPI_Bcast(&imageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int numPoints = edgePoints.size();
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int pointsPerProcess = numPoints / size;
    int extraPoints = numPoints % size;

    std::vector<int> localEdgePoints;
    MPI_Request request1, request2;

    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            int startIdx = i * pointsPerProcess + std::min(i, extraPoints);
            int endIdx = startIdx + pointsPerProcess + (i < extraPoints ? 1 : 0);
            int count = endIdx - startIdx;
            MPI_Isend(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request1);
            MPI_Isend(edgePoints.data() + startIdx, count, MPI_INT, i, 1, MPI_COMM_WORLD, &request2);
        }
        int startIdx = rank * pointsPerProcess + std::min(rank, extraPoints);
        int endIdx = startIdx + pointsPerProcess + (rank < extraPoints ? 1 : 0);
        localEdgePoints.insert(localEdgePoints.end(), edgePoints.begin() + startIdx, edgePoints.begin() + endIdx);
    } else {
        int count;
        MPI_Irecv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request1);
        MPI_Wait(&request1, MPI_STATUS_IGNORE);
        localEdgePoints.resize(count);
        MPI_Irecv(localEdgePoints.data(), count, MPI_INT, 0, 1, MPI_COMM_WORLD, &request2);
    }

    double rhoMax = std::sqrt(imageWidth * imageWidth + imageHeight * imageHeight);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<int> localAccumulator(rhoSize * thetaResolution, 0);
    std::vector<Segment> segments;

    double centerX = imageWidth / 2.0;
    double centerY = imageHeight / 2.0;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    if (rank != 0)
        MPI_Wait(&request2, MPI_STATUS_IGNORE);

    for (int point : localEdgePoints) {
        int x = point % imageWidth;
        int y = point / imageWidth;
        for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
            double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
            int rhoIndex = static_cast<int>(rho + rhoMax);
            if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                localAccumulator[rhoIndex * thetaResolution + thetaIndex]++;
            }
        }
    }

    std::vector<int> globalAccumulatorFlatten(localAccumulator.size(), 0);
    MPI_Reduce(localAccumulator.data(), globalAccumulatorFlatten.data(), localAccumulator.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<std::vector<int>> globalAccumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    if (rank == 0) {
        for (int i = 0; i < rhoSize; i++) {
            for (int j = 0; j < thetaResolution; j++) {
                globalAccumulator[i][j] = globalAccumulatorFlatten[i * thetaResolution + j];
            }
        }

        // Detect segments from the global accumulator
        for (int rhoIndex = 0; rhoIndex < rhoSize; rhoIndex++) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
                if (globalAccumulator[rhoIndex][thetaIndex] > voteThreshold) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double thetaDeg = thetaRad * (180.0 / M_PI);
                    double rho = rhoIndex - rhoMax;
                    Point start, end;
                    std::tie(start, end) = calculateEndpoints(rho, thetaRad, imageWidth, imageHeight);
                    segments.push_back(Segment(start, end, rho, thetaRad, thetaDeg, globalAccumulator[rhoIndex][thetaIndex]));
                }
            }
        }
    }

    return {globalAccumulator, segments};
}

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_MPI(Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);

    int imageWidth = image.width;
    int imageHeight = image.height;
    std::vector<int> edgePoints;

    // Broadcast image dimensions to all processes
    MPI_Bcast(&imageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Serialize the image data to broadcast
    int imageSize = imageWidth * imageHeight;
    std::vector<unsigned char> imageData;
    if (rank == 0) {
        imageData = image.data;
    } else {
        imageData.resize(imageSize);
    }

    MPI_Bcast(imageData.data(), imageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Recreate the image object in each process
    Image localImage;
    localImage.width = imageWidth;
    localImage.height = imageHeight;
    localImage.data = imageData;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100);

    // Calculate edge points in each process
    for (int y = 0; y < localImage.height; y++) {
        for (int x = 0; x < localImage.width; x++) {
            if (localImage.data[y * localImage.width + x] > 0) {
                if (dis(gen) <= samplingRate) {
                    edgePoints.push_back(y * localImage.width + x);
                }
            }
        }
    }

    double rhoMax = std::sqrt(imageWidth * imageWidth + imageHeight * imageHeight);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<int> localAccumulator(rhoSize * thetaResolution, 0);
    std::vector<Segment> segments;

    double centerX = imageWidth / 2.0;
    double centerY = imageHeight / 2.0;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    std::vector<bool> processed(edgePoints.size(), false); // To mark points as processed
    std::uniform_int_distribution<> point_dis(0, edgePoints.size() - 1);

    while (!edgePoints.empty()) {
        int randomIndex = point_dis(gen);
        int randomPoint = edgePoints[randomIndex];
        int x = randomPoint % imageWidth;
        int y = randomPoint / imageWidth;

        if (!processed[randomIndex]) {
            processed[randomIndex] = true;
            for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
                double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                int rhoIndex = static_cast<int>(rho + rhoMax);
                if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                    localAccumulator[rhoIndex * thetaResolution + thetaIndex]++;
                    if (localAccumulator[rhoIndex * thetaResolution + thetaIndex] > voteThreshold) {
                        double thetaRad = thetaIndex * (M_PI / thetaResolution);
                        double thetaDeg = thetaRad * (180.0 / M_PI);
                        double sinThetaValue = sinTheta[thetaIndex];
                        double cosThetaValue = cosTheta[thetaIndex];
                        std::vector<Point> linePoints;

                        // Check points along the line
                        for (int yi = std::max(0, y - lineGap); yi < std::min(imageHeight, y + lineGap); yi++) {
                            for (int xi = std::max(0, x - lineGap); xi < std::min(imageWidth, x + lineGap); xi++) {
                                if (localImage.data[yi * imageWidth + xi] > 0) {
                                    double calculatedRho = static_cast<double>((xi - centerX) * cosThetaValue + (yi - centerY) * sinThetaValue);
                                    if (std::abs(calculatedRho - rho) < 5) {
                                        if (!linePoints.empty() && (std::abs(linePoints.back().x - xi) > lineGap || std::abs(linePoints.back().y - yi) > lineGap)) {
                                            if (static_cast<int>(linePoints.size()) >= lineLength) {
                                                segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, localAccumulator[rhoIndex * thetaResolution + thetaIndex]});
                                            }
                                            linePoints.clear();
                                        }
                                        linePoints.push_back(Point{xi, yi});
                                    }
                                }
                            }
                        }

                        if (static_cast<int>(linePoints.size()) >= lineLength) {
                            segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, localAccumulator[rhoIndex * thetaResolution + thetaIndex]});
                        }

                        // Unvote the points
                        for (const auto& point : linePoints) {
                            int unvoteX = point.x;
                            int unvoteY = point.y;
                            double unvoteRho = (unvoteX - centerX) * cosThetaValue + (unvoteY - centerY) * sinThetaValue;
                            int unvoteRhoIndex = static_cast<int>(unvoteRho + rhoMax);
                            if (unvoteRhoIndex >= 0 && unvoteRhoIndex < rhoSize) {
                                localAccumulator[unvoteRhoIndex * thetaResolution + thetaIndex]--;
                            }
                        }
                    }
                }
            }
        }

        // Remove the processed point
        edgePoints[randomIndex] = edgePoints.back();
        edgePoints.pop_back();
    }

    // Combine all accumulators
    std::vector<int> globalAccumulatorFlatten(localAccumulator.size(), 0);
    MPI_Reduce(localAccumulator.data(), globalAccumulatorFlatten.data(), localAccumulator.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<std::vector<int>> globalAccumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    if (rank == 0) {
        globalAccumulator = reshape(globalAccumulatorFlatten, rhoSize, thetaResolution);
    }

    // Combine all segments
    MPI_Datatype segmentType;
    createSegmentMPIType(&segmentType);

    int localSegmentCount = segments.size();
    std::vector<int> segmentCounts(size);
    MPI_Gather(&localSegmentCount, 1, MPI_INT, segmentCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displsSegments(size);
    if (rank == 0) {
        displsSegments[0] = 0;
        for (int i = 1; i < size; ++i) {
            displsSegments[i] = displsSegments[i - 1] + segmentCounts[i - 1];
        }
    }

    std::vector<Segment> allSegments;
    if (rank == 0) {
        allSegments.resize(std::accumulate(segmentCounts.begin(), segmentCounts.end(), 0));
    }

    MPI_Gatherv(segments.data(), localSegmentCount, segmentType, allSegments.data(), segmentCounts.data(), displsSegments.data(), segmentType, 0, MPI_COMM_WORLD);

    MPI_Type_free(&segmentType);

    if (rank == 0) {
        return {globalAccumulator, allSegments};
    } else {
        return {globalAccumulator, segments}; // segments is empty for non-root processes
    }
}

/********************
 *  PARALLEL - OMP  *
*********************/

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    bool probabilistic = (parameters["HT_version"] == "PHT");
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int numThreads = std::stoi(parameters["omp_threads"]);

    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;

    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<Segment> segments;
    std::random_device rd;
    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    std::vector<int> points;
    int totalPoints = 0;

    #pragma omp parallel num_threads(numThreads)
    {
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_int_distribution<> dis(0, 100);

        #pragma omp for collapse(2) reduction(+:totalPoints)
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                if (image.data[y * image.width + x] > 0) {
                    ++totalPoints;
                    if (!probabilistic || dis(gen) <= samplingRate) {
                        #pragma omp critical
                        points.push_back(y * image.width + x);
                    }
                }
            }
        }
    }

    // Accumulate votes
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        int x = points[i] % image.width;
        int y = points[i] / image.width;
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
            int rhoIndex = static_cast<int>(rho + rhoMax);
            if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                #pragma omp atomic
                accumulator[rhoIndex][thetaIndex]++;
            }
        }
    }

    // Identify segments
    #pragma omp parallel for collapse(2) num_threads(numThreads)
    for (int rhoIndex = 0; rhoIndex < rhoSize; ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;
                Point start, end;
                std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);

                #pragma omp critical
                segments.push_back(Segment(start, end, rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
            }
        }
    }

    return {accumulator, segments};
}


/***********************
 *  PARALLEL - Hybrid  *
************************/

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI_OMP(Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool probabilistic = (parameters["HT_version"] == "PHT");
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int numThreads = std::stoi(parameters["omp_threads"]);

    int imageWidth, imageHeight;
    std::vector<int> edgePoints;

    if (rank == 0) {
        imageWidth = image.width;
        imageHeight = image.height;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 100);

        for (int y = 0; y < image.height; y++) {
            for (int x = 0; x < image.width; x++) {
                if (image.data[y * image.width + x] > 0) {
                    if (!probabilistic || dis(gen) <= samplingRate) {
                        edgePoints.push_back(y * image.width + x);
                    }
                }
            }
        }
    }

    MPI_Bcast(&imageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int numPoints = edgePoints.size();
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int pointsPerProcess = numPoints / size;
    int extraPoints = numPoints % size;

    std::vector<int> localEdgePoints;
    MPI_Request request1, request2;

    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            int startIdx = i * pointsPerProcess + std::min(i, extraPoints);
            int endIdx = startIdx + pointsPerProcess + (i < extraPoints ? 1 : 0);
            int count = endIdx - startIdx;
            MPI_Isend(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request1);
            MPI_Isend(edgePoints.data() + startIdx, count, MPI_INT, i, 1, MPI_COMM_WORLD, &request2);
        }
        int startIdx = rank * pointsPerProcess + std::min(rank, extraPoints);
        int endIdx = startIdx + pointsPerProcess + (rank < extraPoints ? 1 : 0);
        localEdgePoints.insert(localEdgePoints.end(), edgePoints.begin() + startIdx, edgePoints.begin() + endIdx);
    } else {
        int count;
        MPI_Irecv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request1);
        MPI_Wait(&request1, MPI_STATUS_IGNORE);
        localEdgePoints.resize(count);
        MPI_Irecv(localEdgePoints.data(), count, MPI_INT, 0, 1, MPI_COMM_WORLD, &request2);
    }

    double rhoMax = std::sqrt(imageWidth * imageWidth + imageHeight * imageHeight);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<int> localAccumulator(rhoSize * thetaResolution, 0);
    std::vector<Segment> segments;

    double centerX = imageWidth / 2.0;
    double centerY = imageHeight / 2.0;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    #pragma omp parallel for num_threads(numThreads)
    for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    if (rank != 0)
        MPI_Wait(&request2, MPI_STATUS_IGNORE);

    // Parallelize the vote accumulation
    #pragma omp parallel for num_threads(numThreads) collapse(2)
    for (int i = 0; i < static_cast<int>(localEdgePoints.size()); ++i) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            int x = localEdgePoints[i] % imageWidth;
            int y = localEdgePoints[i] / imageWidth;
            double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
            int rhoIndex = static_cast<int>(rho + rhoMax);
            if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                #pragma omp atomic
                localAccumulator[rhoIndex * thetaResolution + thetaIndex]++;
            }
        }
    }

    std::vector<int> globalAccumulatorFlatten(localAccumulator.size(), 0);
    MPI_Reduce(localAccumulator.data(), globalAccumulatorFlatten.data(), localAccumulator.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<std::vector<int>> globalAccumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    if (rank == 0) {
        for (int i = 0; i < rhoSize; i++) {
            for (int j = 0; j < thetaResolution; j++) {
                globalAccumulator[i][j] = globalAccumulatorFlatten[i * thetaResolution + j];
            }
        }

        // Detect segments from the global accumulator
        #pragma omp parallel for collapse(2) num_threads(numThreads) default(none) shared(globalAccumulator, rhoSize, thetaResolution, voteThreshold, imageWidth, imageHeight, segments, centerX, centerY, cosTheta, sinTheta, rhoMax)
        for (int rhoIndex = 0; rhoIndex < rhoSize; rhoIndex++) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; thetaIndex++) {
                if (globalAccumulator[rhoIndex][thetaIndex] > voteThreshold) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double thetaDeg = thetaRad * (180.0 / M_PI);
                    double rho = rhoIndex - rhoMax;
                    Point start, end;
                    std::tie(start, end) = calculateEndpoints(rho, thetaRad, imageWidth, imageHeight);
                    #pragma omp critical
                    segments.push_back(Segment(start, end, rho, thetaRad, thetaDeg, globalAccumulator[rhoIndex][thetaIndex]));
                }
            }
        }
    }

    return {globalAccumulator, segments};
}


/**
 * 
 * std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    double samplingRate = std::stod(parameters["sampling_rate"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);
    int numThreads = std::stoi(parameters["omp_threads"]);

    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;

    std::vector<std::vector<int>> globalAccumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<Segment> segments;
    std::vector<int> points;
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;

    points.reserve(image.width * image.height);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100);

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if (image.data[y * image.width + x] > 0 && dis(gen) <= samplingRate) {
                points.push_back(y * image.width + x);
            }
        }
    }

    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    std::vector<bool> processed(points.size(), false);

    #pragma omp parallel num_threads(numThreads) 
    {
        std::mt19937 threadGen(rd());
        std::uniform_int_distribution<> point_dis(0, points.size() - 1);

        std::vector<std::vector<int>> localAccumulator(rhoSize, std::vector<int>(thetaResolution, 0));

        #pragma omp for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(points.size()); ++idx) {
            int randomIndex = point_dis(threadGen);
            int randomPoint = points[randomIndex];

            int x = randomPoint % image.width;
            int y = randomPoint / image.width;

            if (!processed[randomIndex]) {
                processed[randomIndex] = true;
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                        localAccumulator[rhoIndex][thetaIndex]++;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < rhoSize; ++i) {
                for (int j = 0; j < thetaResolution; ++j) {
                    globalAccumulator[i][j] += localAccumulator[i][j];
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) num_threads(numThreads) 
    for (int rhoIndex = 0; rhoIndex < rhoSize; ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            if (globalAccumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;
                double sinThetaValue = sinTheta[thetaIndex];
                double cosThetaValue = cosTheta[thetaIndex];
                std::vector<Point> linePoints;

                for (int yi = 0; yi < image.height; ++yi) {
                    for (int xi = 0; xi < image.width; ++xi) {
                        if (image.data[yi * image.width + xi] > 0) {
                            double calculatedRho = (xi - centerX) * cosThetaValue + (yi - centerY) * sinThetaValue;
                            if (std::abs(calculatedRho - rho) < 2) {
                                if (!linePoints.empty() && (std::abs(linePoints.back().x - xi) > lineGap || std::abs(linePoints.back().y - yi) > lineGap)) {
                                    if (static_cast<int>(linePoints.size()) >= lineLength) {
                                        #pragma omp critical
                                        segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, globalAccumulator[rhoIndex][thetaIndex]});
                                    }
                                    linePoints.clear();
                                }
                                linePoints.push_back(Point{xi, yi});
                            }
                        }
                    }
                }

                if (static_cast<int>(linePoints.size()) >= lineLength) {
                    #pragma omp critical
                    segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, globalAccumulator[rhoIndex][thetaIndex]});
                }

                for (const auto& point : linePoints) {
                    int unvoteX = point.x;
                    int unvoteY = point.y;
                    double unvoteRho = (unvoteX - centerX) * cosThetaValue + (unvoteY - centerY) * sinThetaValue;
                    int unvoteRhoIndex = static_cast<int>(unvoteRho + rhoMax);
                    if (unvoteRhoIndex >= 0 && unvoteRhoIndex < rhoSize) {
                        #pragma omp atomic
                        globalAccumulator[unvoteRhoIndex][thetaIndex]--;
                    }
                }
            }
        }
    }

    return {globalAccumulator, segments};
}
 */