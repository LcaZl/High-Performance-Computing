#include "HTs.h"

/**************
 *   SERIAL   *
***************/
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    
    bool probabilistic = (parameters["version"] == "PHT");
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);
    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    // Accumulate votes
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if ((!probabilistic && image.data[y * image.width + x] > 0) || 
                (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)) {
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                        accumulator[rhoIndex][thetaIndex]++;
                    }
                }
            }
        }
    }

    std::vector<Segment> segments;
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    for (int rhoIndex = 0; rhoIndex < rhoSize; ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
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
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);
    double samplingRate = std::stod(parameters["sampling_rate"]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<int> points(image.width * image.height);
    std::iota(points.begin(), points.end(), 0);
    std::vector<bool> processed(points.size(), false); // To mark points as processed

    std::vector<Segment> segments;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    while (!points.empty()) {
        int randomIndex = dis(gen) * points.size();
        int randomPoint = points[randomIndex];

        int x = randomPoint % image.width;
        int y = randomPoint / image.width;

        // Apply sampling rate
        if (dis(gen) > samplingRate) {
            points[randomIndex] = points.back();
            points.pop_back();
            continue;
        }

        if (image.data[y * image.width + x] > 0 && !processed[randomPoint]) {
            processed[randomPoint] = true;
            for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
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
                        for (int xi = 0; xi < image.width; ++xi) {
                            for (int yi = 0; yi < image.height; ++yi) {
                                if (image.data[yi * image.width + xi] > 0) {
                                    double calculatedRho = static_cast<double>((xi - centerX) * cosThetaValue + (yi - centerY) * sinThetaValue);
                                    if (std::abs(calculatedRho - rho) < 2) {
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
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    bool probabilistic = (parameters["version"] == "PHT");
    const double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    const int rhoSize = static_cast<int>(2 * rhoMax) + 1;

    std::vector<std::vector<int>> local_accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);

    // Calculate rows for each process
    int rows_per_process = image.height / world_size;
    int remaining_rows = image.height % world_size;

    // Adjust start and end rows to handle remaining rows
    int start_row = world_rank * rows_per_process + std::min(world_rank, remaining_rows);
    int end_row = start_row + rows_per_process + (world_rank < remaining_rows ? 1 : 0);

    // Perform the Hough Transform on the assigned rows
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if ((!probabilistic && image.data[y * image.width + x] > 0) ||
                (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)) {
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double rho = (x - centerX) * cos(thetaRad) + (y - centerY) * sin(thetaRad);
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                        local_accumulator[rhoIndex][thetaIndex]++;
                    }
                }
            }
        }
    }

    // Allocate the global accumulator only on the root process
    std::vector<std::vector<int>> accumulator;
    if (world_rank == 0) {
        accumulator.resize(rhoSize, std::vector<int>(thetaResolution, 0));
    }

    // Perform a reduction across all processes to combine results
    for (int i = 0; i < rhoSize; ++i) {
        MPI_Reduce(local_accumulator[i].data(), (world_rank == 0 ? accumulator[i].data() : nullptr), thetaResolution, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    std::vector<Segment> local_segments;
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);

    // Calculate rho ranges for each process
    int rhos_per_process = rhoSize / world_size;
    int remaining_rhos = rhoSize % world_size;

    int start_rho = world_rank * rhos_per_process + std::min(world_rank, remaining_rhos);
    int end_rho = start_rho + rhos_per_process + (world_rank < remaining_rhos ? 1 : 0);

    // Segment extraction by each process for its rho range
    if (world_rank == 0) {
        for (int rhoIndex = 0; rhoIndex < rhoSize; ++rhoIndex) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double thetaDeg = thetaRad * (180.0 / M_PI);
                    double rho = rhoIndex - rhoMax;
                    Point start, end;
                    std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);
                    local_segments.push_back(Segment{start, end, rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]});
                }
            }
        }
    } else {
        for (int rhoIndex = start_rho; rhoIndex < end_rho; ++rhoIndex) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                if (local_accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double thetaDeg = thetaRad * (180.0 / M_PI);
                    double rho = rhoIndex - rhoMax;
                    Point start, end;
                    std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);
                    local_segments.push_back(Segment{start, end, rho, thetaRad, thetaDeg, local_accumulator[rhoIndex][thetaIndex]});
                }
            }
        }
    }

    // Gather segments on the root process
    MPI_Datatype segmentType;
    createSegmentMPIType(&segmentType);

    std::vector<int> all_num_segments(world_size);
    int num_local_segments = local_segments.size();

    MPI_Gather(&num_local_segments, 1, MPI_INT, all_num_segments.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Segment> segments;
    if (world_rank == 0) {
        int total_segments = std::accumulate(all_num_segments.begin(), all_num_segments.end(), 0);
        segments.resize(total_segments);

        std::vector<int> displs(world_size);
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + all_num_segments[i - 1];
        }

        MPI_Gatherv(local_segments.data(), num_local_segments, segmentType,
                    segments.data(), all_num_segments.data(), displs.data(), segmentType, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(local_segments.data(), num_local_segments, segmentType,
                    nullptr, nullptr, nullptr, segmentType, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&segmentType);

    return {accumulator, segments};
}

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_MPI(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int thetaResolution = std::stoi(parameters["hough_theta"]);
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> local_accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);
    double samplingRate = std::stod(parameters["sampling_rate"]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<int> points(image.width * image.height);
    std::iota(points.begin(), points.end(), 0);
    std::vector<bool> processed(points.size(), false); // To mark points as processed

    // Calculate rows for each process
    int rows_per_process = image.height / world_size;
    int remaining_rows = image.height % world_size;

    // Adjust start and end rows to handle remaining rows
    int start_row = world_rank * rows_per_process + std::min(world_rank, remaining_rows);
    int end_row = start_row + rows_per_process + (world_rank < remaining_rows ? 1 : 0);

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    std::vector<Segment> local_segments;

    // Process only the assigned rows
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if (image.data[y * image.width + x] > 0 && dis(gen) <= samplingRate) {
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                        local_accumulator[rhoIndex][thetaIndex]++;
                        if (local_accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                            double thetaDeg = thetaRad * (180.0 / M_PI);
                            double sinThetaValue = sinTheta[thetaIndex];
                            double cosThetaValue = cosTheta[thetaIndex];
                            std::vector<Point> linePoints;

                            // Check points along the line
                            for (int xi = 0; xi < image.width; ++xi) {
                                for (int yi = 0; yi < image.height; ++yi) {
                                    if (image.data[yi * image.width + xi] > 0) {
                                        double calculatedRho = static_cast<double>((xi - centerX) * cosThetaValue + (yi - centerY) * sinThetaValue);
                                        if (std::abs(calculatedRho - rho) < 2) {
                                            if (!linePoints.empty() && (std::abs(linePoints.back().x - xi) > lineGap || std::abs(linePoints.back().y - yi) > lineGap)) {
                                                if (static_cast<int>(linePoints.size()) >= lineLength) {
                                                    local_segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, local_accumulator[rhoIndex][thetaIndex]});
                                                }
                                                linePoints.clear();
                                            }
                                            linePoints.push_back(Point{xi, yi});
                                        }
                                    }
                                }
                            }

                            if (static_cast<int>(linePoints.size()) >= lineLength) {
                                local_segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, local_accumulator[rhoIndex][thetaIndex]});
                            }

                            // Unvote the points
                            for (const auto& point : linePoints) {
                                int unvoteX = point.x;
                                int unvoteY = point.y;
                                double unvoteRho = (unvoteX - centerX) * cosThetaValue + (unvoteY - centerY) * sinThetaValue;
                                int unvoteRhoIndex = static_cast<int>(unvoteRho + rhoMax);
                                if (unvoteRhoIndex >= 0 && unvoteRhoIndex < rhoSize) {
                                    local_accumulator[unvoteRhoIndex][thetaIndex]--;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Allocate the global accumulator only on the root process
    std::vector<std::vector<int>> accumulator;
    if (world_rank == 0) {
        accumulator.resize(rhoSize, std::vector<int>(thetaResolution, 0));
    }

    // Perform a reduction across all processes to combine results
    for (int i = 0; i < rhoSize; ++i) {
        MPI_Reduce(local_accumulator[i].data(), (world_rank == 0 ? accumulator[i].data() : nullptr), thetaResolution, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Gather segments on the root process
    MPI_Datatype segmentType;
    createSegmentMPIType(&segmentType);

    std::vector<int> all_num_segments(world_size);
    int num_local_segments = local_segments.size();

    MPI_Gather(&num_local_segments, 1, MPI_INT, all_num_segments.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Segment> segments;
    if (world_rank == 0) {
        int total_segments = std::accumulate(all_num_segments.begin(), all_num_segments.end(), 0);
        segments.resize(total_segments);

        std::vector<int> displs(world_size);
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + all_num_segments[i - 1];
        }

        MPI_Gatherv(local_segments.data(), num_local_segments, segmentType,
                    segments.data(), all_num_segments.data(), displs.data(), segmentType, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(local_segments.data(), num_local_segments, segmentType,
                    nullptr, nullptr, nullptr, segmentType, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&segmentType);

    return {accumulator, segments};
}

/********************
 *  PARALLEL - OMP  *
*********************/
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> houghTransformParallel_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    // Center of the image
    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    int thetaResolution = std::stod(parameters["hough_theta"]);
    bool probabilistic = (parameters["version"] == "PHT" || parameters["version"] == "PPHT");
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int numThreads = std::stoi(parameters["omp_threads"]);

    // Calculate the maximum rho value
    const double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    const int rhoSize = static_cast<int>(2 * rhoMax) + 1;

    // Allocate the accumulator
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<std::vector<std::vector<int>>> local_accumulators(numThreads, std::vector<std::vector<int>>(rhoSize, std::vector<int>(thetaResolution, 0)));

    // Random device for probabilistic sampling
    std::random_device rd;

    // Parallel region with specified number of threads and static scheduling
    #pragma omp parallel num_threads(numThreads) default(none) shared(image, thetaResolution, probabilistic, samplingRate, rhoMax, rhoSize, centerX, centerY, accumulator, rd, local_accumulators, numThreads)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(rd() + thread_id);
        std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);

        // Parallelize both the outer and inner loops with collapse to flatten nested loops into a single parallel loop
        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                // Process only edge pixels (non-zero values)
                if ((!probabilistic && image.data[y * image.width + x] > 0) ||
                    (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)) {  // Probabilistic sampling
                    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                        double thetaRad = thetaIndex * (M_PI / thetaResolution);
                        double rho = (x - centerX) * cos(thetaRad) + (y - centerY) * sin(thetaRad);
                        int rhoIndex = static_cast<int>(rho + rhoMax);
                        if (rhoIndex >= 0 && rhoIndex < rhoSize) { // Ensure index is within bounds
                            local_accumulators[thread_id][rhoIndex][thetaIndex]++;
                        }
                    }
                }
            }
        }

        // Combine local_accumulator into the global accumulator
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < rhoSize; ++i) {
            for (int j = 0; j < thetaResolution; ++j) {
                for (int t = 0; t < numThreads; ++t) {
                    #pragma omp atomic
                    accumulator[i][j] += local_accumulators[t][i][j];
                }
            }
        }
    }

    std::vector<Segment> segments = (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT") ?
        linesExtractionParallel_OMP(accumulator, image, parameters) :
        linesProgressiveExtractionParallel_OMP(accumulator, image, parameters);

    return { accumulator, segments };
}

std::vector<Segment> linesExtractionParallel_OMP(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters) {

    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int thetaResolution = std::stod(parameters["hough_theta"]);
    int numThreads = std::stoi(parameters["omp_threads"]);

    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    std::vector<Segment> lines;
    std::vector<std::vector<Segment>> local_lines(numThreads);

    #pragma omp parallel num_threads(numThreads) default(none) shared(accumulator, image, voteThreshold, thetaResolution, rhoMax, local_lines)
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for collapse(2) schedule(static)
        for (size_t rhoIndex = 0; rhoIndex < accumulator.size(); ++rhoIndex) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double thetaDeg = thetaRad * (180.0 / M_PI);
                    double rho = rhoIndex - rhoMax;

                    Point start, end;
                    std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);
                    local_lines[thread_id].push_back(Segment(start, end, rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                }
            }
        }
    }

    // Combine local_lines into the global lines
    for (int t = 0; t < numThreads; ++t) {
        lines.insert(lines.end(), local_lines[t].begin(), local_lines[t].end());
    }

    return lines;
}

std::vector<Segment> linesProgressiveExtractionParallel_OMP(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    std::vector<Segment> segments;
    int numThreads = std::stoi(parameters["omp_threads"]);
    std::vector<std::vector<Segment>> local_segments(numThreads);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int thetaResolution = std::stod(parameters["hough_theta"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);

    #pragma omp parallel num_threads(numThreads) default(none) shared(accumulator, image, voteThreshold, thetaResolution, lineGap, lineLength, rhoMax, local_segments)
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for collapse(2) schedule(static)
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
                                            local_segments[thread_id].push_back(Segment(linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                                        }
                                        linePoints.clear();
                                    }
                                    linePoints.push_back(Point(x, y));
                                }
                            }
                        }
                    }

                    if (linePoints.size() >= static_cast<size_t>(lineLength)) {
                        local_segments[thread_id].push_back(Segment(linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]));
                    }
                }
            }
        }
    }

    // Combine local_segments into the global segments
    for (int t = 0; t < numThreads; ++t) {
        segments.insert(segments.end(), local_segments[t].begin(), local_segments[t].end());
    }

    return segments;
}


/***********************
 *  PARALLEL - Hybrid  *
************************/
/*
std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> houghTransformParallel_Hybrid(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int thetaResolution = std::stod(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    bool probabilistic = (parameters["version"] == "PHT" || parameters["version"] == "PPHT");
    const double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    const int rhoSize = static_cast<int>(2 * rhoMax) + 1;

    std::vector<std::vector<int>> local_accumulator(rhoSize, std::vector<int>(thetaResolution, 0));

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);

    // Calculate rows for each process
    int rows_per_process = image.height / world_size;
    int remaining_rows = image.height % world_size;

    // Adjust start and end rows to handle remaining rows
    int start_row = world_rank * rows_per_process + std::min(world_rank, remaining_rows);
    int end_row = start_row + rows_per_process + (world_rank < remaining_rows ? 1 : 0);

    // Perform the Hough Transform on the assigned rows with OpenMP parallelization
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if ((!probabilistic && image.data[y * image.width + x] > 0) ||
                (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)) {
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    double thetaRad = thetaIndex * (M_PI / thetaResolution);
                    double rho = (x - centerX) * cos(thetaRad) + (y - centerY) * sin(thetaRad);
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                        #pragma omp atomic
                        local_accumulator[rhoIndex][thetaIndex]++;
                    }
                }
            }
        }
    }

    // Allocate the global accumulator only on the root process
    std::vector<std::vector<int>> accumulator;
    if (world_rank == 0) {
        accumulator.resize(rhoSize, std::vector<int>(thetaResolution, 0));
    }

    // Perform a reduction across all processes to combine results
    for (int i = 0; i < rhoSize; ++i) {
        MPI_Reduce(local_accumulator[i].data(), (world_rank == 0 ? accumulator[i].data() : nullptr), thetaResolution, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    std::vector<Segment> segments;
    if (world_rank == 0){
        if (parameters["ht_line_extraction"] == "None")
            segments = (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT") ?
                linesExtraction(accumulator, image, parameters) :
                linesProgressiveExtraction(accumulator, image, parameters);

        else if (parameters["ht_line_extraction"] == "openMP")
            segments = (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT") ?
                    linesExtractionParallel_OMP(accumulator, image, parameters) :
                    linesProgressiveExtractionParallel_OMP(accumulator, image, parameters);
    }

    return { accumulator, segments };
}

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<Segment> segments;

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, image.width * image.height - 1);
    std::vector<int> points(image.width * image.height);
    std::iota(points.begin(), points.end(), 0);

    int voteThreshold = 1; // Initial threshold

    while (!points.empty()) {
        int randomIndex = dis(gen) % points.size();
        int randomPoint = points[randomIndex];
        std::swap(points[randomIndex], points.back());
        points.pop_back();

        int x = randomPoint % image.width;
        int y = randomPoint / image.width;

        if (image.data[y * image.width + x] > 0) {
            for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double rho = (x - centerX) * std::cos(thetaRad) + (y - centerY) * std::sin(thetaRad);
                int rhoIndex = static_cast<int>(rho + rhoMax);
                if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                    accumulator[rhoIndex][thetaIndex]++;
                    if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                        double thetaDeg = thetaRad * (180.0 / M_PI);
                        double sinTheta = std::sin(thetaRad);
                        double cosTheta = std::cos(thetaRad);
                        std::vector<Point> linePoints;

                        for (int xi = std::max(0, x - lineGap); xi < std::min(image.width, x + lineGap); ++xi) {
                            for (int yi = std::max(0, y - lineGap); yi < std::min(image.height, y + lineGap); ++yi) {
                                if (image.data[yi * image.width + xi] > 0) {
                                    double calculatedRho = (xi - centerX) * cosTheta + (yi - centerY) * sinTheta;
                                    if (std::abs(calculatedRho - rho) < 2) {
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

                        for (const auto& point : linePoints) {
                            int unvoteX = point.x;
                            int unvoteY = point.y;
                            double unvoteRho = (unvoteX - centerX) * cosTheta + (unvoteY - centerY) * sinTheta;
                            int unvoteRhoIndex = static_cast<int>(unvoteRho + rhoMax);
                            if (unvoteRhoIndex >= 0 && unvoteRhoIndex < rhoSize) {
                                accumulator[unvoteRhoIndex][thetaIndex]--;
                            }
                        }

                        // Dynamic adjustment of the vote threshold based on the binomial distribution approximation
                        double expectedNoiseVotes = voteThreshold * 0.99; // Adjust the confidence level if necessary
                        voteThreshold = std::max(voteThreshold, static_cast<int>(std::round(expectedNoiseVotes)));
                    }
                }
            }
        }
    }

    return {accumulator, segments};
}
*/