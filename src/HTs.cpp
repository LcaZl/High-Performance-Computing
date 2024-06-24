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

    std::vector<int> points;
    points.reserve(image.width * image.height);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if (image.data[y * image.width + x] > 0) {
                points.push_back(y * image.width + x);
            }
        }
    }
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
                            for (int yi = std::max(0, y - lineGap); yi < std::min(image.height, y + lineGap); ++yi) {
                                for (int xi = std::max(0, x - lineGap); xi < std::min(image.width, x + lineGap); ++xi) {
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

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    bool probabilistic = (parameters["version"] == "PHT");
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int numThreads = std::stoi(parameters["omp_threads"]);
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<std::vector<std::vector<int>>> local_accumulators(numThreads, std::vector<std::vector<int>>(rhoSize, std::vector<int>(thetaResolution, 0)));

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    std::random_device rd;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    #pragma omp parallel num_threads(numThreads) default(none) shared(image, thetaResolution, probabilistic, samplingRate, rhoMax, rhoSize, centerX, centerY, local_accumulators, rd, cosTheta, sinTheta, numThreads, accumulator)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(rd() + thread_id);
        std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);

        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                if ((!probabilistic && image.data[y * image.width + x] > 0) ||
                    (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)) {
                    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                        double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                        int rhoIndex = static_cast<int>(rho + rhoMax);
                        if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                            local_accumulators[thread_id][rhoIndex][thetaIndex]++;
                        }
                    }
                }
            }
        }

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

    std::vector<Segment> segments;
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    
    #pragma omp parallel for collapse(2) num_threads(numThreads) default(none) shared(accumulator, rhoSize, thetaResolution, voteThreshold, image, centerX, centerY, cosTheta, sinTheta, segments, rhoMax)
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


std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> PPHT_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int thetaResolution = std::stoi(parameters["hough_theta"]);
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    int numThreads = std::stoi(parameters["omp_threads"]);
    std::vector<std::vector<int>> accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<std::vector<std::vector<int>>> local_accumulators(numThreads, std::vector<std::vector<int>>(rhoSize, std::vector<int>(thetaResolution, 0)));

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);
    double samplingRate = std::stod(parameters["sampling_rate"]);

    std::random_device rd;

    std::vector<int> points;
    points.reserve(image.width * image.height);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if (image.data[y * image.width + x] > 0) {
                points.push_back(y * image.width + x);
            }
        }
    }
    std::vector<bool> processed(points.size(), false); // To mark points as processed

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    // Point processing and voting
    #pragma omp parallel num_threads(numThreads) default(none) shared(image, thetaResolution, rhoMax, rhoSize, centerX, centerY, local_accumulators, rd, cosTheta, sinTheta, numThreads, samplingRate, processed, points)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(rd() + thread_id);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        while (true) {
            int randomIndex;
            int randomPoint;
            #pragma omp critical
            {
                if (points.empty()) {
                    randomIndex = -1;
                } else {
                    randomIndex = dis(gen) * points.size();
                    randomPoint = points[randomIndex];
                    points[randomIndex] = points.back();
                    points.pop_back();
                }
            }

            if (randomIndex == -1) break;

            int x = randomPoint % image.width;
            int y = randomPoint / image.width;

            if (dis(gen) > samplingRate) {
                continue;
            }

            if (!processed[randomIndex]) {
                processed[randomIndex] = true;
                for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                    double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                    int rhoIndex = static_cast<int>(rho + rhoMax);
                    if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                        local_accumulators[thread_id][rhoIndex][thetaIndex]++;
                    }
                }
            }
        }
    }

    // Combine local accumulators into the global accumulator
    #pragma omp parallel for collapse(2) schedule(static) num_threads(numThreads)
    for (int i = 0; i < rhoSize; ++i) {
        for (int j = 0; j < thetaResolution; ++j) {
            for (int t = 0; t < numThreads; ++t) {
                #pragma omp atomic
                accumulator[i][j] += local_accumulators[t][i][j];
            }
        }
    }

    std::vector<Segment> segments;

    // Segment extraction
    #pragma omp parallel for collapse(2) num_threads(numThreads) default(none) shared(accumulator, rhoSize, thetaResolution, voteThreshold, image, centerX, centerY, cosTheta, sinTheta, segments, lineGap, lineLength, rhoMax)
    for (int rhoIndex = 0; rhoIndex < rhoSize; ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;
                double sinThetaValue = sinTheta[thetaIndex];
                double cosThetaValue = cosTheta[thetaIndex];
                std::vector<Point> linePoints;

                // Check points along the line
                for (int yi = 0; yi < image.height; ++yi) {
                    for (int xi = 0; xi < image.width; ++xi) {
                        if (image.data[yi * image.width + xi] > 0) {
                            double calculatedRho = static_cast<double>((xi - centerX) * cosThetaValue + (yi - centerY) * sinThetaValue);
                            if (std::abs(calculatedRho - rho) < 2) {
                                if (!linePoints.empty() && (std::abs(linePoints.back().x - xi) > lineGap || std::abs(linePoints.back().y - yi) > lineGap)) {
                                    if (static_cast<int>(linePoints.size()) >= lineLength) {
                                        #pragma omp critical
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
                    #pragma omp critical
                    segments.push_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]});
                }

                // Unvote the points
                for (const auto& point : linePoints) {
                    int unvoteX = point.x;
                    int unvoteY = point.y;
                    double unvoteRho = (unvoteX - centerX) * cosThetaValue + (unvoteY - centerY) * sinThetaValue;
                    int unvoteRhoIndex = static_cast<int>(unvoteRho + rhoMax);
                    if (unvoteRhoIndex >= 0 && unvoteRhoIndex < rhoSize) {
                        #pragma omp atomic
                        accumulator[unvoteRhoIndex][thetaIndex]--;
                    }
                }
            }
        }
    }

    return {accumulator, segments};
}

/***********************
 *  PARALLEL - Hybrid  *
************************/

std::tuple<std::vector<std::vector<int>>, std::vector<Segment>> HT_PHT_MPI_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int thetaResolution = std::stoi(parameters["hough_theta"]);
    int numThreads = std::stoi(parameters["omp_threads"]);
    bool probabilistic = (parameters["version"] == "PHT");
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int rhoSize = static_cast<int>(2 * rhoMax) + 1;
    
    std::vector<std::vector<int>> local_accumulator(rhoSize, std::vector<int>(thetaResolution, 0));
    std::vector<std::vector<int>> accumulator;
    if (world_rank == 0) {
        accumulator.resize(rhoSize, std::vector<int>(thetaResolution, 0));
    }

    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    std::random_device rd;

    // Precompute cosine and sine values
    std::vector<double> cosTheta(thetaResolution), sinTheta(thetaResolution);
    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
        double thetaRad = thetaIndex * (M_PI / thetaResolution);
        cosTheta[thetaIndex] = cos(thetaRad);
        sinTheta[thetaIndex] = sin(thetaRad);
    }

    // Calculate rows for each process
    int rows_per_process = image.height / world_size;
    int remaining_rows = image.height % world_size;

    // Adjust start and end rows to handle remaining rows
    int start_row = world_rank * rows_per_process + std::min(world_rank, remaining_rows);
    int end_row = start_row + rows_per_process + (world_rank < remaining_rows ? 1 : 0);

    // Point processing and voting
    #pragma omp parallel num_threads(numThreads) default(none) shared(image, thetaResolution, rhoMax, rhoSize, centerX, centerY, local_accumulator, rd, cosTheta, sinTheta, numThreads, probabilistic, samplingRate, start_row, end_row)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(rd() + thread_id);
        std::uniform_int_distribution<> dis(0, 100 / samplingRate - 1);

        #pragma omp for collapse(2) schedule(static)
        for (int y = start_row; y < end_row; ++y) {
            for (int x = 0; x < image.width; ++x) {
                if ((!probabilistic && image.data[y * image.width + x] > 0) ||
                    (probabilistic && image.data[y * image.width + x] > 0 && dis(gen) == 0)) {
                    for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
                        double rho = (x - centerX) * cosTheta[thetaIndex] + (y - centerY) * sinTheta[thetaIndex];
                        int rhoIndex = static_cast<int>(rho + rhoMax);
                        if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                            #pragma omp atomic
                            local_accumulator[rhoIndex][thetaIndex]++;
                        }
                    }
                }
            }
        }
    }

    // Perform a reduction across all processes to combine results
    for (int i = 0; i < rhoSize; ++i) {
        MPI_Reduce(local_accumulator[i].data(), (world_rank == 0 ? accumulator[i].data() : nullptr), thetaResolution, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    std::vector<Segment> segments;

    if (world_rank == 0) {
        int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
        #pragma omp parallel for collapse(2) num_threads(numThreads) default(none) shared(accumulator, rhoSize, thetaResolution, voteThreshold, image, centerX, centerY, cosTheta, sinTheta, segments, rhoMax)
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
    }

    return {accumulator, segments};
}