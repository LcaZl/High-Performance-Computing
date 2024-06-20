#include "HTs.h"

/**************
 *   SERIAL   *
***************/

std::vector<std::vector<int>> houghTransform(const Image& image, std::unordered_map<std::string, std::string>& parameters) {

    bool probabilistic = (parameters["version"] == "PHT" || parameters["version"] == "PPHT");
    double thetaResolution = std::stod(parameters["hough_theta"]);
    int samplingRate = std::stoi(parameters["sampling_rate"]);

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

std::vector<Segment> linesExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    std::vector<Segment> lines;
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    double thetaResolution = std::stod(parameters["hough_theta"]);

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

std::vector<Segment> linesProgressiveExtraction(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    std::vector<Segment> segments;
    double centerX = image.width / 2.0;
    double centerY = image.height / 2.0;
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    double thetaResolution = std::stod(parameters["hough_theta"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);

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

/********************
 *  PARALLEL - MPI  *
*********************/

std::vector<std::vector<int>> houghTransformParallel_MPI(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    double thetaResolution = std::stod(parameters["hough_theta"]);
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

    // Broadcast the global accumulator to all processes
    if (world_rank == 0) {
        for (int i = 0; i < rhoSize; ++i) {
            MPI_Bcast(accumulator[i].data(), thetaResolution, MPI_INT, 0, MPI_COMM_WORLD);
        }
    } else {
        accumulator.resize(rhoSize, std::vector<int>(thetaResolution, 0));
        for (int i = 0; i < rhoSize; ++i) {
            MPI_Bcast(accumulator[i].data(), thetaResolution, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    return accumulator;
}

std::vector<Segment> linesExtractionParallel_MPI(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    double thetaResolution = std::stod(parameters["hough_theta"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);

    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    std::vector<Segment> local_lines;
    std::vector<Segment> lines;

    // Calculate rows for each process
    int rows_per_process = accumulator.size() / world_size;
    int remaining_rows = accumulator.size() % world_size;

    // Adjust start and end rows to handle remaining rows
    int start_row = world_rank * rows_per_process + std::min(world_rank, remaining_rows);
    int end_row = start_row + rows_per_process + (world_rank < remaining_rows ? 1 : 0);

    for (int rhoIndex = start_row; rhoIndex < end_row; ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;

                Point start, end;
                std::tie(start, end) = calculateEndpoints(rho, thetaRad, image.width, image.height);
                local_lines.push_back(Segment{start, end, rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]});
            }
        }
    }

    int local_size = local_lines.size();
    std::vector<int> all_sizes(world_size);
    MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(world_size, 0);
    int total_size = 0;
    if (world_rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + all_sizes[i - 1];
        }
        total_size = displs[world_size - 1] + all_sizes[world_size - 1];
    }

    MPI_Datatype segmentType;
    createSegmentMPIType(&segmentType);

    if (world_rank == 0) {
        lines.resize(total_size);
    }

    MPI_Gatherv(local_lines.data(), local_size, segmentType,
                lines.data(), all_sizes.data(), displs.data(), segmentType,
                0, MPI_COMM_WORLD);

    MPI_Type_free(&segmentType);

    return lines;
}


std::vector<Segment> linesProgressiveExtractionParallel_MPI(const std::vector<std::vector<int>>& accumulator, const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double rhoMax = std::sqrt(image.width * image.width + image.height * image.height);
    std::vector<Segment> local_segments;
    std::vector<Segment> segments;
    double thetaResolution = std::stod(parameters["hough_theta"]);
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    int lineGap = std::stoi(parameters["ppht_line_gap"]);
    int lineLength = std::stoi(parameters["ppht_line_len"]);
    // Calculate rows for each process
    int rows_per_process = accumulator.size() / world_size;
    int remaining_rows = accumulator.size() % world_size;

    // Adjust start and end rows to handle remaining rows
    int start_row = world_rank * rows_per_process + std::min(world_rank, remaining_rows);
    int end_row = start_row + rows_per_process + (world_rank < remaining_rows ? 1 : 0);

    for (int rhoIndex = start_row; rhoIndex < end_row; ++rhoIndex) {
        for (int thetaIndex = 0; thetaIndex < thetaResolution; ++thetaIndex) {
            if (accumulator[rhoIndex][thetaIndex] > voteThreshold) {
                double thetaRad = thetaIndex * (M_PI / thetaResolution);
                double thetaDeg = thetaRad * (180.0 / M_PI);
                double rho = rhoIndex - rhoMax;
                double sinTheta = sin(thetaRad);
                double cosTheta = cos(thetaRad);
                std::vector<Point> linePoints;

                for (int x = 0; x < image.width; ++x) {
                    for (int y = 0; y < image.height; ++y) {
                        if (image.data[y * image.width + x] > 0) {
                            int calculatedRho = static_cast<int>((x - image.width / 2) * cosTheta + (y - image.height / 2) * sinTheta);
                            if (std::abs(calculatedRho - rho) < 2) {
                                if (!linePoints.empty() && (std::abs(linePoints.back().x - x) > lineGap || std::abs(linePoints.back().y - y) > lineGap)) {
                                    if (linePoints.size() >= static_cast<size_t>(lineLength)) {
                                        local_segments.emplace_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]});
                                    }
                                    linePoints.clear();
                                }
                                linePoints.push_back(Point{x, y});
                            }
                        }
                    }
                }

                if (linePoints.size() >= static_cast<size_t>(lineLength)) {
                    local_segments.emplace_back(Segment{linePoints.front(), linePoints.back(), rho, thetaRad, thetaDeg, accumulator[rhoIndex][thetaIndex]});
                }
            }
        }
    }

    int local_size = local_segments.size();
    std::vector<int> all_sizes(world_size);
    MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(world_size, 0);
    int total_size = 0;
    if (world_rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + all_sizes[i - 1];
        }
        total_size = displs[world_size - 1] + all_sizes[world_size - 1];
    }

    MPI_Datatype segmentType;
    createSegmentMPIType(&segmentType);

    if (world_rank == 0) {
        segments.resize(total_size);
    }

    MPI_Gatherv(local_segments.data(), local_size, segmentType,
                segments.data(), all_sizes.data(), displs.data(), segmentType,
                0, MPI_COMM_WORLD);

    MPI_Type_free(&segmentType);

    return segments;
}

/********************
 *  PARALLEL - OMP  *
*********************/

std::vector<std::vector<int>> houghTransformParallel_OMP(const Image& image, std::unordered_map<std::string, std::string>& parameters) {
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

    return accumulator;
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


