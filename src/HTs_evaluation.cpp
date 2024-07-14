#include "HTs_evaluation.h"

std::tuple<double, double> evaluate(const std::vector<Segment>& gt_segments, std::vector<Segment>& detected_segments, std::unordered_map<std::string, std::string>& parameters) {
    
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    double coverage_threshold = 0.8;
    std::vector<bool> detected_flags(detected_segments.size(), false); // Flags for detected segments

    for (const auto& gt_segment : gt_segments) {
        double gt_length = segmentLength(gt_segment);
        double covered_length = 0.0;

        for (size_t i = 0; i < detected_segments.size(); ++i) {
            if (detected_flags[i]) continue; // Skip already matched segments
            covered_length += computeOverlapLength(gt_segment, detected_segments[i]);
            if (covered_length / gt_length >= coverage_threshold) {
                true_positives++;
                detected_flags[i] = true; // Mark segment as matched
                break;
            }
        }

        if (covered_length / gt_length < coverage_threshold) {
            false_negatives++;
        }
    }

    for (size_t i = 0; i < detected_segments.size(); ++i) {
        if (!detected_flags[i]) {
            false_positives++;
        }
    }

    int total_detected = detected_segments.size();
    int total_ground_truth = gt_segments.size();
    double precision = total_detected > 0 ? static_cast<double>(true_positives) / total_detected : 0;
    double recall = total_ground_truth > 0 ? static_cast<double>(true_positives) / total_ground_truth : 0;

    parameters["recall"] = std::to_string(recall);
    parameters["precision"] = std::to_string(precision);

    std::cout 
        << "  |- Total detection         : " << total_detected << std::endl
        << "  |- Total GT                : " << total_ground_truth << std::endl
        << "  |- True Positive           : " << true_positives << std::endl
        << "  |- False Positive          : " << false_positives << std::endl
        << "  |- False Negative          : " << false_negatives << std::endl;

    return {precision, recall};
}

std::tuple<int, int, int, double> analyzeAccumulator(const std::vector<std::vector<int>>& accumulator, int voteThreshold) {
    int linesCount = 0;
    int maxVotes = 0;
    int totalVotes = 0;
    int linesAboveThreshold = 0;

    for (const auto &rhoRow : accumulator){
        for (int vote : rhoRow) {
            if (vote > voteThreshold)
                linesAboveThreshold++;
            linesCount++;                          // Increment the line count for each entry exceeding the threshold
            totalVotes += vote;                   // Accumulate votes exceeding the threshold
            maxVotes = std::max(maxVotes, vote); // Update the maximum votes found
        }
    }

    // Calculate the average votes for entries exceeding the threshold
    double averageVotes = linesCount > 0 ? static_cast<double>(totalVotes) / linesCount : 0.0;

    return std::make_tuple(linesCount, maxVotes, linesAboveThreshold, averageVotes);
}

std::tuple<Point, Point> calculateEndpoints(double rho, double theta, int width, int height) {
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    std::vector<Point> points;

    // Check intersections with vertical (left and right) image boundaries
    if (std::fabs(sinTheta) > std::numeric_limits<double>::epsilon()) {
        int x1 = 0;
        int y1 = static_cast<int>((rho - (x1 - width / 2) * cosTheta) / sinTheta + height / 2);
        int x2 = width - 1;
        int y2 = static_cast<int>((rho - (x2 - width / 2) * cosTheta) / sinTheta + height / 2);

        if (0 <= y1 && y1 < height) points.emplace_back(x1, y1);
        if (0 <= y2 && y2 < height) points.emplace_back(x2, y2);
    } 

    // Check intersections with horizontal (top and bottom) image boundaries
    if (std::fabs(cosTheta) > std::numeric_limits<double>::epsilon()) {
        int y1 = 0;
        int x1 = static_cast<int>((rho - (y1 - height / 2) * sinTheta) / cosTheta + width / 2);
        int y2 = height - 1;
        int x2 = static_cast<int>((rho - (y2 - height / 2) * sinTheta) / cosTheta + width / 2);

        if (0 <= x1 && x1 < width) points.emplace_back(x1, y1);
        if (0 <= x2 && x2 < width) points.emplace_back(x2, y2);
    }

    Point start = points.front();
    Point end = points.back();

    return {start, end};
}

double computeOverlapLength(const Segment& seg1, const Segment& seg2) {

    // Overlap calculation assuming collinear segments
    double x1 = std::max(std::min(seg1.start.x, seg1.end.x), std::min(seg2.start.x, seg2.end.x));
    double x2 = std::min(std::max(seg1.start.x, seg1.end.x), std::max(seg2.start.x, seg2.end.x));
    double y1 = std::max(std::min(seg1.start.y, seg1.end.y), std::min(seg2.start.y, seg2.end.y));
    double y2 = std::min(std::max(seg1.start.y, seg1.end.y), std::max(seg2.start.y, seg2.end.y));
    
    if (x1 < x2 && y1 < y2) {
        return euclideanDistance(Point(x1, y1), Point(x2, y2));
    }
    return 0.0;
}


std::vector<Segment> clustering(std::vector<Segment>& lines, const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    std::vector<Segment> mergedLines;
    double thetaThresholdDegrees = std::stod(parameters["cluster_theta_threshold"]);
    double rhoThreshold = std::stod(parameters["cluster_rho_threshold"]);
    const double thetaThresholdRadians = degreeToRadiant(thetaThresholdDegrees);
    std::string HT_version = parameters["HT_version"];

    bool linesMerged;
    do {
        linesMerged = false;
        std::vector<std::vector<Segment>> groups;

        // Group lines that are similar in terms of rho, theta and endpoints
        for (auto& line : lines) {
            bool merged = false;
            for (auto& group : groups) {
                if (HT_version == "PPHT") {
                    // Use both midpoint distance and angle for PPHT -> NOT USED.
                    double distance = midpointDistance(group[0].start, group[0].end, line.start, line.end);
                    double angleDiff = std::abs(group[0].thetaRad - line.thetaRad);
                    if (distance < rhoThreshold && angleDiff < thetaThresholdRadians) {
                        group.push_back(line);
                        merged = true;
                        break;
                    }
                } else {
                    // Use rho and theta for HT and PHT
                    if (std::abs(group[0].rho - line.rho) < rhoThreshold &&
                        std::abs(group[0].thetaRad - line.thetaRad) < thetaThresholdRadians) {
                        group.push_back(line);
                        merged = true;
                        break;
                    }
                }
            }
            if (!merged) {
                groups.push_back({line});
            }
        }

        // Step 2: For each group, calculate the average line properties and create the merged line
        mergedLines.clear();
        for (auto& group : groups) {
            if (group.size() == 1) {
                mergedLines.push_back(group[0]);
            } else {
                int totalVotes = 0;
                double sumRho = 0.0, sumThetaRad = 0.0;
                Point sumStart(0, 0), sumEnd(0, 0);

                // Summing properties weighted by votes
                for (auto& line : group) {
                    totalVotes += line.votes;
                    sumRho += line.rho * line.votes;
                    sumThetaRad += line.thetaRad * line.votes;
                    sumStart.x += line.start.x * line.votes;
                    sumStart.y += line.start.y * line.votes;
                    sumEnd.x += line.end.x * line.votes;
                    sumEnd.y += line.end.y * line.votes;
                }

                // Calculating weighted averages
                double avgRho = sumRho / totalVotes;
                double avgThetaRad = sumThetaRad / totalVotes;
                Point avgStart(sumStart.x / totalVotes, sumStart.y / totalVotes);
                Point avgEnd(sumEnd.x / totalVotes, sumEnd.y / totalVotes);

                // For HT and PHT, recalculate the endpoints -> DIfferent logic for PPHT.
                Point newStart, newEnd;
                std::tie(newStart, newEnd) = calculateEndpoints(avgRho, avgThetaRad, image.width, image.height);
                mergedLines.push_back({newStart, newEnd, avgRho, avgThetaRad, avgThetaRad * (180 / M_PI), totalVotes});
            }
        }

        // Check if any segments were merged
        if (mergedLines.size() < lines.size()) {
            linesMerged = true;
            lines = mergedLines;
        }

    } while (linesMerged);

    return mergedLines;
}

void drawLine(int x0, int y0, int x1, int y1, std::vector<unsigned char>& rgb_data, int width, int height, int color) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            int idx = (y0 * width + x0) * 3; // Calculate pixel index in the RGB array
            if (color == 0) {
                rgb_data[idx] = 255;     // Set red value
                rgb_data[idx + 1] = 0;   // Set green value
                rgb_data[idx + 2] = 0;   // Set blue value
            } else if (color == 1) {
                rgb_data[idx] = 0;       // Set red value
                rgb_data[idx + 1] = 0; // Set green value
                rgb_data[idx + 2] = 255;   // Set blue value
            }
        }
        if (x0 == x1 && y0 == y1) break; // Exit loop when end point is reached
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; } // Adjust x position and error term
        if (e2 <= dx) { err += dx; y0 += sy; } // Adjust y position and error term
    }
}

void drawLinesOnImage(std::vector<Segment> &lines, Image& image, int color) {

    for (const Segment& line : lines) {
            drawLine(line.start.x, line.start.y, line.end.x, line.end.y, image.data, image.width, image.height, color);
    }

    image.isColor = true;
}


/**
 * CLUSTERING WITH PPHT SUPPORT. NOT EFFECTIVE.
 * 
 * std::vector<Segment> clustering(std::vector<Segment>& lines, const Image& image, std::unordered_map<std::string, std::string>& parameters) {
    std::vector<Segment> mergedLines;
    double thetaThresholdDegrees = std::stod(parameters["cluster_theta_threshold"]);
    double rhoThreshold = std::stod(parameters["cluster_rho_threshold"]);
    const double thetaThresholdRadians = degreeToRadiant(thetaThresholdDegrees);
    std::string HT_version = parameters["HT_version"];

    bool linesMerged;
    do {
        linesMerged = false;
        std::vector<std::vector<Segment>> groups;

        // Step 1: Group lines that are similar in terms of rho, theta, and endpoints
        for (auto& line : lines) {
            bool merged = false;
            for (auto& group : groups) {
                if (HT_version == "PPHT") {
                    // Use both midpoint distance and angle for PPHT
                    double distance = midpointDistance(group[0].start, group[0].end, line.start, line.end);
                    double angleDiff = std::abs(group[0].thetaRad - line.thetaRad);
                    if (distance < rhoThreshold && angleDiff < thetaThresholdRadians) {
                        group.push_back(line);
                        merged = true;
                        break;
                    }
                } else {
                    // Use rho and theta for HT and PHT
                    if (std::abs(group[0].rho - line.rho) < rhoThreshold &&
                        std::abs(group[0].thetaRad - line.thetaRad) < thetaThresholdRadians) {
                        group.push_back(line);
                        merged = true;
                        break;
                    }
                }
            }
            if (!merged) {
                groups.push_back({line});
            }
        }

        // Step 2: For each group, calculate the average line properties and create the merged line
        mergedLines.clear();
        for (auto& group : groups) {
            if (group.size() == 1) {
                mergedLines.push_back(group[0]);
            } else {
                int totalVotes = 0;
                double sumRho = 0.0, sumThetaRad = 0.0;
                Point sumStart(0, 0), sumEnd(0, 0);

                // Summing properties weighted by votes
                for (auto& line : group) {
                    totalVotes += line.votes;
                    sumRho += line.rho * line.votes;
                    sumThetaRad += line.thetaRad * line.votes;
                    sumStart.x += line.start.x * line.votes;
                    sumStart.y += line.start.y * line.votes;
                    sumEnd.x += line.end.x * line.votes;
                    sumEnd.y += line.end.y * line.votes;
                }

                // Calculating weighted averages
                double avgRho = sumRho / totalVotes;
                double avgThetaRad = sumThetaRad / totalVotes;
                Point avgStart(sumStart.x / totalVotes, sumStart.y / totalVotes);
                Point avgEnd(sumEnd.x / totalVotes, sumEnd.y / totalVotes);

                if (HT_version == "PPHT") {
                    // For PPHT, use the averaged start and end points directly
                    mergedLines.push_back({avgStart, avgEnd, avgRho, avgThetaRad, avgThetaRad * (180 / M_PI), totalVotes});
                } else {
                    // For HT and PHT, recalculate the endpoints
                    Point newStart, newEnd;
                    std::tie(newStart, newEnd) = calculateEndpoints(avgRho, avgThetaRad, image.width, image.height);
                    mergedLines.push_back({newStart, newEnd, avgRho, avgThetaRad, avgThetaRad * (180 / M_PI), totalVotes});
                }
            }
        }

        // Step 3: Concatenate segments for PPHT
        if (HT_version == "PPHT") {
            std::vector<Segment> concatenatedLines;
            std::vector<bool> used(mergedLines.size(), false);

            for (size_t i = 0; i < mergedLines.size(); ++i) {
                if (used[i]) continue;

                Segment current = mergedLines[i];
                used[i] = true;

                for (size_t j = i + 1; j < mergedLines.size(); ++j) {
                    if (used[j]) continue;

                    double distance = midpointDistance(current.end, current.start, mergedLines[j].start, mergedLines[j].end);
                    double angleDiff = std::abs(current.thetaRad - mergedLines[j].thetaRad);

                    if (distance < distance_threshold && angleDiff < thetaThresholdRadians) {
                        // Concatenate the segments
                        current.end = mergedLines[j].end;
                        current.votes += mergedLines[j].votes;
                        used[j] = true;
                    }
                }

                concatenatedLines.push_back(current);
            }

            mergedLines = concatenatedLines;
        }

        // Check if any segments were merged
        if (mergedLines.size() < lines.size()) {
            linesMerged = true;
            lines = mergedLines;
        }

    } while (linesMerged);

    return mergedLines;
}
 */