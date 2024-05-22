#include "HTs_evaluation.h"

#include <vector>

double computeDistance(const Point &startA, const Point &endA, const Point &startB, const Point &endB){

    double start_distance = euclideanDistance(startA, startB);
    double end_distance = euclideanDistance(endA, endB);

    return (start_distance + end_distance) / 2.0; // Average distance between start and end points

}

std::tuple<double, double> evaluate(const std::vector<Segment>& gt_segments, const std::vector<Segment>& detectedSegments, double maxDistance, const std::string& version) {
    int true_positives = 0;
    std::vector<bool> detected_matched(detectedSegments.size(), false);

    for (const auto& gt : gt_segments) {
        for (size_t i = 0; i < detectedSegments.size(); ++i) {
            double distance;

            if (version == "PPHT")
                distance = computeDistance(gt.start, gt.end, detectedSegments[i].start, detectedSegments[i].end);
            else
                distance = computeDistance(gt.intersectionStart, gt.intersectionEnd, detectedSegments[i].start, detectedSegments[i].end);

            if (!detected_matched[i] && distance < maxDistance)
            {
                true_positives++;
                detected_matched[i] = true;
                break; // Match each gt segment to only one detected segment
                }
        }
    }

    int total_detected = detectedSegments.size();
    int total_ground_truth = gt_segments.size();
    double precision = total_detected > 0 ? static_cast<double>(true_positives) / total_detected : 0;
    double recall = total_ground_truth > 0 ? static_cast<double>(true_positives) / total_ground_truth : 0;

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
    double average_votes = linesCount > 0 ? static_cast<double>(totalVotes) / linesCount : 0.0;

    return std::make_tuple(linesCount, maxVotes, linesAboveThreshold, average_votes);
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

    // Ensure we have valid points and sort them to choose endpoints as the furthest apart
    Point start = points.front();
    Point end = points.back();

    return {start, end};
}

std::vector<Segment> mergeSimilarLines(std::vector<Segment>& lines, const Image& image, double rhoThreshold, double thetaThresholdDegrees) {
    std::vector<Segment> mergedLines;
    const double thetaThresholdRadians = degreeToRadiant(thetaThresholdDegrees);
    Point newStart, newEnd;

    for (auto &line : lines) {
        bool merged = false;
        for (auto& mergedLine : mergedLines) {
            if (std::abs(mergedLine.rho - line.rho) < rhoThreshold &&
                std::abs(mergedLine.thetaRad - line.thetaRad) < thetaThresholdRadians) {
                // Merge lines by averaging the parameters and summing the votes
                int totalVotes = mergedLine.votes + line.votes;
                double newRho = (mergedLine.rho * mergedLine.votes + line.rho * line.votes) / totalVotes;
                double newThetaRad = (mergedLine.thetaRad * mergedLine.votes + line.thetaRad * line.votes) / totalVotes;

                // Recalculate endpoints based on the new rho and theta
                std::tie(newStart, newEnd) = calculateEndpoints(newRho, newThetaRad, image.width, image.height);

                // Update merged line
                mergedLine.start = newStart;
                mergedLine.end = newEnd;
                mergedLine.rho = newRho;
                mergedLine.thetaRad = newThetaRad;
                mergedLine.thetaDeg = newThetaRad * (180 / M_PI); // Keep theta in degrees for the Segment struct
                mergedLine.votes = totalVotes;
                merged = true;
                break;
            }
        }
        if (!merged) {
            mergedLines.push_back(line);
        }
    }
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
                rgb_data[idx + 1] = 0;   // Set green value to zero
                rgb_data[idx + 2] = 0;   // Set blue value to zero
            } else if (color == 1) {
                rgb_data[idx] = 0;       // Set red value to zero
                rgb_data[idx + 1] = 0; // Set green value
                rgb_data[idx + 2] = 255;   // Set blue value to zero
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