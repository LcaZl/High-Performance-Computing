
#ifndef HTS_EVALUATION_H
#define HTS_EVALUATION_H

#include "image_preprocessing.h"

/**
 * Computes the average Euclidean distance between two lines, given the four endpoints.
 *
 * @param startA Start point of segment A.
 * @param endA End point of segment A.
 * @param startB Start point of segment B.
 * @param endB End point of segment B.
 * @return The average distance between the start and end points of line segments A and B.
 */
double computeDistance(const Point &startA, const Point &endA, const Point &startB, const Point &endB);

/**
 * Computes the precision and recall of detected line segments against ground truth segments.
 * Precision: how many of the detected lines are correct.
 * Recall: how many of the ground truth lines were correctly detected.
 *
 * @param gt_segments Ground truth line segments.
 * @param detected_segments Detected line segments with chosen HT version.
 * @param maxDistance Maximum allowed distance for matching. If a line differ less than maxDistance it's a match.
 * @param version Distance type: 0 for HT-based, 1 for PPHT-based. Change the type of endpoints used.
 * With HT and PHT are used the point of intersection of the segment projected to the image boundaries. With PPHT the exact two points that define the segment.
 * @return A tuple containing the precision and recall values.
 */
std::tuple<double, double> evaluate(const std::vector<Segment> &gt_segments, const std::vector<Segment> &detectedSegments, double maxDistance, const std::string &version);

/**
 * Processes the accumulator to find: detected lines, detected lines above threshold, average votes for all lines and maximum number of votes.
 *
 * @param accumulator The 2D vector representing the Hough accumulator with votes for each (rho, theta) pair.
 * @param voteTreshold The minimum number of votes to consider a detection valid.
 * @return A tuple containing the founded information about the accumulator.
 */
std::tuple<int, int, int, double> analyzeAccumulator(const std::vector<std::vector<int>> &accumulator, int voteTreshold);

/**
 * Calculates the intersection points of a line defined by its rho and theta parameters with the image boundaries.
 *
 * @param rho The distance from the origin to the line along a vector perpendicular to the line.
 * @param theta The angle of rotation from the origin.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 */
std::tuple<Point, Point> calculateEndpoints(double rho, double theta, int width, int height);

/**
 * Merges similar line segments based on specified rho and theta thresholds.
 * Lines close enough in both rho and theta values are combined into a single line segment,
 * averaging their positions and summing their votes.
 *
 * @param lines Vector of segments that may contain similar lines to be merged.
 * @param image Reference to the image on which the lines are detected.
 * @param rhoThreshold The threshold for difference in rho values to consider two lines as similar.
 * @param thetaThresholdDegrees The threshold for difference in theta values (in degrees) to consider two lines as similar.
 * @return A vector of merged line segments.
 */
std::vector<Segment> mergeSimilarLines(std::vector<Segment>& lines, const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Draws a line from (x0, y0) to (x1, y1) using Bresenham's algorithm on RGB image data.
 * 
 * @param x0 Starting x-coordinate of the line.
 * @param y0 Starting y-coordinate of the line.
 * @param x1 Ending x-coordinate of the line.
 * @param y1 Ending y-coordinate of the line.
 * @param rgbData Reference to vector containing RGB image data.
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 */
void drawLine(int x0, int y0, int x1, int y1, std::vector<unsigned char> &rgbData, int width, int height, int color);

/**
 * Draws lines detected with PPHT on an image.
 *
 * @param segments Vector of line segments detected by PPHT.
 * @param image Image object to be drawn upon.
 */
void drawLinesOnImage(std::vector<Segment> &lines, Image &image, int color);

#endif