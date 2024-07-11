
#ifndef HTS_EVALUATION_H
#define HTS_EVALUATION_H

#include "image_preprocessing.h"


/**
 * Evaluates the precision and recall of detected line segments against ground truth segments.
 * Precision and recall are calculated based on the overlap between the ground truth segments and the detected segments.
 * A segment is considered a true positive if it covers at least a certain percentage of a ground truth segment. This
 * percentage is fixed at 0.8 (from "Progressive Probabilistic Hough Transform"  paper by Matas and Galambos).
 *
 * @param gt_segments Ground truth line segments used as the reference for evaluation.
 * @param detected_segments Detected line segments that are evaluated against the ground truth.
 * @param parameters Parameters map to store the calculated precision and recall values.
 * @return A tuple containing the precision and recall values. 
 * 
 * Precision is the fraction of detected segments that are true positives, while recall is the fraction of ground truth segments that are correctly detected.
 */
std::tuple<double, double> evaluate(const std::vector<Segment> &gt_segments, std::vector<Segment> &detected_segments, std::unordered_map<std::string, std::string> &parameters);

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
 * Calculates the overlapping length between two collinear segments.
 * This function assumes the segments are collinear (as in "Progressive Probabilistic Hough Transform" paper by Matas and Galambos). 
 *
 * @param seg1 First segment to check for overlap.
 * @param seg2 Second segment to check for overlap.
 * @return The length of the overlap if present, otherwise returns 0.0.
 */
double computeOverlapLength(const Segment &seg1, const Segment &seg2);

/**
 * Merges similar line segments based on specified rho and theta thresholds.
 * Lines close enough in both rho and theta values are combined into a single line segment,
 * averaging their positions and summing their votes.
 * 
 * USED ONLY WITH HT AND PHT
 *
 * @param lines Vector of segments that may contain similar lines to be merged.
 * @param image Reference to the image on which the lines are detected.
 * @param rhoThreshold The threshold for difference in rho values to consider two lines as similar.
 * @param thetaThresholdDegrees The threshold for difference in theta values (in degrees) to consider two lines as similar.
 * @return A vector of merged line segments.
 */
std::vector<Segment> clustering(std::vector<Segment>& lines, const Image& image, std::unordered_map<std::string, std::string>& parameters);

/**
 * Processes the accumulator to find: detected lines, detected lines above threshold, average votes for all lines and maximum number of votes.
 *
 * @param accumulator The 2D vector representing the Hough accumulator with votes for each (rho, theta) pair.
 * @param voteTreshold The minimum number of votes to consider a detection valid.
 * @return A tuple containing the founded information about the accumulator.
 */
std::tuple<int, int, int, double> analyzeAccumulator(const std::vector<std::vector<int>> &accumulator, int voteTreshold);

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