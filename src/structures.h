#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <sys/stat.h> // For stat()
#include <cstdlib> // For std::system
#include <iomanip> // Include for std::setprecision
#include <chrono> // Include for std::chrono
#include <dirent.h> // for I/O
#include <unistd.h> //for I/O
#include <random>
#include <cstdlib>
#include <tuple>
#include <mpi.h>
#include <cstring>

#ifdef _OPENMP
    #include <omp.h>
#endif

/************************
 *  Internal Structure  *
*************************/

/**
 * Used to store an image and relative information. 
 * Has two function for serialization and deserialization, used only for the MPI communication when
 * image must be shared among processes.
*/

struct Image {
    std::vector<unsigned char> data; 
    int width, height;
    bool isColor; // true per PPM, false per PGM

    Image() : width(0), height(0), isColor(false) {}

    Image(const std::vector<unsigned char>& data, int width, int height, bool isColor)
        : data(data), width(width), height(height), isColor(isColor) {}


    // Function to serialize the Image structure
    std::vector<unsigned char> serialize() const {
        std::vector<unsigned char> serializedData;
        serializedData.resize(data.size() + sizeof(width) + sizeof(height) + sizeof(isColor));
        
        unsigned char* ptr = serializedData.data();
        std::memcpy(ptr, &width, sizeof(width));
        ptr += sizeof(width);
        std::memcpy(ptr, &height, sizeof(height));
        ptr += sizeof(height);
        std::memcpy(ptr, &isColor, sizeof(isColor));
        ptr += sizeof(isColor);
        std::memcpy(ptr, data.data(), data.size());

        return serializedData;
    }

    // Function to deserialize the Image structure
    static Image deserialize(const std::vector<unsigned char>& serializedData) {
        Image img;
        const unsigned char* ptr = serializedData.data();
        
        std::memcpy(&img.width, ptr, sizeof(img.width));
        ptr += sizeof(img.width);
        std::memcpy(&img.height, ptr, sizeof(img.height));
        ptr += sizeof(img.height);
        std::memcpy(&img.isColor, ptr, sizeof(img.isColor));
        ptr += sizeof(img.isColor);

        img.data.resize(serializedData.size() - (ptr - serializedData.data()));
        std::memcpy(img.data.data(), ptr, img.data.size());

        return img;
    }
};

/**
 * Point structure
 */
struct Point {
    int x, y;
    Point(int px, int py) : x(px), y(py) {}
    Point() : x(0), y(0) {}
};

/**
 * Used to store segments information of different kind.
 * 
 * 1 - Used for detected segments/lines with different HT versions (first constructor).
 * 2 - Used for ground truth data (<inter>.. attributes and second constructor).
*/
struct Segment {
    // Attributes used only by HTs.
    Point start;
    Point end;
    double rho;        // Distance from the origin to the line
    double thetaRad;   // Angle in radians
    double thetaDeg;   // Angle in degrees (-180, 180)
    int votes; // Number of votes from the accumulator

    // Segment augmented to reach the image border, keeping the slope.
    // Attributes used only for ground truth data
    Point intersectionStart; // Start point projection on image border
    Point intersectionEnd; // End point projection on image border
    double interRho; // Rho of the longer segment
    double interThetaRad; // Intersection line's theta in radians
    double interThetaDeg; // Intersection line's theta in degree

    Segment(Point s, Point e, double r, double tr, double td, int v) :
        start(s), end(e), rho(r), thetaRad(tr), thetaDeg(td), votes(v), intersectionStart(Point(0,0)), intersectionEnd(Point(0,0)), interRho(0), interThetaRad(0), interThetaDeg(0){}

    Segment(Point s, Point e, double r, double tr, double td, Point i1, Point i2, double interR, double interTr, double interTd) :
        start(s), end(e), rho(r), thetaRad(tr), thetaDeg(td), votes(0), intersectionStart(i1), intersectionEnd(i2), interRho(interR), interThetaRad(interTr), interThetaDeg(interTd){}

    Segment() : start(Point(0, 0)), end(Point(0, 0)), rho(0), thetaRad(0),
                thetaDeg(0), votes(0), intersectionStart(Point(0, 0)),
                intersectionEnd(Point(0, 0)), interRho(0), interThetaRad(0),
                interThetaDeg(0) {}
};

/**
 * Creates a custom MPI data type for the Segment structure to facilitate MPI communication.
 *
 * @param segmentType Pointer to the MPI_Datatype to be created.
 */
void createSegmentMPIType(MPI_Datatype* segmentType);

/**
 * Deserializes a string into a map of parameters.
 * 
 * @param serializedParameters The string containing the serialized parameters.
 * @return A map with the deserialized parameters.
 */
std::unordered_map<std::string, std::string> deserializeParameters(const std::string& serializedParameters);

/**
 * Serializes a map of parameters into a string.
 * 
 * @param parameters The map containing the parameters to serialize.
 * @return A string with the serialized parameters.
 */
std::string serializeParameters(const std::unordered_map<std::string, std::string>& parameters);

#endif
