#include "utils.h"


/*********************
 * Utility functions *
**********************/

double calculateMean(const std::vector<double>& values) {
    if (values.empty()) return 0; // Handle the case where the vector is empty
 
    double sum = 0.0;
    for (double recall : values) {
        sum += recall;
    }
    double mean = sum / values.size();
    return mean;
}

double degreeToRadiant(double degreeValues){
    return degreeValues * (M_PI / 180);
}

double euclideanDistance(const Point& p1, const Point& p2) {

    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    
    return std::sqrt(dx * dx + dy * dy);
}


double midpointDistance(const Point& aStart, const Point& aEnd, const Point& bStart, const Point& bEnd) {
    
    Point aMid((aStart.x + aEnd.x) / 2, (aStart.y + aEnd.y) / 2);
    Point bMid((bStart.x + bEnd.x) / 2, (bStart.y + bEnd.y) / 2);
    return euclideanDistance(aMid, bMid);
}

double segmentLength(const Segment& segment) {
    return euclideanDistance(segment.start, segment.end);
}


std::vector<int> flatten(const std::vector<std::vector<int>>& matrix) {
    std::vector<int> flat;
    for (const auto& row : matrix) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

std::vector<std::vector<int>> reshape(const std::vector<int>& flat, int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
    return matrix;
}

/**************************
 * Presentation functions *
***************************/

void printParameters(const std::unordered_map<std::string, std::string>& parameters) {
    // Find the maximum key length
    std::cout << "------ Program Parameters ------\n";
    size_t maxKeyLength = 0;
    for (const auto& pair : parameters) {
        if (pair.first.length() > maxKeyLength) {
            maxKeyLength = pair.first.length();
        }
    }
    // Print the parameters with aligned keys
    for (const auto& pair : parameters) {
        std::cout << std::setw(maxKeyLength) << std::left << pair.first << " : " << pair.second << std::endl;
    }
    std::cout << "--------------------------------\n";

}

void imageInfo(const Image& img) {

    std::cout << "------- Image Information ------" << std::endl;
    std::cout << "Width      : " << img.width << std::endl;
    std::cout << "Height     : " << img.height << std::endl;
    std::cout << "Color Type : " << (img.isColor ? "Color (PPM)" : "Grayscale (PGM)") << std::endl;
    std::cout << "Data Size  : " << img.data.size() << "\n";
    std::cout << "--------------------------------\n";
}

void printGaussianKernel(const std::vector<std::vector<float>>& kernel) {
    for (const auto& row : kernel) {
        for (const auto& val : row) {
            std::cout << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printSegmentsInfo(const std::vector<Segment>& segments) {
    std::cout << "--Segments Info--\n";
    std::cout << "Total lines: " << segments.size() << "\n\n";

    int i = 0;
    for (const Segment &segment : segments) {
        std::cout << "Segment ID: " << i << "\n";
        std::cout << "HT (Main Segment):\n";
        std::cout << "  Start Point: (" << segment.start.x << ", " << segment.start.y << ")\n";
        std::cout << "  End Point: (" << segment.end.x << ", " << segment.end.y << ")\n";
        std::cout << "  Rho: " << segment.rho << "\n";
        std::cout << "  Theta (Radians): " << segment.thetaRad << "\n";
        std::cout << "  Theta (Degrees): " << segment.thetaDeg << "\n";
        std::cout << "  Votes: " << segment.votes << "\n\n";

        std::cout << "GT (Intersection Segment):\n";
        std::cout << "  Start Point: (" << segment.intersectionStart.x << ", " << segment.intersectionStart.y << ")\n";
        std::cout << "  End Point: (" << segment.intersectionEnd.x << ", " << segment.intersectionEnd.y << ")\n";
        std::cout << "  Rho: " << segment.interRho << "\n";
        std::cout << "  Theta (Radians): " << segment.interThetaRad << "\n";
        std::cout << "  Theta (Degrees): " << segment.interThetaDeg << "\n";
        std::cout << "-------------------------------------------------\n";
        i++;
    }
}

void houghTransformInfo(std::unordered_map<std::string, std::string>& parameters){
    std::cout   
    << "--- Hough Transformation Parameters ----" << std::endl
    << " - Version                 : " << parameters["HT_version"] << std::endl
    << " - Probabilistic           : " << ((parameters["HT_version"] == "PHT" || parameters["HT_version"] == "PPHT") ? "Yes" : "No") << std::endl
    << " - Parallel                : " << (parameters["parallel_ht"] == "true" ? "Enabled" : "Disabled") << std::endl
    << " - Parallelization Library : " << parameters["HT_parallelism"] << std::endl
    << " - Threads                 : " << parameters["omp_threads"] << std::endl
    << " - Vote threshold          : " << parameters["hough_vote_threshold"] << std::endl
    << " - Theta                   : " << parameters["hough_theta"] << std::endl
    << " - Sampling Rate           : " << parameters["sampling_rate"] << "% (only for probabilistic version)" << std::endl
    << " - Line Length             : " << parameters["ppht_line_len"] << " (only for progressive probabilistic version)" << std::endl
    << " - Line Gap                : " << parameters["ppht_line_gap"] << " (only for progressive probabilistic version)" << std::endl
    << " - Clustering              : " << (parameters["cluster_similar_lines"] == "true" ? "Yes" : "No") << std::endl
    << " - Cls. Rho Threshold      : " << parameters["cluster_rho_threshold"] << std::endl
    << " - Cls. Theta Threshold    : " << parameters["cluster_theta_threshold"] << std::endl
    << "----------------------------------------" << std::endl;

}

/***************************
 * Input/Output management *
****************************/


bool processInputs(int argc, char* argv[], std::unordered_map<std::string, std::string>& parameters) {

    if (argc != 2) {
        std::cerr << "Usage: ./HoughTransform <parameters-file>.properties" << std::endl;
        return false;
    }

    // Parse the parameters file
    std::ifstream propertiesFile(argv[1]);
    std::string line;
    size_t lastDotIndex;
    size_t lastSlashIndex;

    if (!propertiesFile.is_open()) {
        std::cerr << "Could not open the parameters file." << std::endl;
        return false;
    }

    while (getline(propertiesFile, line))
    {
        std::istringstream isLine(line);
        std::string key;
        if (std::getline(isLine, key, '=')) {
            std::string value;
            if (std::getline(isLine, value)) {
                // Remove single quotes from the value and store in the parameters map
                value.erase(std::remove(value.begin(), value.end(), '\''), value.end());
                parameters[key] = value;
            }
        }
    }

    std::string inputPath(parameters["input"]);
    if (!fileExists(inputPath)) { // Check if file exists
        std::cerr << "File does not exist.\nInput path:" << inputPath << std::endl;
        return false;
    }

    // Check if file is a .pnm file, if not update the parameters
    if (inputPath.size() < 4 || inputPath.substr(inputPath.size() - 4) != ".pnm") {
        std::cout << "[!] Input image is not a .pnm file." << std::endl;
        convertImages(inputPath, "pnm", parameters);

        // Ensure the imagePath ends with ".pnm" after conversion
        lastDotIndex = inputPath.find_last_of(".");
        if (lastDotIndex != std::string::npos) {
            inputPath = inputPath.substr(0, lastDotIndex) + ".pnm";
        } else {
            inputPath += ".pnm"; // Append .pnm if no extension is present
        }

        parameters["input"] = inputPath;
    }

    // Extract the folder path and image name
    lastSlashIndex = inputPath.find_last_of("/\\");
    lastDotIndex = inputPath.find_last_of(".");

    if (lastSlashIndex != std::string::npos) {
        parameters["input_folder"] = inputPath.substr(0, lastSlashIndex + 1);
    } else {
        parameters["input_folder"] = "./"; // Fallback to current directory if no slash is found
    }

    std::string image_name = inputPath.substr(lastSlashIndex + 1, lastDotIndex - lastSlashIndex - 1);

    parameters["image_name"] = image_name;
    parameters["image_format"] = ".pnm";

    return true;
}


Image readImage(const std::string& imagePath) {
    std::ifstream file(imagePath, std::ios::binary);
    Image img;
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + imagePath);
    }

    std::string line;
    std::getline(file, line); // Read magic line (P5 = Binary PGM, P6 = Binary PPM)

    // Checks if the format is supported
    if (line != "P5" && line != "P6") {
        throw std::runtime_error("Unsupported image format or not a PNM file: " + line);
    }

    img.isColor = (line == "P6");

    // Skip comments and empty lines
    while (std::getline(file, line)) {
        if (line[0] != '#') {
            break;
        }
    }

    std::istringstream iss(line);
    iss >> img.width >> img.height;

    // Read the maximum pixel value (usually 255)
    std::getline(file, line);
    while (line[0] == '#') {
        std::getline(file, line); // skip comments
    }
    int maxPixelValue = std::stoi(line);

    if (maxPixelValue != 255) {
        throw std::runtime_error("Unsupported max pixel value: " + std::to_string(maxPixelValue));
    }

    int dataSize = img.width * img.height * (img.isColor ? 3 : 1);
    img.data.resize(dataSize);

    file.read(reinterpret_cast<char*>(img.data.data()), dataSize);

    if (!file) {
        throw std::runtime_error("Error reading the image data.");
    }

    return img;
}

std::unordered_map<std::string, std::vector<Segment>> loadGroundTruthData(const std::string &gtPath){

    std::ifstream file(gtPath);
    std::string line;
    std::unordered_map<std::string, std::vector<Segment>> gtData;

    std::getline(file, line); // Header

    while (std::getline(file, line)) {

        std::istringstream iss(line);
        std::string part;
        std::vector<std::string> parts;

        while (std::getline(iss, part, ','))
            parts.push_back(part);
        
        if (parts.size() != 16) // ----------------------------------------------------------------------------------------- Update with number of dataset cols !
            throw std::invalid_argument("Error: Incorrect number of fields in dataset line.");
        

        std::string imageName = parts[0];
        int x1 = std::stoi(parts[1]);
        int y1 = std::stoi(parts[2]);
        int x2 = std::stoi(parts[3]);
        int y2 = std::stoi(parts[4]);
        int intersectionStartX = std::stoi(parts[9]);
        int intersectionStartY = std::stoi(parts[10]);
        int intersectionEndX = std::stoi(parts[11]);
        int intersectionEndY = std::stoi(parts[12]);
        double intersectionRho = std::stod(parts[13]);
        double rho = std::stod(parts[5]);
        double thetaRad = std::stod(parts[6]);
        double thetaDeg = std::stod(parts[7]);
        double intersectionThetaRad = std::stod(parts[14]);
        double intersectionThetaDeg = std::stod(parts[15]);

        Segment gt_seg(Point(x1, y1), Point(x2, y2), rho, thetaRad, thetaDeg, Point(intersectionStartX, intersectionStartY), Point(intersectionEndX, intersectionEndY), intersectionRho, intersectionThetaRad, intersectionThetaDeg);

        // Add the gt line segment to the corresponding image in the map
        gtData[imageName].push_back(gt_seg);
    }

    return gtData;
}

// Function to save performance data to a CSV file
void savePerformance(const std::unordered_map<std::string, std::string>& parameters) {

    // Keys to exclude from saving
    std::vector<std::string> keys_to_exclude = {
        "input", "output_folder", "performance_path", "converter_program_location", "conversion_format", "verbose", "image_format",
        "input_folder","convert_output",
    };

    std::string path = parameters.at("performance_path");
    std::string filename = path + "/performance.csv";
    bool file_exists_flag = fileExists(filename);
    std::ofstream file(filename, std::ios::app);

    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // If the file doesn't exist, write the header
    if (!file_exists_flag) {
        bool first = true;
        for (const auto& pair : parameters) {
            if (std::find(keys_to_exclude.begin(), keys_to_exclude.end(), pair.first) == keys_to_exclude.end()) {
                if (!first) {
                    file << ",";
                }
                file << pair.first;
                first = false;
            }
        }
        file << "\n";
    }

    // Write the values
    bool first = true;
    for (const auto& pair : parameters) {
        if (std::find(keys_to_exclude.begin(), keys_to_exclude.end(), pair.first) == keys_to_exclude.end()) {
            if (!first) {
                file << ",";
            }
            file << pair.second;
            first = false;
        }
    }
    file << "\n";
    file.close();
}

void saveImage(const Image& img, const std::string& outputPath) {
    std::ofstream file(outputPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Unable to open file for writing: " << outputPath << std::endl;
        return;
    }

    // Determine the format based on isColor and write the appropriate magic line.
    if (img.isColor) {
        file << "P6\n";
    } else {
        file << "P5\n";
    }

    // Write the image dimensions and the maximum pixel value. Then data.
    file << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());

    if (!file) {
        std::cerr << "[ERROR] Failed to write the image data." << std::endl;
    } else {
        std::cout << "Image saved to " << outputPath << std::endl;
    }
}

void convertImages(const std::string& path, const std::string& outputFormat, std::unordered_map<std::string, std::string>& parameters) {

    if (!pathExists(path)){
        std::cerr << "\nThe specified path does not exist." << std::endl;
        return;
    }

    std::string conversion_program_path(parameters["converter_program_location"]);

    // Command to convert images with python script
    std::string convert_command = "python3 " + conversion_program_path + " " + path + " " + outputFormat;
    std::cout << "Executing conversion command: " << convert_command << std::endl;
    int convertResult = std::system(convert_command.c_str());

    if (convertResult != 0) {
        std::cerr << "Failed to execute the conversion command successfully." << std::endl;
    }

    // Remove all .pnm files
    std::string removeCommand = "find " + path + " -type f -name '*.pnm' -exec rm {} +";
    int removeResult = std::system(removeCommand.c_str());

    if (removeResult != 0) {
        std::cerr << "Failed to remove .pnm files." << std::endl;
    }
}

void createOrEmptyDirectory(const std::string& path) {
    if (pathExists(path)) {
        // Empty the directory if it exists
        removeContents(path);
        // Remove and recreate the directory to ensure it's empty
        rmdir(path.c_str());
    }

    // Create the directory
    if (mkdir(path.c_str(), 0777) != 0) { // 0777 permissions allow read/write/execute for all users
        std::cerr << "Failed to create directory: " << path << std::endl;
    } else {
        std::cout << "Output directory ready." << std::endl;
    }
}

void removeContents(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string entryName = entry->d_name;
            if (entryName == "." || entryName == "..")
                continue;

            std::string fullPath = path + "/" + entryName;
            struct stat entry_stat;
            if (stat(fullPath.c_str(), &entry_stat) == 0) {
                if (S_ISDIR(entry_stat.st_mode)) {
                    removeContents(fullPath); // Recursively remove subdirectories
                    rmdir(fullPath.c_str());
                } else {
                    unlink(fullPath.c_str()); // Remove files
                }
            }
        }
        closedir(dir);
    }
}

bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool pathExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

