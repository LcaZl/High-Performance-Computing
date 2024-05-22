#include "utils.h"

/*********************
 * Utility functions *
**********************/

double calculateMean(const std::vector<double>& recalls) {
    if (recalls.empty()) return 0; // Handle the case where the vector is empty
 
    double sum = 0.0;
    for (double recall : recalls) {
        sum += recall;
    }
    double mean = sum / recalls.size();
    return mean;
}

double degreeToRadiant(double degree_values){
    return degree_values * (M_PI / 180);
}

double euclideanDistance(const Point& p1, const Point& p2) {

    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    
    return std::sqrt(dx * dx + dy * dy);
}

/**************************
 * Presentation functions *
***************************/

void printImageInfo(const Image& img) {
    std::cout << "\nImage Information:" << std::endl;
    std::cout << " - Width: " << img.width << std::endl;
    std::cout << " - Height: " << img.height << std::endl;
    std::cout << " - Color Type: " << (img.isColor ? "Color (PPM)" : "Grayscale (PGM)") << std::endl;
    std::cout << " - Data Size: " << img.data.size() << "\n";
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
                std::cout << key + ": " + value << std::endl;
            }
        }
    }

    std::cout << std::endl;

    std::string inputPath(parameters["input"]);
    if (!fileExists(inputPath)) { // Check if file exists
        std::cerr << "File does not exist.\nInput path:" << inputPath << std::endl;
        return false;
    }
    // Check if file is a .pnm file
    if (parameters["run_for"] == "single_image_test"){
        
        if (inputPath.size() < 4 || inputPath.substr(inputPath.size() - 4) != ".pnm") {
            std::cerr << "[!] Input image is not a .pnm file.\n";
            convertImages(inputPath, "pnm", parameters);

            // Ensure the imagePath ends with ".pnm" after conversion
            size_t lastDotIndex = inputPath.find_last_of(".");
            if (lastDotIndex != std::string::npos) {
                inputPath = inputPath.substr(0, lastDotIndex) + ".pnm";
            } else {
                inputPath += ".pnm"; // Append .pnm if no extension is present
            }

            parameters["input"] = inputPath;
        }

        size_t lastSlashIndex = inputPath.find_last_of("/\\"); // Handle both Unix and Windows paths
        size_t lastDotIndex = inputPath.find_last_of(".");
        std::string image_name = inputPath.substr(lastSlashIndex + 1, lastDotIndex - lastSlashIndex - 1);

        parameters["image_name"] = image_name;
    }

    

    parameters["image_format"] = ".pnm";

    return true;
}

Image readImage(const std::string& imagePath) {
    std::ifstream file(imagePath, std::ios::binary);
    Image img;
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << imagePath << std::endl;
        return img; // Returns an empty Image object
    }

    std::string line;
    std::getline(file, line); // Read magic line (P5 = Binary PGM, P6 = Binary PPM)

    // Checks if the format is supported
    if (line != "P5" && line != "P6") {
        std::cerr << "Unsupported image format or not a PNM file: " << line << std::endl;
        return img;
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
    int maxPixelValue = 0;
    std::getline(file, line);
    while (line[0] == '#') {
        std::getline(file, line); // skip comments
    }
    maxPixelValue = std::stoi(line);

    if (maxPixelValue != 255) {
        std::cerr << "Unsupported max pixel value: " << maxPixelValue << std::endl;
        return img;
    }

    int dataSize = img.width * img.height * (img.isColor ? 3 : 1);
    img.data.resize(dataSize);

    file.read(reinterpret_cast<char*>(img.data.data()), dataSize);

    if (!file) {
        std::cerr << "Error reading the image data." << std::endl;
        return img;
    }

    return img;
}

// Function to save performance data to a CSV file
void savePerformance(const std::unordered_map<std::string, std::string>& parameters) {
    // Keys to exclude from saving
    std::vector<std::string> keys_to_exclude = {
        "input", "output_folder", "performance_path", "converter_program_location", "conversion_format", "verbose", "image_format"
    };

    // Extract the path from the parameters
    std::string path = parameters.at("performance_path");

    // Determine the filename based on the "run_for" key in the parameters
    std::string filename;
    if (parameters.at("run_for") == "single_image_test") {
        filename = path + "/single_image_performances.csv";
    } else if (parameters.at("run_for") == "dataset_evaluation") {
        filename = path + "/dataset_eval_performances.csv";
    } else {
        std::cerr << "Unknown run_for value: " << parameters.at("run_for") << std::endl;
        return;
    }

    // Check if the file exists
    bool file_exists_flag = fileExists(filename);

    // Open the file in append mode
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

    // Close the file
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

    // Write the image dimensions and the maximum pixel value. 
    file << img.width << " " << img.height << "\n255\n";

    // Write the image data.
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());

    if (!file) {
        std::cerr << "[ERROR] Failed to write the image data." << std::endl;
    } else {
        std::cout << " - Image saved to " << outputPath << std::endl;
    }
}

std::vector<ImagePart> splitImage(const Image& img, int parts, int overlap) {
    std::vector<ImagePart> imageParts;
    int basePartHeight = img.height / parts; // Calculate base height for each part
    int excess = img.height % parts; // Calculate excess rows to distribute among parts evenly

    for (int part = 0; part < parts; part++) {
        // Calculate starting row for the current part
        int startRow = part * basePartHeight + std::min(part, excess);
        // Calculate height of the current part, adding an extra row if this part includes excess
        int height = basePartHeight + (part < excess ? 1 : 0);
        // Calculate the actual start and end rows, considering the overlap
        int actualStartRow = std::max(0, startRow - overlap);
        int actualEndRow = std::min(img.height, startRow + height + overlap);
        // Calculate overlap at the top and bottom of the current part
        int overlapTop = startRow - actualStartRow;
        int overlapBottom = actualEndRow - (startRow + height);

        // Initialize the current ImagePart
        ImagePart imgPart;
        imgPart.startRow = startRow;
        imgPart.width = img.width;
        imgPart.height = actualEndRow - actualStartRow;
        imgPart.overlapTop = overlapTop;
        imgPart.overlapBottom = overlapBottom;
        imgPart.data.resize(imgPart.width * imgPart.height);

        // Copy the relevant part of the image data
        for (int row = 0; row < imgPart.height; row++) {
            std::copy_n(img.data.begin() + ((actualStartRow + row) * img.width),
                        img.width,
                        imgPart.data.begin() + (row * imgPart.width));
        }

        imageParts.push_back(imgPart);
    }

    return imageParts;
}

Image recombineImage(const std::vector<ImagePart>& parts) {
    if (parts.empty()) return Image(); // Handle empty input

    // Calculate the total width and height of the recombined image
    int width = parts.front().width;
    int height = std::accumulate(parts.begin(), parts.end(), 0,
                                 [&](int acc, const ImagePart& part) {
                                     return acc + part.height - part.overlapTop - part.overlapBottom;
                                 });

    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);

    int currentRow = 0;
    for (const auto& part : parts) {
        // Copy each part's data, skipping the overlapped rows
        for (int row = part.overlapTop; row < part.height - part.overlapBottom; row++) {
            std::copy_n(part.data.begin() + (row * part.width),
                        part.width,
                        img.data.begin() + ((currentRow++) * width));
        }
    }

    return img;
}

void convertImages(const std::string& path, const std::string& outputFormat, std::unordered_map<std::string, std::string>& parameters) {

    if (!pathExists(path)){
        std::cerr << "\nThe specified path does not exist." << std::endl;
        return;
    }

    std::string conversion_program_path(parameters["converter_program_location"]);
    // Command to convert images with python script
    std::string convert_command = "python3 " + conversion_program_path + " " + path + " " + outputFormat;
    std::cout << "\nExecuting conversion command: " << convert_command << std::endl;
    int convertResult = std::system(convert_command.c_str());

    if (convertResult != 0) {
        std::cerr << "Failed to execute the conversion command successfully." << std::endl;
    }

    // Command to remove all .pnm files
    std::string removeCommand = "find " + path + " -type f -name '*.pnm' -exec rm {} +";
    int removeResult = std::system(removeCommand.c_str());

    if (removeResult != 0) {
        std::cerr << "Failed to remove .pnm files." << std::endl;
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



