#include "utils.h"

/***************************
 * Input/Output management *
****************************/

/**
 * Processes command line inputs for configuring the execution mode and paths related to image processing.
 * Validates the number of arguments, execution mode, existence, and format of the image file, 
 * and parses the configuration file to populate an output configuration map.
 * 
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @param outConfig Map to store configuration settings extracted from the command line and the configuration file.
 * @return Returns true if inputs are processed successfully, false otherwise.
 */
bool processInputs(int argc, char* argv[], std::unordered_map<std::string, std::string>& outConfig) {

    if (argc != 4) {
        std::cerr << "Usage: ./HoughTransform <image-name>.pnm <config-file>.properties [parallel,serial]" << std::endl;
        return false;
    }

    std::string mode(argv[3]);
    if (mode != "parallel" && mode != "serial") {
        std::cerr << "Execution mode must be either 'parallel' or 'serial'." << std::endl;
        return false;
    }
    // Set execution mode in the output configuration
    outConfig["parallel"] = (mode == "parallel") ? "true" : "false";

    std::string imagePath(argv[1]);
    if (!fileExists(imagePath)) { // Check if file exists
        std::cerr << "File does not exist." << std::endl;
        return false;
    }
    // Check if file is a .pnm file
    if (imagePath.size() < 4 || imagePath.substr(imagePath.size() - 4) != ".pnm") {
        std::cerr << "File is not a .pnm file." << std::endl;
        return false;
    }

    // Extract and store image path information into the configuration map
    size_t lastSlashPos = imagePath.find_last_of("/\\");
    std::string folderPath = (lastSlashPos != std::string::npos) ? imagePath.substr(0, lastSlashPos + 1) : "";
    std::string fileName = (lastSlashPos != std::string::npos) ? imagePath.substr(lastSlashPos + 1) : imagePath;
    size_t lastDotPos = fileName.find_last_of('.');
    std::string fileStem = (lastDotPos != std::string::npos) ? fileName.substr(0, lastDotPos) : fileName;

    outConfig["input_folder"] = folderPath;
    outConfig["image_name"] = fileStem;
    outConfig["image_format"] = ".pnm";

    // Parse the configuration file
    std::ifstream configFile(argv[2]);
    std::string line;
    if (!configFile.is_open()) {
        std::cerr << "Could not open the configuration file." << std::endl;
        return false;
    }

    while (getline(configFile, line)) {
        std::istringstream is_line(line);
        std::string key;
        if (std::getline(is_line, key, '=')) {
            std::string value;
            if (std::getline(is_line, value)) {
                // Remove single quotes from the value and store in the configuration map
                value.erase(std::remove(value.begin(), value.end(), '\''), value.end());
                outConfig[key] = value;
            }
        }
    }

    return true;
}


/**
 * Reads an image from a file in PNM format (P5 for grayscale, P6 for color).
 * 
 * @param imagePath Path to the image file.
 * @return Image object containing the image data. Returns an empty Image object on failure.
 */
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
    int maxval;
    iss >> img.width >> img.height;
    file >> maxval; // Read the maximum value per pixel
    file.ignore(); // Ignore the newline after the maximum value

    if (maxval > 255) {
        std::cerr << "Unsupported max value: " << maxval << std::endl;
        return img;
    }

    int dataSize = img.width * img.height * (img.isColor ? 3 : 1);
    img.data.resize(dataSize);

    file.read(reinterpret_cast<char*>(img.data.data()), dataSize);

    if (!file) {
        std::cerr << "Error reading the image data." << std::endl;
        return img;
    }

    std::cout << "Image read successfully: " << imagePath << std::endl;
    return img;
}

/**
 * Saves an image to a file in PNM format (P5 for grayscale, P6 for color).
 * 
 * @param img Image object containing the image data to be saved.
 * @param outputPath Path where the image file should be saved.
 */
void saveImage(const Image& img, const std::string& outputPath) {
    std::ofstream file(outputPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Unable to open file for writing: " << outputPath << std::endl;
        return;
    }

    /* Determine the format based on isColor and write the appropriate magic line. */
    if (img.isColor) {
        file << "P6\n";
    } else {
        file << "P5\n";
    }

    /* Write the image dimensions and the maximum pixel value. */
    file << img.width << " " << img.height << "\n255\n";

    /* Write the image data. */
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());

    if (!file) {
        std::cerr << "[ERROR] Failed to write the image data." << std::endl;
    } else {
        std::cout << "-> Image saved to " << outputPath << std::endl;
    }
}

/**
 * Converts all images in the given directory, or a single image, to the specified output format.
 * 
 * @param path Path to the image or directory containing images.
 * @param outputFormat Desired output image format.
 */
void convertImages(const std::string& path, const std::string& outputFormat) {
    std::cout << "\n[7] Converting images at output folder in " + outputFormat + " format ..." << std::endl;

    if (!pathExists(path)) {
        std::cerr << "The specified path does not exist." << std::endl;
        return;
    }

    std::string command = "python3 src/image_converter.py " + path + " " + outputFormat;
    std::cout << "Executing command: " << command << std::endl;
    int result = std::system(command.c_str());

    if (result != 0) {
        std::cerr << "Failed to execute the command successfully." << std::endl;
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

/**
 * Creates a new directory at the specified path, or empties it if it already exists.
 * 
 * @param path Path of the directory to create or empty.
 */

void createOrEmptyDirectory(const std::string& path) {
    if (pathExists(path)) {
        // Empty the directory if it exists
        removeContents(path);
        // Optional: Remove and recreate the directory to ensure it's empty
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
            struct stat entryStat;
            if (stat(fullPath.c_str(), &entryStat) == 0) {
                if (S_ISDIR(entryStat.st_mode)) {
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

/**************************
 * Presentation functions *
***************************/

/**
 * Prints information about the given Image object to the console.
 * 
 * @param img The Image object to print information for.
 */
void printImageInfo(const Image& img) {
    std::cout << "\nImage Information:" << std::endl;
    std::cout << " - Width: " << img.width << std::endl;
    std::cout << " - Height: " << img.height << std::endl;
    std::cout << " - Color Type: " << (img.isColor ? "Color (PPM)" : "Grayscale (PGM)") << std::endl;
    std::cout << " - Data Size: " << img.data.size() << std::endl;
}
/**
 * Prints the Gaussian kernel to the console.
 * 
 * @param kernel The Gaussian kernel represented as a 2D vector of floats.
 */
void printGaussianKernel(const std::vector<std::vector<float>>& kernel) {
    for (const auto& row : kernel) {
        for (const auto& val : row) {
            std::cout << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}



