#include "src/image_processing.h"

int main(int argc, char* argv[]) {

    auto totStart = std::chrono::high_resolution_clock::now();
    std::cout << "Starting the Hough Transform program ...\n" << std::endl;
    std::unordered_map<std::string, std::string> configuration;
    if (!processInputs(argc, argv, configuration)) {
        return 1;
    }

    std::string const input_folder = configuration["input_folder"];
    std::string const image_name = configuration["image_name"];
    std::string const image_format = configuration["image_format"];
    std::string const input_image = input_folder + image_name + image_format;
    std::string const output_folder = configuration["output_folder"] + image_name + "/";
    std::string const output_format = configuration["output_format"];
    
    createOrEmptyDirectory(output_folder); // Empty output directory.
    Image img = readImage(input_image);
    Image img_copy = readImage(input_image);
    
    if (img.data.empty()){
        std::cerr << "[ERROR] Failed to read the image." << std::endl;
        return 1;
    }

    printImageInfo(img);
    preprocessImage(img, configuration);

    std::vector<std::vector<int>> accumulator;
    try
    {
        accumulator = HoughTransformation(img, configuration);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    draw_hough_lines(accumulator, img_copy, configuration);
    saveImage(img_copy, output_folder + image_name + "-5_hough" + image_format);

    convertImages(output_folder, output_format);

    auto totEnd = std::chrono::high_resolution_clock::now();
    auto totDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totEnd - totStart);
    
    std::cout << "\n\nProgram completed succesfully in " << totDuration.count() << " milliseconds" << std::endl;

    return 0;
}
