#include "src/functions.h"

int main(int argc, char* argv[]) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    std::cout << "Starting the Hough Transform program ...\n" << std::endl;

    std::unordered_map<std::string, std::string> parameters;
    auto totStart = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, double> times;

    std::cout << "------ Loading program parameters ------\n\n";
    if (!processInputs(argc, argv, parameters)) 
        return 1;

    environmentInformation(parameters);
    createOrEmptyDirectory(parameters["output_folder"]); // Empty specified output directory

    if (parameters["run_for"] == "single_image_test"){
 
        Image img = readImage(parameters["input"]);
        Image imgCopy = readImage(parameters["input"]);
        std::cout << "Image read successfully: " << parameters["input"] << std::endl;

        printImageInfo(img);
        preprocessImage(img, parameters, true);
        
        try{

            std::vector<Segment> segments = HoughTransformation(img, parameters, true);
            std::cout << " - Time                    : " << parameters["htDuration"] << " ms" << std::endl;

            if (segments.size() > 0)
                drawLinesOnImage(segments, imgCopy, 0);

            saveImage(imgCopy, parameters["output_folder"] + parameters["image_name"] + "-" + parameters["HT_version"] + parameters["image_format"]);
        
        } catch(const std::exception& e){

            std::cerr << e.what() << '\n';
            return 1;
        }
    }
    
    else if (parameters["run_for"] == "dataset_evaluation"){

        try{

            processDataset(parameters);

            std::cout << "\nEvaluated dataset of " << parameters["dataset_size"] << " synthetic images:\n";
            std::cout << " - Avg. Preprocessing Duration : " << parameters["htDuration"] << " ms" << std::endl;
            std::cout << " - Avg. HT Duration            : " << parameters["preprocessingDuration"] << " ms" << std::endl;
            std::cout << " - Avg. Precision           : " << parameters["precision"] << "\n";
            std::cout << " - Avg. Recall              : " << parameters["recall"] << "\n";
            std::cout << " - Total Processing Time       : " << parameters["dataset_processing_time"] << " ms\n";

        } catch(const std::exception& e){

            std::cerr << e.what() << '\n';
            return 1;
        }

    }
    else{
        std::cout << "Specified parameter \'run_for\' is not valid. Allowed: single_image_test or dataset_evaluation\n\n";
    }


    if (parameters["convert_output"] == "true"){
        std::cout << "\n\nOutput images conversion to ." << parameters["conversion_format"] << " (from .pnm)" << std::endl;
        convertImages(parameters["output_folder"], parameters["conversion_format"], parameters);
    }

    auto totEnd = std::chrono::high_resolution_clock::now();
    auto totDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totEnd - totStart);

    parameters["total_duration"] = std::to_string(totDuration.count());
    savePerformance(parameters);

    std::cout << "\n\nProgram completed succesfully in " << totDuration.count() << " milliseconds" << std::endl;

    MPI_Finalize();

    return 0;
}

