#include "src/functions.h"

int main(int argc, char* argv[]) {

    // MPI SETUP AND VARIABLES DECLARATION
    int provided;
    int world_size, world_rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double totStart, totEnd;
    std::unordered_map<std::string, std::string> parameters;
    std::unordered_map<std::string, double> times;
    std::string serializedParameters; // parameter string used for communication
    int stringLength; // Length of the above string
    Image img; // Image sample
    Image imgCopy; // Copy of original used for final output presentation (only process 0)

    if (world_rank == 0)
        std::cout << "Starting the Hough Transform program" << std::endl; 
    sleep(1); std::cout << "[PROCESS "<< world_rank << "] Started." << std::endl; sleep(1);

    try{

        // LOADING PROGRAM PARAMETERS AND SHARE THEM AMONG PROCESSES
        if (world_rank == 0){
            totStart = MPI_Wtime();

            std::cout << "------ Program Parameters ------\n";
            if (!processInputs(argc, argv, parameters)) 
                return 1;
            std::cout << "--------------------------------\n";

            createOrEmptyDirectory(parameters["output_folder"]); // Empty specified output directory
            environmentInfo(parameters); // Print PBS & openMP environment information

            // Serialize parameters on root process in order to share them
            serializedParameters = serializeParameters(parameters);
        }

        stringLength = serializedParameters.size();
        MPI_Bcast(&stringLength, 1, MPI_INT, 0, MPI_COMM_WORLD); // Share firstly the length of the string

        serializedParameters.resize(stringLength);
        MPI_Bcast(&serializedParameters[0], stringLength, MPI_CHAR, 0, MPI_COMM_WORLD); // Share the parameters

        if (world_rank != 0)  // Deserialize parameters on all receiving processes
            parameters = deserializeParameters(serializedParameters);

        MPI_Barrier(MPI_COMM_WORLD);

        if (parameters["run_for"] == "single_image_test") {
            // SINGLE IMAGE PREPROCESSING & HOUGH TRANSFORM
            std::vector<unsigned char> serializedImage;
            Image img, imgCopy;
            int serializedImageSize;

            if (world_rank == 0) {
                img = readImage(parameters["input"]);
                imgCopy = readImage(parameters["input"]);
                std::cout << "Image read successfully: " << parameters["input"] << std::endl;
                imageInfo(img);
                preprocessImage(img, parameters, parameters["verbose"] == "true"); // Image prepared accordingly to parameters

                serializedImage = img.serialize();
                serializedImageSize = serializedImage.size();
                std::cout << "Serialized image size on root: " << serializedImageSize << std::endl;

            }

            // Broadcast the size of the serialized image
            MPI_Bcast(&serializedImageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Resize the buffer on all processes to hold the serialized image data
            serializedImage.resize(serializedImageSize);

            // Broadcast the serialized image data
            MPI_Bcast(serializedImage.data(), serializedImageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

            if (world_rank != 0) 
                img = Image::deserialize(serializedImage);

            MPI_Barrier(MPI_COMM_WORLD);
            std::vector<Segment> segments = HoughTransformation(img, parameters, parameters["verbose"] == "true");
            std::cout << " - Time                    : " << parameters["htDuration"] << " s" << std::endl;

            if (world_rank == 0) {
                if (!segments.empty())
                    drawLinesOnImage(segments, imgCopy, 0);
                saveImage(imgCopy, parameters["output_folder"] + parameters["image_name"] + "-" + parameters["HT_version"] + parameters["image_format"]);
            }
        }
        
        else if (parameters["run_for"] == "dataset_evaluation"){

                // DATASET OF SYNTHETIC IMAGES PREPROCESSING & HOUGH TRANSFORM

                processDataset(parameters);

                if (world_rank == 0){
                    std::cout << "\nEvaluated dataset of " << parameters["dataset_size"] << " synthetic images:\n";
                    std::cout << " - Avg. Preprocessing Duration : " << parameters["htDuration"] << " ms" << std::endl;
                    std::cout << " - Avg. HT Duration            : " << parameters["preprocessingDuration"] << " ms" << std::endl;
                    std::cout << " - Avg. Precision           : " << parameters["precision"] << "\n";
                    std::cout << " - Avg. Recall              : " << parameters["recall"] << "\n";
                    std::cout << " - Total Processing Time       : " << parameters["dataset_processing_time"] << " ms\n";
                }
        }
        else{
            std::cout << "Specified parameter \'run_for\' is not valid. Allowed: single_image_test or dataset_evaluation\n\n";
        }

        if (world_rank == 0){ // Output is converted accordingly to parameters and performance saved by node 0.

            if (parameters["convert_output"] == "true"){
                std::cout << "\n\nOutput images conversion to ." << parameters["conversion_format"] << " (from .pnm)" << std::endl;
                convertImages(parameters["output_folder"], parameters["conversion_format"], parameters);
            }

            totEnd = MPI_Wtime();
            auto totDuration = totEnd - totStart;
            parameters["total_duration"] = std::to_string(totDuration);
            savePerformance(parameters);
            std::cout << "\nProgram completed succesfully in " << totDuration << " seconds" << std::endl;
        }

    } catch(const std::exception& e){

        std::cerr << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

