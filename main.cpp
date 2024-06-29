#include "src/functions.h"

int main(int argc, char* argv[]) {

    // MPI SETUP AND VARIABLES DECLARATION
    int provided;
    int world_size, world_rank;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    //MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double totStart, totEnd;
    std::unordered_map<std::string, std::string> parameters;
    std::string serializedParameters; // parameter string used for communication
    int stringLength; // Length of the above string

    Image img; // Image sample
    Image imgCopy; // Copy of original used for final output presentation (only process 0)
    std::unordered_map<std::string, std::vector<Segment>> gtData;
    std::vector<Segment> gtLines;

    try{

        if (world_rank == 0){
            std::cout << "Starting the Hough Transform program" << std::endl; 
            totStart = MPI_Wtime();

            if (!processInputs(argc, argv, parameters)) 
                return 1;
            serializedParameters = serializeParameters(parameters);
            
            if (parameters["output_disabled"] == "false")
                createOrEmptyDirectory(parameters["output_folder"]); // Empty specified output directory
            
            environmentInfo(parameters); // Print PBS & openMP environment information
            img = readImage(parameters["input"]);
            imgCopy = readImage(parameters["input"]);
            std::cout << "Image read successfully: " << parameters["input"] << std::endl;
            imageInfo(img);

            gtData = loadGroundTruthData(parameters["input_folder"] + "img_synthetic_gt.csv");
            gtLines = gtData[parameters["image_name"] + ".pnm"];
            std::cout << "Loaded " << gtLines.size() << " ground truth lines for image " << parameters["image_name"] << std::endl;
            if (gtLines.size() == 0)
                std::cout << "Precision and recall metrics not available." << std::endl;
            
            printParameters(parameters);
            preprocessImage(img, parameters); // Image prepared accordingly to parameters
        }

        stringLength = serializedParameters.size();
        MPI_Bcast(&stringLength, 1, MPI_INT, 0, MPI_COMM_WORLD); // Share firstly the length of the string

        serializedParameters.resize(stringLength);
        MPI_Bcast(&serializedParameters[0], stringLength, MPI_CHAR, 0, MPI_COMM_WORLD); // Share the parameters

        if (world_rank != 0)  // Deserialize parameters on all receiving processes
            parameters = deserializeParameters(serializedParameters);

        std::vector<Segment> segments = HoughTransformation(img, parameters, gtLines);
        
        if (world_rank == 0){ // Output and performance metrics are saved accordingly to parameters by node 0.

            if (parameters["output_disabled"] == "false"){
                if (!segments.empty())
                    drawLinesOnImage(segments, imgCopy, 0);
                saveImage(imgCopy, parameters["output_folder"] + parameters["image_name"] + "-" + parameters["HT_version"] + parameters["image_format"]);
                
                if (parameters["convert_output"] == "true"){
                    std::cout << "\n\nOutput images conversion to ." << parameters["conversion_format"] << " (from .pnm)" << std::endl;
                    convertImages(parameters["output_folder"], parameters["conversion_format"], parameters);
                }
            }



            totEnd = MPI_Wtime();
            auto totDuration = totEnd - totStart;
            parameters["total_duration"] = std::to_string(totDuration);
            savePerformance(parameters);
            std::cout << "\nProgram completed succesfully in " << totDuration << " seconds\n\n\n" << std::endl;
        }

    } catch(const std::exception& e){

        std::cerr << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

