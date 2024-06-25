
#include "functions.h"

void environmentInfo(std::unordered_map<std::string, std::string>& parameters){
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    parameters["pbs_select"] = std::getenv("PBS_SELECT");
    parameters["pbs_ncpus"] = std::getenv("PBS_NCPUS");
    parameters["pbs_mem"] = std::getenv("PBS_MEM");
    parameters["number_processes"] = std::getenv("NP_VALUE");

    std::cout << "--- Environment information ----" << std::endl;
    std::cout << "PBS Select       : " << parameters["pbs_select"] << std::endl;
    std::cout << "PBS Tot CPUs     : " << parameters["pbs_ncpus"] << std::endl;
    std::cout << "PBS Memory       : " << parameters["pbs_mem"] << std::endl;
    std::cout << "OMP Req. Threads : " << parameters["omp_threads"] << std::endl;
    std::cout << "Processes for MPI: " << world_size << std::endl;
    std::cout << "Threading for OMP: " << omp_get_max_threads() << std::endl;
    std::cout << "--------------------------------\n";
}

void preprocessImage(Image& img, std::unordered_map<std::string, std::string>& parameters) {

    bool verbose = parameters["verbose"] == "true";
    auto startTime = MPI_Wtime();

    // Paths and File Names
    std::string outputFolder = parameters["output_folder"];
    std::string outputFormat = parameters["image_format"];

    // Parallel Processing
    const bool parallel = parameters["parallel_preprocessing"] == "true";
    const int numThreads = std::stoi(parameters["omp_threads"]);
    int stepCount = 1;

    if (verbose){  
        std::cout << "\nPreprocessing image:" << std::endl;
        std::cout << " - Input: " << parameters["input"] << std::endl;
        std::cout << " - Parallel Processing: " << (parallel ? "Enabled" : "Disabled") << std::endl;
        if (parallel) std::cout << " - Threads: " << numThreads << std::endl;
    }

    auto process_step = [&](auto func, const std::string& stepDescription) {
    auto startTime = MPI_Wtime();
        func();
        auto endTime = MPI_Wtime();
        saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "-" + stepDescription + outputFormat); 
        auto duration = endTime - startTime;
        stepCount++;
        if (verbose)
            std::cout << " - Time taken for " << stepDescription << ": " << duration << " s" << std::endl;
    };

    // Process each step
    if (parameters["greyscale_conversion"] == "true") {
        if (verbose)
            std::cout << "\n[" << stepCount << "] Converting the image to grayscale..." << std::endl;

        process_step([&] { convertToGrayscale(img); }, "grayscale_conversion");
    }

    if (parameters["histogram_equalization"] == "true") {
        if (verbose)
            std::cout << "\n[" << stepCount << "] Applying histogram equalization..." << std::endl;

        process_step([&] { equalizeHistogram(img); }, "histogram_equalization");
    }

    if (parameters["gaussian_blur"] == "true") {

        int gbKernelSize = std::stoi(parameters["gb_kernel_size"]);
        float gbSigma = std::stof(parameters["gb_sigma"]);

        if (verbose){
            std::cout << "\n[" << stepCount << "] Blurring using a Gaussian filter..." << std::endl;
            std::cout << "Kernel Size: " << gbKernelSize << ", Sigma: " << gbSigma << std::endl;
        }

        if (parallel)
            process_step([&] { gaussianBlurParallel(img, gbKernelSize, gbSigma, verbose, numThreads); }, "gaussian_blur_paralle");

        else
            process_step([&] { gaussianBlur(img, gbKernelSize, gbSigma, verbose);}, "gaussian_blur");
    }

    if (parameters["sobel_edge_detection"] == "true") {

        int sedThreshold = std::stoi(parameters["sed_threshold"]);
        float sedScaleFactor = std::stof(parameters["sed_scale_factor"]);

        if (verbose){
            std::cout << "\n[" << stepCount << "] Applying Sobel edge detection..." << std::endl;
            std::cout << "Threshold: " << sedThreshold << ", Scale Factor: " << sedScaleFactor << std::endl;
        }

        if (parallel)
            process_step([&] { sobelEdgeDetectionParallel(img, sedThreshold, sedScaleFactor, numThreads); }, "sed_parallel");

        else
            process_step([&] { sobelEdgeDetection(img, sedThreshold, sedScaleFactor); }, "sed");


        if (verbose)
            std::cout << "\n[" << stepCount << "] Converting image to binary..." << std::endl;
        

        process_step([&] { toBinary(img); }, "binary_conversion");
    }

    auto endTime = MPI_Wtime();
    parameters["preprocessingDuration"] = std::to_string(endTime - startTime);
}


std::vector<Segment> HoughTransformation(Image& img, std::unordered_map<std::string, std::string>& parameters, std::vector<Segment> gtLines) {
    
    int world_rank, linesCount, averageVotes, maxVotes, linesAboveThreshold, serializedImageSize;
    std::string version = parameters["HT_version"];
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    std::vector<std::vector<int>> accumulator;
    std::vector<unsigned char> serializedImage;
    std::vector<Segment> segments;
    double precision, recall;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Hough Transform

    auto startTime = MPI_Wtime();
    if (parameters["HT_parallelism"] == "openMP" && world_rank == 0){

        if (parameters["HT_version"] == "PPHT"){
            std::tie(accumulator, segments) = PPHT_OMP(img, parameters);
        }
        else if (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT"){ 
            std::tie(accumulator, segments) = HT_PHT_OMP(img, parameters);
        }
    }
    
    else if (parameters["HT_parallelism"] == "MPI" || parameters["HT_parallelism"] == "Hybrid"){

        if (world_rank == 0){
            serializedImage = img.serialize();
            serializedImageSize = serializedImage.size();
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&serializedImageSize, 1, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast the size of the serialized image
        serializedImage.resize(serializedImageSize); // Resize the buffer on all processes to hold the serialized image data
        MPI_Bcast(serializedImage.data(), serializedImageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD); // Broadcast the serialized image data

        if (world_rank != 0) 
            img = Image::deserialize(serializedImage);

        std::cout << "[PROCESS "<< world_rank << "] Started HT." << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);

        if (parameters["HT_version"] == "PPHT")
                std::tie(accumulator, segments) = PPHT_MPI(img, parameters);
        else
            std::tie(accumulator, segments) = HT_PHT_MPI(img, parameters);
            
        //houghTransformParallel_Hybrid(img, parameters);
    }
    else if (parameters["HT_parallelism"] == "None" && world_rank == 0){

        if (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT")
            std::tie(accumulator, segments) = HT_PHT(img, parameters);
        else
            std::tie(accumulator, segments) = PPHT(img, parameters);
    }

    
    if (world_rank == 0){
        auto endTime = MPI_Wtime();

        // Cluster lines if requested
        if ( parameters["cluster_similar_lines"] == "true" && world_rank == 0) 
            segments = mergeSimilarLines(segments, img, parameters);
        
        parameters["htDuration"] = std::to_string(endTime - startTime);

        // Analyze accumulator
        std::tie(linesCount, maxVotes, linesAboveThreshold, averageVotes) = analyzeAccumulator(accumulator, voteThreshold);
        std::cout << std::endl 
        << "HT for "<< parameters["image_name"] << std::endl
        << "  |- # Lines analyzed        : " << linesCount << std::endl
        << "  |- Max. Votes for a line   : " << maxVotes << std::endl
        << "  |- Avg. Votes for a line   : " << averageVotes << std::endl
        << "  |- # Lines above threshold : " << linesAboveThreshold << std::endl
        << "  |- # Final Segments        : " << segments.size() << std::endl
        << "  |- Time                    : " << parameters["htDuration"] << " s" << std::endl;

        if (gtLines.size() > 0)
            std::tie(precision, recall) = evaluate(gtLines, segments, parameters);

    }

    return segments;
}

