
#include "functions.h"

void environmentInfo(std::unordered_map<std::string, std::string>& parameters){

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    parameters["pbs_select"] = std::getenv("PBS_SELECT");
    parameters["pbs_ncpus"] = std::getenv("PBS_NCPUS");
    parameters["pbs_mem"] = std::getenv("PBS_MEM");
    
    std::cout << "--- Environment information ----" << std::endl;
    std::cout << "PBS Select       : " << parameters["pbs_select"] << std::endl;
    std::cout << "PBS Tot CPUs     : " << parameters["pbs_ncpus"] << std::endl;
    std::cout << "PBS Memory       : " << parameters["pbs_mem"] << std::endl;
    std::cout << "OMP Req. Threads : " << parameters["omp_threads"] << std::endl;
    std::cout << "Processes for MPI: " << world_size << std::endl;
    std::cout << "Requested -np    : " << parameters["pbs_np"] << std::endl;
    std::cout << "Threading for OMP: " << omp_get_max_threads() << std::endl;
    std::cout << "--------------------------------\n";
}

void preprocessImage(Image& img, std::unordered_map<std::string, std::string>& parameters) {

    bool verbose = parameters["verbose"] == "true";
    auto startTime = MPI_Wtime();

    // Paths and files
    std::string outputFolder = parameters["output_folder"];
    std::string outputFormat = parameters["image_format"];

    // Parallel processing settings
    const bool parallel = parameters["parallel_preprocessing"] == "true";
    const int numThreads = std::stoi(parameters["omp_threads"]);
    int stepCount = 1; // For output presentation

    if (verbose){  
        std::cout << "\nPreprocessing image:" << std::endl;
        std::cout << " - Input: " << parameters["input"] << std::endl;
        std::cout << " - Parallel Processing: " << (parallel ? "Enabled" : "Disabled") << std::endl;
        if (parallel) std::cout << " - Threads: " << numThreads << std::endl;
    }

    // Convenient function that essentialy perform the passed preprocessing step and then store time and image, eventually.
    auto process_step = [&](auto func, const std::string& stepDescription) {

        // Preprocessing operation
        auto startTime = MPI_Wtime();
        func();
        auto endTime = MPI_Wtime();

        if (parameters["output_disabled"] == "false")
            saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "-" + stepDescription + outputFormat); 
        
        auto duration = endTime - startTime;
        stepCount++;
        if (verbose)
            std::cout << " - Time taken for " << stepDescription << ": " << duration << " s" << std::endl;
    };

    // Preprocessing steps

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
    
    int linesCount, averageVotes, maxVotes, linesAboveThreshold;

    // HT output. For all versions.
    std::vector<std::vector<int>> accumulator;
    std::vector<Segment> segments;
    double precision, recall; // Only with ground truth data -> synthetic samples.

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Hough Transform
    auto startTime = MPI_Wtime();


    // OpenMP only

    if (parameters["HT_parallelism"] == "openMP" && world_rank == 0){

        if (parameters["HT_version"] == "PPHT"){
            std::tie(accumulator, segments) = PPHT_OMP(img, parameters);
        }
        else if (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT"){ 
            std::tie(accumulator, segments) = HT_PHT_OMP(img, parameters);
        }
    }
    
    // MPI only
    else if (parameters["HT_parallelism"] == "MPI"){

        if (parameters["HT_version"] == "PPHT")
                std::tie(accumulator, segments) = PPHT_MPI(img, parameters);
        else
            std::tie(accumulator, segments) = HT_PHT_MPI(img, parameters);
    }

    // OpenMP and MPI
    else if (parameters["HT_parallelism"] == "Hybrid"){
        
        if (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT")
                std::tie(accumulator, segments) = HT_PHT_MPI_OMP(img, parameters);

    }

    // Sequential version (runned with select=1 and ncpus=1) -> Baseline
    else if (parameters["HT_parallelism"] == "None" && world_rank == 0){

        if (parameters["HT_version"] == "HT" || parameters["HT_version"] == "PHT")
            std::tie(accumulator, segments) = HT_PHT(img, parameters);
        else
            std::tie(accumulator, segments) = PPHT(img, parameters);
    }

    
    if (world_rank == 0){

        auto endTime = MPI_Wtime();

        // Cluster lines if requested -> improve precision and recall.
        if ( parameters["HT_version"] != "PPHT" && parameters["cluster_similar_lines"] == "true" && world_rank == 0) 
            segments = clustering(segments, img, parameters);
        
        parameters["htDuration"] = std::to_string(endTime - startTime);

        // Analyze accumulator produced by the transformation.
        std::tie(linesCount, maxVotes, linesAboveThreshold, averageVotes) = analyzeAccumulator(accumulator, std::stoi(parameters["hough_vote_threshold"]));

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

