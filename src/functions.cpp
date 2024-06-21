
#include "functions.h"

void environmentInfo(std::unordered_map<std::string, std::string>& parameters){
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_rank == 0){
        // Get the number of available threads, processors and nodes.

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
}

void preprocessImage(Image& img, std::unordered_map<std::string, std::string>& parameters, bool verbose) {

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
        std::cout << " - Threads: " << numThreads << std::endl;
    }

    auto process_step = [&](auto func, const std::string& stepDescription) {
    auto startTime = MPI_Wtime();
        func();
        auto endTime = MPI_Wtime();
        auto duration = endTime - startTime;
        if (verbose)
            std::cout << " - Time taken for " << stepDescription << ": " << duration << " s" << std::endl;
    };

    // Process each step
    if (parameters["greyscale_conversion"] == "true") {

        if (verbose)
            std::cout << "\n[" << stepCount << "] Converting the image to grayscale..." << std::endl;

        process_step([&] { 
            convertToGrayscale(img); 
            if (parameters["run_for"] == "single_image_test")
                saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "_greyscale" + outputFormat); 
        }, "grayscale_conversion");

        stepCount++;
    }

    if (parameters["histogram_equalization"] == "true") {
        if (verbose)
            std::cout << "\n[" << stepCount << "] Applying histogram equalization..." << std::endl;

        process_step([&] { 
            equalizeHistogram(img); 
            if (parameters["run_for"] == "single_image_test")
                saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "_equalized" + outputFormat); 
        }, "histogram_equalization");

        stepCount++;
    }

    if (parameters["gaussian_blur"] == "true") {

        int gbKernelSize = std::stoi(parameters["gb_kernel_size"]);
        float gbSigma = std::stof(parameters["gb_sigma"]);

        if (verbose){
            std::cout << "\n[" << stepCount << "] Blurring using a Gaussian filter..." << std::endl;
            std::cout << "Kernel Size: " << gbKernelSize << ", Sigma: " << gbSigma << std::endl;
        }

        if (parallel)

            process_step([&] { 
                gaussianBlurParallel(img, gbKernelSize, gbSigma, verbose, numThreads); 
                if (parameters["run_for"] == "single_image_test")
                    saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "_blur" + outputFormat); 
            }, "gaussian_blur");

        else

            process_step([&] { 
                gaussianBlur(img, gbKernelSize, gbSigma, verbose);
                if (parameters["run_for"] == "single_image_test")
                    saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "_blur" + outputFormat); 
            }, "gaussian_blur");

        stepCount++;
    }

    if (parameters["sobel_edge_detection"] == "true") {
        int sedThreshold = std::stoi(parameters["sed_threshold"]);
        float sedScaleFactor = std::stof(parameters["sed_scale_factor"]);

        if (verbose){
            std::cout << "\n[" << stepCount << "] Applying Sobel edge detection..." << std::endl;
            std::cout << "Threshold: " << sedThreshold << ", Scale Factor: " << sedScaleFactor << std::endl;
        }
        if (parallel)

            process_step([&] { 
                sobelEdgeDetectionParallel(img, sedThreshold, sedScaleFactor, numThreads); 
                if (parameters["run_for"] == "single_image_test")
                    saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "_sobel" + outputFormat); 
            }, "sobel_edge_detection");

        else

            process_step([&] { 
                sobelEdgeDetection(img, sedThreshold, sedScaleFactor); 
                if (parameters["run_for"] == "single_image_test")
                    saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "_sobel" + outputFormat); 
            }, "sobel_edge_detection");

        stepCount++;

        if (verbose){
            std::cout << "\n[" << stepCount << "] Converting image to binary..." << std::endl;
        }

        process_step([&] { 
            toBinary(img); 
            if (parameters["run_for"] == "single_image_test")
                saveImage(img, outputFolder + parameters["image_name"] + "-" + std::to_string(stepCount) + "_binary" + outputFormat); 
        }, "binary_conversion");
    }

    auto endTime = MPI_Wtime();
    parameters["preprocessingDuration"] = std::to_string(endTime - startTime);
}


std::vector<Segment> HoughTransformation(Image& img, std::unordered_map<std::string, std::string>& parameters, bool verbose) {
    
    int world_rank, linesCount, averageVotes, maxVotes, linesAboveThreshold, serializedImageSize;
    std::string version = parameters["HT_version"];
    int voteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    std::vector<std::vector<int>> accumulator;
    bool clustering = parameters["cluster_similar_lines"] == "true";
    std::vector<unsigned char> serializedImage;
    std::vector<Segment> segments;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Hough Transform

    auto startTime = MPI_Wtime();
    if (parameters["parallel_ht_type"] == "openMP" && world_rank == 0)
            std::tie(accumulator, segments) = houghTransformParallel_OMP(img, parameters);
    
    else if (parameters["parallel_ht_type"] == "MPI" || parameters["parallel_ht_type"] == "Hybrid"){

        if (world_rank == 0){
            serializedImage = img.serialize();
            serializedImageSize = serializedImage.size();
            std::cout << "Serialized image size on root: " << serializedImageSize << std::endl;

        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&serializedImageSize, 1, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast the size of the serialized image
        serializedImage.resize(serializedImageSize); // Resize the buffer on all processes to hold the serialized image data
        MPI_Bcast(serializedImage.data(), serializedImageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD); // Broadcast the serialized image data

        if (world_rank != 0) 
            img = Image::deserialize(serializedImage);

        MPI_Barrier(MPI_COMM_WORLD);       

        std::cout << "[PROCESS "<< world_rank << "] Started HT." << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);

        std::tie(accumulator, segments) = parameters["parallel_ht_type"] == "MPI" ?
        houghTransformParallel_MPI(img, parameters) :
        houghTransformParallel_Hybrid(img, parameters);
    
    }
    
    else if (parameters["parallel_ht_type"] == "None" && world_rank == 0)
            std::tie(accumulator, segments) = houghTransform(img, parameters);
    
    if (world_rank == 0){
        auto endTime = MPI_Wtime();

        // Cluster lines if applicable
        if (clustering && world_rank == 0) {
            segments = mergeSimilarLines(segments, img, parameters);
        }
        parameters["htDuration"] = std::to_string(endTime - startTime);

        // Analyze accumulator
        std::tie(linesCount, maxVotes, linesAboveThreshold, averageVotes) = analyzeAccumulator(accumulator, voteThreshold);
        std::cout << std::endl 
        << "HT for "<< parameters["image_name"] << std::endl
        << "  |- # Lines analyzed        : " << linesCount << std::endl
        << "  |- Max. Votes for a line   : " << maxVotes << std::endl
        << "  |- Avg. Votes for a line   : " << averageVotes << std::endl
        << "  |- # Lines above threshold : " << linesAboveThreshold << std::endl
        << "  |- # Final Segments        : " << segments.size() << std::endl;
    }

    return segments;
}

void processDataset(std::unordered_map<std::string, std::string>& parameters) {
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    auto startTime = MPI_Wtime();
    const bool verbose = parameters["verbose"] == "true";
    std::vector<double> precisions;
    std::vector<double> recalls;
    std::vector<double> preprocessingTimes;
    std::vector<double> htTimes;

    std::unordered_map<std::string, std::vector<Segment>> gtData;
    std::unordered_map<std::string, int> gtLinesPerImage;
    std::vector<std::string> imageNames;
    
    if (world_rank == 0) {
        houghTransformInfo(parameters);
        std::cout << "[Process 0] Loading ground truth data...\n";
        loadGroundTruthData(parameters["input"] + "/ground_truth.csv", gtData, gtLinesPerImage);

        for (const auto& entry : gtData) {
            imageNames.push_back(entry.first);
        }

    }
    
    if (parameters["parallel_ht_type"] != "None"){

        // Broadcast ground truth data size
        int gtDataSize = gtData.size();
        MPI_Bcast(&gtDataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast the image names
        imageNames.resize(gtDataSize);
        for (int i = 0; i < gtDataSize; ++i) {
            char buffer[256];
            if (world_rank == 0) {
                strncpy(buffer, imageNames[i].c_str(), 256);
            }
            MPI_Bcast(buffer, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
            if (world_rank != 0) {
                imageNames[i] = buffer;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Each process prints its assigned images for debugging
        std::cout << "[Process " << world_rank << "] Assigned images: ";
        for (int i = world_rank; i < static_cast<int>(imageNames.size()); i += world_size) {
            std::cout << imageNames[i] << " ";
        }
        std::cout << "\n";

        MPI_Barrier(MPI_COMM_WORLD);

    }
    else
        world_size = 1;
    
    // Each process handles a subset of the dataset
    for (int i = world_rank; i < static_cast<int>(imageNames.size()); i += world_size) {
        const std::string imageName = imageNames[i];
        std::vector<Segment> gtSegments = gtData[imageName];
        std::vector<std::vector<int>> accumulator;
        std::vector<Segment> segments;
        double precision, recall;

        parameters["image_name"] = imageName.substr(0, imageName.find_last_of("."));
        std::string inputImage = parameters["input"] + imageName;

        Image img = readImage(inputImage);
        Image imgCopy = readImage(inputImage);

        if (verbose) 
            std::cout << "[Process " << world_rank << "] Image read successfully: " << inputImage << std::endl;
        
        preprocessImage(img, parameters, false);
        preprocessingTimes.push_back(std::stod(parameters["preprocessingDuration"]));

        segments = HoughTransformation(img, parameters, false);
        htTimes.push_back(std::stod(parameters["htDuration"]));

        drawLinesOnImage(gtSegments, imgCopy, 1);
        if (segments.size() > 0)
            drawLinesOnImage(segments, imgCopy, 0);

        saveImage(imgCopy, parameters["output_folder"] + parameters["image_name"] + "-" + parameters["HT_version"] + parameters["image_format"]);

        if (false) { // Set to true to inspect each line detected
            std::cout << "[Process " << world_rank << "] Most voted line:\n";
            printSegmentsInfo(segments);
            std::cout << "[Process " << world_rank << "] Ground truth data:\n";
            printSegmentsInfo(gtSegments);
        }

        std::tie(precision, recall) = evaluate(gtSegments, segments, std::stoi(parameters["detection_distance_threshold"]), parameters["HT_version"]);
        precisions.push_back(precision);
        recalls.push_back(recall);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "[Process " << world_rank << "] Ended for loop.\n";

    // Calculate local sums for the reduction
    double local_precision_sum = std::accumulate(precisions.begin(), precisions.end(), 0.0);
    double local_recall_sum = std::accumulate(recalls.begin(), recalls.end(), 0.0);
    double local_preprocessing_time_sum = std::accumulate(preprocessingTimes.begin(), preprocessingTimes.end(), 0.0);
    double local_ht_time_sum = std::accumulate(htTimes.begin(), htTimes.end(), 0.0);
    double global_precision_sum, global_recall_sum, global_preprocessing_time_sum, global_ht_time_sum;
    int local_size = precisions.size();
    int global_size;
    
    if (parameters["parallel_ht_type"] != "None"){

        // Gather counts and sums from all processes
        MPI_Reduce(&local_precision_sum, &global_precision_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_recall_sum, &global_recall_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_preprocessing_time_sum, &global_preprocessing_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_ht_time_sum, &global_ht_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else{
        global_precision_sum = local_precision_sum;
        global_recall_sum = local_recall_sum;
        global_preprocessing_time_sum = local_preprocessing_time_sum;
        global_ht_time_sum = local_ht_time_sum;
        global_size = local_size;
    }

    if (world_rank == 0) {
        parameters["precision"] = std::to_string(global_precision_sum / global_size);
        parameters["recall"] = std::to_string(global_recall_sum / global_size);
        parameters["dataset_size"] = std::to_string(global_size);
        parameters["preprocessingDuration"] = std::to_string(global_preprocessing_time_sum / global_size);
        parameters["htDuration"] = std::to_string(global_ht_time_sum / global_size);

        auto endTime = MPI_Wtime();
        parameters["dataset_processing_time"] = std::to_string(endTime - startTime);
    }
}