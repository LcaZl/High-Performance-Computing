
#include "functions.h"

void environmentInformation(std::unordered_map<std::string, std::string>& parameters){
    
    // Get the number of available threads, processors and nodes.
    int numThreads = omp_get_max_threads();
    int numProcs = omp_get_num_procs();
    int maxThreads = omp_get_thread_limit();

    parameters["omp_num_threads"] = std::to_string(numThreads);
    parameters["omp_num_procs"] = std::to_string(numProcs);
    parameters["omp_max_threads"] = std::to_string(maxThreads);
    parameters["pbs_select"] = std::getenv("PBS_SELECT");
    parameters["pbs_ncpus"] = std::getenv("PBS_NCPUS");
    parameters["pbs_mem"] = std::getenv("PBS_MEM");

    std::cout << "------ Environment information ------\n\n";
    std::cout << "PBS Select      : " << parameters["pbs_select"] << std::endl;
    std::cout << "PBS N. cpus     : " << parameters["pbs_ncpus"] << std::endl;
    std::cout << "PBS Memory      : " << parameters["pbs_mem"] << std::endl;
    std::cout << "OMP Threads     : " << parameters["omp_num_threads"] << std::endl;
    std::cout << "OMP Processes   : " << parameters["omp_num_procs"] << std::endl;
    std::cout << "OMP Max Threads : " << parameters["omp_num_procs"] << std::endl;
}

void preprocessImage(Image& img, std::unordered_map<std::string, std::string>& parameters, bool verbose) {
    
    auto startTime = std::chrono::high_resolution_clock::now();

    // Paths and File Names
    std::string outputFolder = parameters["output_folder"];
    std::string outputFormat = parameters["image_format"];

    // Parallel Processing
    const bool parallel = parameters["parallel"] == "true";
    const int threadCount = std::stoi(parameters["thread_count"]);
    int stepCount = 1;
    std::cout << "\nPreprocessing image:" << std::endl;

    if (verbose){  
        std::cout << " - Input: " << parameters["input"] << std::endl;
        std::cout << " - Parallel Processing: " << (parallel ? "Enabled" : "Disabled") << std::endl;
        std::cout << " - Threads: " << threadCount << std::endl;
    }

    auto process_step = [&](auto func, const std::string& stepDescription) {
        auto startTime = std::chrono::high_resolution_clock::now();
        func();
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << " - Time taken for " << stepDescription << ": " << duration << " ms" << std::endl;
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
                gaussianBlurParallel(img, gbKernelSize, gbSigma, threadCount, verbose); 
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
                sobelEdgeDetectionParallel(img, sedThreshold, sedScaleFactor, threadCount); 
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

    auto endTime = std::chrono::high_resolution_clock::now();
    parameters["preprocessingDuration"] = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
}


std::vector<Segment> HoughTransformation(Image& img, std::unordered_map<std::string, std::string>& parameters, bool verbose) {
    
    auto startTime = std::chrono::high_resolution_clock::now();

    int houghVoteThreshold = std::stoi(parameters["hough_vote_threshold"]);
    double houghTheta = std::stod(parameters["hough_theta"]);
    int pphtLineGap = std::stoi(parameters["ppht_line_gap"]);
    int pphtLineLen = std::stoi(parameters["ppht_line_len"]);
    bool parallel = parameters["parallel"] == "true";
    std::string parallelType = parameters["parallel_type"];
    std::string version = parameters["HT_version"];
    int samplingRate = std::stoi(parameters["sampling_rate"]);
    int threadCount = std::stoi(parameters["thread_count"]);
    bool probabilistic = (version == "PHT" || version == "PPHT");
    bool clustering = parameters["cluster_similar_lines"] == "true";
    double clusterThetaTr = std::stod(parameters["cluster_theta_threshold"]);
    double clusterRhoTr = std::stod(parameters["cluster_rho_threshold"]);
    int lineCount, maxVotes, avgVotes;
    double detectedLines;

    // Verbose logging
    if (verbose) {
        std::cout << "\nHough Transform for Line Detection:\nParameters:\n"
                  << " - Version                 : " << version << std::endl
                  << " - Probabilistic           : " << (probabilistic ? "Yes" : "No") << std::endl
                  << " - Parallel                : " << (parallel ? "Enabled" : "Disabled") << std::endl
                  << " - Parallelization Library : " << parallelType << std::endl
                  << " - Threads                 : " << threadCount << std::endl
                  << " - Vote threshold          : " << houghVoteThreshold << std::endl
                  << " - Theta                   : " << houghTheta << std::endl
                  << " - Sampling Rate           : " << samplingRate << "% (only for probabilistic version)" << std::endl
                  << " - Clustering              : " << (clustering ? "Yes" : "No") << std::endl
                  << " - Cls. Rho Threshold      : " << clusterRhoTr << std::endl
                  << " - Cls. Theta Threshold    : " << clusterThetaTr << std::endl;
    }

    // Compute Hough Transform
    std::vector<std::vector<int>> accumulator;
    if (parallel && parallelType == "openMP"){
        accumulator = parallelHoughTransform(img, houghTheta, probabilistic, samplingRate, threadCount);
    }
    else if (parallel && parallelType == "MPI"){
        // MPI part integration

    }
    else{
        accumulator = houghTransform(img, houghTheta, probabilistic, samplingRate);
    }

    // Analyze accumulator
    std::tie(lineCount, maxVotes, detectedLines, avgVotes) = analyzeAccumulator(accumulator, houghVoteThreshold);

    if (verbose) {
        std::cout << "Output:\n"
                  << " - Number of lines found   : " << lineCount << "\n"
                  << " - Maximum votes for a line: " << maxVotes << "\n"
                  << " - Average votes for a line: " << avgVotes << "\n"
                  << " - Number of valid lines   : " << detectedLines << "\n";
    }

    // Line extraction based on the version and parallelization
    std::vector<Segment> segments;
    if (parallel && parallelType == "openMP") {
        segments = (version == "HT" || version == "PHT") ?
            linesExtractionParallel(accumulator, img, houghVoteThreshold, houghTheta, threadCount) :
            linesProgressiveExtractionParallel(accumulator, img, houghVoteThreshold, houghTheta, pphtLineGap, pphtLineLen, threadCount);
    }
    else if (parallel && parallelType == "MPI"){

        // MPI part integration
    }
    else {
        segments = (version == "HT" || version == "PHT") ?
            linesExtraction(accumulator, img, houghVoteThreshold, houghTheta) :
            linesProgressiveExtraction(accumulator, img, houghVoteThreshold, houghTheta, pphtLineGap, pphtLineLen);
    }

    // Cluster lines if applicable
    if (clustering && version != "PPHT") {
        segments = mergeSimilarLines(segments, img, clusterRhoTr, clusterThetaTr);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    parameters["htDuration"] = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());

    return segments;
}

void loadGroundTruthData(const std::string &gtPath, std::unordered_map<std::string, std::vector<Segment>>& gtData, std::unordered_map<std::string, int>& gtLinesPerImage){

    std::ifstream file(gtPath);
    std::string line;

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
        int linesCount = std::stoi(parts[8]);
        double intersectionRho = std::stod(parts[13]);
        double rho = std::stod(parts[5]);
        double thetaRad = std::stod(parts[6]);
        double thetaDeg = std::stod(parts[7]);
        double intersectionThetaRad = std::stod(parts[14]);
        double intersectionThetaDeg = std::stod(parts[15]);

        Segment gt_seg(Point(x1, y1), Point(x2, y2), rho, thetaRad, thetaDeg, Point(intersectionStartX, intersectionStartY), Point(intersectionEndX, intersectionEndY), intersectionRho, intersectionThetaRad, intersectionThetaDeg);

        // Add the gt line segment to the corresponding image in the map
        gtData[imageName].push_back(gt_seg);
        gtLinesPerImage[imageName] = linesCount;
    }
}

void processDataset(std::unordered_map<std::string, std::string>& parameters){

    auto startTime = std::chrono::high_resolution_clock::now();

    const bool verbose = parameters["verbose"] == "true";
    std::vector<double> precisions;
    std::vector<double> recalls;
    std::unordered_map<std::string, std::vector<Segment>> gtData;
    std::unordered_map<std::string, int> gtLinesPerImage;
    std::vector<double> preprocessingTimes;
    std::vector<double> htTimes;

    loadGroundTruthData(parameters["input"] + "/ground_truth.csv", gtData, gtLinesPerImage);

    for (auto& entry : gtData) {

        const std::string imageName = entry.first;
        std::vector<Segment>& gtSegments = entry.second;
        std::vector<std::vector<int>> accumulator;
        std::vector<Segment> segments;
        double precision, recall;

        // Prevent duplication of paths
        parameters["image_name"] = imageName.substr(0, imageName.find_last_of("."));
        std::string const inputImage = parameters["input"] + imageName;

        std::cout << "\n------------------------------------------ SAMPLE " + imageName + "\n";

        Image img = readImage(inputImage);
        Image imgCopy = readImage(inputImage);
        std::cout << "Image read successfully: " << inputImage << std::endl;

        if (verbose)
            printImageInfo(img);

        preprocessImage(img, parameters, verbose);
        preprocessingTimes.push_back(std::stod(parameters["preprocessingDuration"]));

        segments = HoughTransformation(img, parameters, verbose);
        htTimes.push_back(std::stod(parameters["htDuration"]));

        drawLinesOnImage(gtSegments, imgCopy, 1);
        if (segments.size() > 0)
            drawLinesOnImage(segments, imgCopy, 0);

        saveImage(imgCopy, parameters["output_folder"] + parameters["image_name"] + "-" + parameters["HT_version"] + parameters["image_format"]);

        if (false) { // Set to true to inspect each line detected ----------------------------------------------------------------------------------!
            std::cout << "Most voted line:" << std::endl; printSegmentsInfo(segments);
            std::cout << "Ground truth data:" << std::endl; printSegmentsInfo(gtSegments);
        }

        std::tie(precision, recall) = evaluate(gtSegments, segments, std::stoi(parameters["detection_distance_threshold"]), parameters["HT_version"]);
        precisions.push_back(precision);
        recalls.push_back(recall);

        std::cout << " - Precision: " << precision << std::endl;
        std::cout << " - Recall: " << recall << std::endl;
    }

    parameters["precision"] = std::to_string(calculateMean(precisions));
    parameters["recall"] = std::to_string(calculateMean(recalls));
    parameters["dataset_size"] = std::to_string(recalls.size());
    parameters["preprocessingDuration"] = std::to_string(calculateMean(preprocessingTimes));
    parameters["htDuration"] = std::to_string(calculateMean(htTimes));

    auto endTime = std::chrono::high_resolution_clock::now();
    parameters["dataset_processing_time"] = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());

}