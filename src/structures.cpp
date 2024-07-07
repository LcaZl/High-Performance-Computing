#include "structures.h"

/*************************************************
 *   Structures Encoding for MPI Communication   *
**************************************************/

void createSegmentMPIType(MPI_Datatype* segmentType) {

    int lengths[11] = {2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
    const int numBlocks = 11;
    MPI_Aint displacements[numBlocks];
    MPI_Datatype types[numBlocks] = {
        MPI_2INT, MPI_2INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
        MPI_INT, MPI_2INT, MPI_2INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE
    };

    Segment tempSegment;
    MPI_Aint baseAddress;
    MPI_Get_address(&tempSegment, &baseAddress);
    MPI_Get_address(&tempSegment.start, &displacements[0]);
    MPI_Get_address(&tempSegment.end, &displacements[1]);
    MPI_Get_address(&tempSegment.rho, &displacements[2]);
    MPI_Get_address(&tempSegment.thetaRad, &displacements[3]);
    MPI_Get_address(&tempSegment.thetaDeg, &displacements[4]);
    MPI_Get_address(&tempSegment.votes, &displacements[5]);
    MPI_Get_address(&tempSegment.intersectionStart, &displacements[6]);
    MPI_Get_address(&tempSegment.intersectionEnd, &displacements[7]);
    MPI_Get_address(&tempSegment.interRho, &displacements[8]);
    MPI_Get_address(&tempSegment.interThetaRad, &displacements[9]);
    MPI_Get_address(&tempSegment.interThetaDeg, &displacements[10]);

    for (int i = 0; i < numBlocks; i++) {
        displacements[i] -= baseAddress;
    }

    MPI_Type_create_struct(numBlocks, lengths, displacements, types, segmentType);
    MPI_Type_commit(segmentType);
}

std::unordered_map<std::string, std::string> deserializeParameters(const std::string& serializedParameters) {

    std::unordered_map<std::string, std::string> parameters;
    std::istringstream iss(serializedParameters);
    std::string line;

    while (std::getline(iss, line)) {
        size_t pos = line.find('=');

        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            parameters[key] = value;
        }
    }
    return parameters;
}

std::string serializeParameters(const std::unordered_map<std::string, std::string>& parameters) {

    std::ostringstream oss;

    for (const auto& kv : parameters) 
        oss << kv.first << "=" << kv.second << "\n";

    return oss.str();
}
