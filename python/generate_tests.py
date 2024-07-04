import os
import itertools

def save_parameters_file(output_dir, cpus, select, place, param_content):
    # Create the directory if it doesn't exist
    dir_path = os.path.join(output_dir, f't_{select}_{cpus}_{place}')
    os.makedirs(dir_path, exist_ok=True)
    
    # List all parameter files in the directory
    existing_files = [f for f in os.listdir(dir_path) if f.startswith('parameters_')]
    
    # Extract indices and find the maximum
    indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    next_index = max(indices, default=0) + 1
    
    # Save the parameter file with the next index
    param_filename = os.path.join(dir_path, f'parameters_{next_index}')
    with open(param_filename, 'w') as param_file:
        param_file.write(param_content)
  

def generate_hough_transform_script(dir_path, cpus, select, place, place_str):
    # Count the number of parameter files in the directory
    param_files = [f for f in os.listdir(dir_path) if f.startswith('parameters_')]

    # Generate the HoughTransform.sh script
    script_content = f"""#!/bin/bash
#PBS -l select={select}:ncpus={cpus}:mem=8gb -l place={place}
#PBS -l walltime=3:00:00
#PBS -N ja_tests_hts
#PBS -q short_cpuQ
#PBS -o output/tests/ht_{select}_{cpus}_{place_str}.out
#PBS -e output/tests/ht_{select}_{cpus}_{place_str}.err

module load python-3.7.2
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0
module load mpich-3.2.1--gcc-9.1.0

# Use previously created virtual environment with OpenCV (see README)
source cv2/bin/activate
PARAM_DIR="HPC/tests/t_{select}_{cpus}_{place_str}"

for PARAM_FILE in $PARAM_DIR/parameters_*; do

    # Dynamically set the environment variables from PBS directives
    export PBS_SELECT={select}
    export PBS_NCPUS={select * cpus} # (select * cpus) -> total number of cpus.
    export PBS_MEM=8
    export NP_VALUE=$(grep "pbs_np=" "$PARAM_FILE" | cut -d '=' -f 2)
    export OMP_PLACES=threads

    # Print the environment variables
    echo "Running with PARAM_FILE=$PARAM_FILE and NP_VALUE=$NP_VALUE"
    echo "PBS_SELECT=$PBS_SELECT"
    echo "PBS_NCPUS=$PBS_NCPUS"
    echo "PBS_MEM=$PBS_MEM"
    echo "OMP_PLACES=$OMP_PLACES"
    echo "NP_VALUE=$NP_VALUE"

    # Get the parameter file for this array job
    mpiexec -np $NP_VALUE ./HPC/HoughTransform "$PARAM_FILE"
    mpiexec -np $NP_VALUE ./HPC/HoughTransform "$PARAM_FILE"
    mpiexec -np $NP_VALUE ./HPC/HoughTransform "$PARAM_FILE"
    
done

# Unset the environment variables and deactivate virtual environment
unset PBS_SELECT
unset PBS_NCPUS
unset PBS_MEM
unset OMP_PLACES
unset NP_VALUE
deactivate
"""
    
    # Save the script in the directory
    script_filename = os.path.join(dir_path, 'HoughTransform.sh')
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
        


if __name__ == "__main__":


    TESTS = 0
    
    
    # Define configurations
    configurations = {
        "images": [
            {
                "input": "HPC/dataset/synthetic_images/syn_img_5k.pnm",
                "output_folder": "HPC/output/syn_img_5k/"
            },
            {
                "input": "HPC/dataset/synthetic_images/syn_img_10k.pnm",
                "output_folder": "HPC/output/syn_img_10k/"
            },
            {
                "input": "HPC/dataset/synthetic_images/syn_img_20k.pnm",
                "output_folder": "HPC/output/syn_img_20k/"
            }
        ],
        "HT_versions": ["HT", "PHT", "PPHT"],
        "HT_parallelisms": [ "openMP", "MPI"],
        "selects": [1, 2, 4, 8],
        "cpus": [2, 4, 8],
        "places":["pack","scatter","pack:excl","scatter:excl"]
    }

    # Template for the parameter file
    param_template = """HT_version='{HT_version}'
HT_parallelism='{HT_parallelism}'
parallel_preprocessing={parallel_preprocessing}
omp_threads={omp_threads}
pbs_np={pbs_np}
pbs_place='{places}'
input='{input}'
output_folder='{output_folder}'
performance_path='HPC/performance/tests/'
verbose=true
output_disabled=true
convert_output=false
converter_program_location='HPC/python/image_converter.py'
conversion_format='jpg'
greyscale_conversion=true
gaussian_blur=false
gb_kernel_size=7
gb_sigma=1.9
histogram_equalization=false
sobel_edge_detection={sobel_edge_detection}
sed_threshold=100
sed_scale_factor=1
hough_vote_threshold={hough_vote_threshold}
hough_theta=360
cluster_similar_lines={cluster_similar_lines}
cluster_theta_threshold=5.0
cluster_rho_threshold=75.0
sampling_rate={sampling_rate}
ppht_line_gap=25
ppht_line_len=50
"""


    # Directory to save the parameter files
    output_dir = "HPC/tests/"
    os.makedirs(output_dir, exist_ok=True)
    mem = 8
    dirs = set()
    
    # Generate the configurations
    for image in configurations["images"]:
        
        # BASELINE SEQUENTIAL TESTS
        for version in configurations["HT_versions"]:
            select = cpus = np = 1
            
            if version == "PPHT":
                cluster_similar_lines = str(False).lower()
                hough_vote_threshold = 50
                sobel_edge_detection = str(True).lower()
                sampling_rate=90
                
                param_content = param_template.format( # Parallel preprocessing True
                    HT_version=version,
                    HT_parallelism="None",
                    parallel_preprocessing=str(True).lower(),
                    omp_threads=1,
                    input=image["input"],
                    output_folder=image["output_folder"],
                    sobel_edge_detection=sobel_edge_detection,
                    hough_vote_threshold=hough_vote_threshold,
                    sampling_rate=sampling_rate,
                    cluster_similar_lines=cluster_similar_lines,
                    pbs_select=select,
                    pbs_cpus=cpus,
                    pbs_mem=mem,
                    pbs_np=np,
                    places="pack"
                )
                save_parameters_file(output_dir, cpus, select, "pack", param_content)
                TESTS += 1
            else:
                cluster_similar_lines = str(True).lower()
                hough_vote_threshold = 120
                sobel_edge_detection = str(False).lower()
                sampling_rate=75

            param_content = param_template.format(  # Parallel preprocessing False
                HT_version=version,
                HT_parallelism="None",
                parallel_preprocessing=str(False).lower(),
                omp_threads=1,
                input=image["input"],
                output_folder=image["output_folder"],
                sobel_edge_detection=sobel_edge_detection,
                hough_vote_threshold=hough_vote_threshold,
                sampling_rate=sampling_rate,
                cluster_similar_lines=cluster_similar_lines,
                pbs_select=select,
                pbs_cpus=cpus,
                pbs_mem=mem,
                pbs_np=np,
                places="pack"
            )
            
            save_parameters_file(output_dir, cpus, select, "pack", param_content)
            TESTS += 1

        
        for version in configurations["HT_versions"]:
    
            # Adjust HT settings accordingly to the version
            if version == "PPHT":
                cluster_similar_lines = str(False).lower()
                hough_vote_threshold = 50
                sobel_edge_detection = str(True).lower()
                sampling_rate=90
            else:
                cluster_similar_lines = str(True).lower()
                hough_vote_threshold = 120
                sobel_edge_detection = str(False).lower()
                sampling_rate=75
            
            
            for  select, cpus, place in itertools.product(configurations["selects"],configurations["cpus"],configurations["places"]):

                # MPI TESTS
                parallel_preprocessing = str(True).lower()
                np = select * cpus
                omp_threads = 1

                place_str = place.replace(':','-')
                
                param_content = param_template.format(
                    HT_version=version,
                    HT_parallelism="MPI",
                    parallel_preprocessing=parallel_preprocessing,
                    omp_threads=omp_threads,
                    input=image["input"],
                    output_folder=image["output_folder"],
                    sobel_edge_detection=sobel_edge_detection,
                    hough_vote_threshold=hough_vote_threshold,
                    sampling_rate=sampling_rate,
                    cluster_similar_lines=cluster_similar_lines,
                    pbs_select=select,
                    pbs_cpus=cpus,
                    pbs_mem=mem,
                    pbs_np=np,
                    places=place_str
                )
                
                save_parameters_file(output_dir, cpus, select, place_str, param_content)
                TESTS += 1

                # OMP TESTS
                if (version != 'PPHT'):
                    parallel_preprocessing = str(True).lower()
                    np = 1
                    omp_threads = cpus * select

                    param_content = param_template.format(
                        HT_version=version,
                        HT_parallelism="openMP",
                        parallel_preprocessing=parallel_preprocessing,
                        omp_threads=omp_threads,
                        input=image["input"],
                        output_folder=image["output_folder"],
                        sobel_edge_detection=sobel_edge_detection,
                        hough_vote_threshold=hough_vote_threshold,
                        sampling_rate=sampling_rate,
                        cluster_similar_lines=cluster_similar_lines,
                        pbs_select=select,
                        pbs_cpus=cpus,
                        pbs_mem=mem,
                        pbs_np=np,
                        places=place_str
                    )
                    save_parameters_file(output_dir, cpus, select, place_str, param_content)
                    TESTS += 1

                    # Hybrid TESTS

                    parallel_preprocessing = str(True).lower()
                    np = select
                    omp_threads = cpus

                    param_content = param_template.format(
                        HT_version=version,
                        HT_parallelism="Hybrid",
                        parallel_preprocessing=parallel_preprocessing,
                        omp_threads=omp_threads,
                        input=image["input"],
                        output_folder=image["output_folder"],
                        sobel_edge_detection=sobel_edge_detection,
                        hough_vote_threshold=hough_vote_threshold,
                        sampling_rate=sampling_rate,
                        cluster_similar_lines=cluster_similar_lines,
                        pbs_select=select,
                        pbs_cpus=cpus,
                        pbs_mem=mem,
                        pbs_np=np,
                        places=place_str
                    )
                    
                    save_parameters_file(output_dir, cpus, select, place_str, param_content)
                    TESTS += 1

    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        if os.path.isdir(folder_path):
            # Extract select and cpus from folder name
            try:
                parts = folder_name.split('_')
                select = int(parts[1])
                cpus = int(parts[2])
                place_str = str(parts[3])
                place = place_str.replace('-',':')
                generate_hough_transform_script(folder_path, cpus, select, place, place_str)
            except (IndexError, ValueError):
                print(f"Skipping invalid directory name: {folder_name}")
                
    print(f"Generated {TESTS} test case. With three-times avg execution each result in {TESTS * 3} test to execute.")