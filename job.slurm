#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --mem 4000 # memory pool for each core (4GB)
#SBATCH -t 0-00:01 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -o log/%x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/%x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J tablab-job # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Job to perform
cd /work/dlc1large

# Set up venv (if not already created)
# Create the venv if it doesn't exist
if [ ! -d "tablab" ]; then
  python3 -m venv tablab
  source tablab/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source tablab/bin/activate
fi

# Run your pipeline script
bash run_pipeline.sh

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";