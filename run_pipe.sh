#!/bin/bash

## load modules, and set root dir names and partition names, depending on cluster ('executor')

hostname=`echo $HOSTNAME | cut -d"." -f1 | cut -d"-" -f1`

if [[ "$hostname" == "fsupeksvr" ]]; then

	conda activate nextflow
	export root_dir="/g"
	export work_dir="$PWD"
	export executor="slurm"
	export partition_fast_short="normal_prio"
	export partition_slow_long="normal_prio_lon"
	export partition_slowest_unlimited="normal_prio_unli"
	export project_group="agendas"

elif [[ "$hostname" == "irblogin01" ]]; then

	module load R/4.2.1-foss-2022a Nextflow Anaconda3
	export root_dir="/data/gds"
	export work_dir="$PWD"
	export partition_fast_short="irb_cpu_iclk"
	export partition_slow_long="?"
	export partition_slowest_unlimited="?"
	export project_group="?"

else
	echo "ERROR: HOSTNAME is not known: '`echo $HOSTNAME | cut -d"." -f1`'"
fi


mkdir -p log/

nextflow -log $PWD/log/nextflow.log run main.nf -resume -with-dag dag_flowchart.svg
