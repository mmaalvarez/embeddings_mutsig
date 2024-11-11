#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import all processes from modules
include { process1 } from './modules/nn_training.nf'
include { process2 } from './modules/nn_training.nf'

workflow {

    // Create channels
    input_files_ch = Channel
        .fromPath(params.input_files_list)
        .splitCsv(header:false)

    // Run processes
    
    process1(
        input_files_ch,
        file(params.sample_list)
    )
    
    process2(
        params.bcftools_command,
        process1.out
    )

    // Collect output
    results.out
        .collectFile(name: 'res/results.tsv', keepHeader: true)
        .view { "Finished! Results saved in res/results.tsv"}
}

// Error handling
workflow.onError {
    println "Pipeline execution stopped with error: ${workflow.errorMessage}"
}

// Completion handler
workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: ${ workflow.success ? 'SUCCESS' : 'FAILED' }"
    println "Duration: $workflow.duration"
}
