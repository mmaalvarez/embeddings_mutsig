#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import all processes from modules
include { nn_training } from './modules/nn_training'

workflow {

    // Create channels with hyperparameter ranges

    batch_size = Channel.from(params.batch_size.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    learning_rate = Channel.from(params.learning_rate.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    patience = Channel.from(params.patience.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    fc1_neurons = Channel.from(params.fc1_neurons.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    fc2_neurons = Channel.from(params.fc2_neurons.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    dropout1_rate = Channel.from(params.dropout1_rate.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    dropout2_rate = Channel.from(params.dropout2_rate.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    kernel_size1 = Channel.from(params.kernel_size1.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    kernel_size2 = Channel.from(params.kernel_size2.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    kernel_size3 = Channel.from(params.kernel_size3.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))

    combined_hyperparameters = batch_size
        .combine(learning_rate)
        .combine(patience)
        .combine(fc1_neurons)
        .combine(fc2_neurons)
        .combine(dropout1_rate)
        .combine(dropout2_rate)
        .combine(kernel_size1)
        .combine(kernel_size2)
        .combine(kernel_size3)

    // Run processes
    
    nn_training(
        params.work_dir,
        params.files_dir,
        params.training_set,
        params.validation_set,
        params.testing_set,
        params.all_sets,
        combined_hyperparameters
    )
}

workflow.onError {
    println "Pipeline execution stopped with error: ${workflow.errorMessage}"
}

workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: ${ workflow.success ? 'SUCCESS' : 'FAILED' }"
    println "Duration: $workflow.duration"
}
