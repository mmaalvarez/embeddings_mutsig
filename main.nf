#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import all processes from modules
include { nn_training } from './modules/nn_training'

workflow {

    // Create channels with hyperparameter ranges

    batch_size = Channel.from(params.batch_size.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    learning_rate = Channel.from(params.learning_rate.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    patience = Channel.from(params.patience.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    kernel_size_conv1 = Channel.from(params.kernel_size_conv1.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    out_channels_conv1 = Channel.from(params.out_channels_conv1.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    kernel_size_maxpool1 = Channel.from(params.kernel_size_maxpool1.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    kernel_size_conv2 = Channel.from(params.kernel_size_conv2.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    out_channels_conv2 = Channel.from(params.out_channels_conv2.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    fc1_neurons = Channel.from(params.fc1_neurons.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    dropout_fc1 = Channel.from(params.dropout_fc1.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    fc2_neurons = Channel.from(params.fc2_neurons.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    dropout_fc2 = Channel.from(params.dropout_fc2.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    kmer = Channel.from(params.kmer.toString().replaceAll(/[\[\]\s]/, '').tokenize(','))
    
    combined_hyperparameters = batch_size
        .combine(learning_rate)
        .combine(patience)
        .combine(kernel_size_conv1)
        .combine(out_channels_conv1)
        .combine(kernel_size_maxpool1)
        .combine(kernel_size_conv2)
        .combine(out_channels_conv2)
        .combine(fc1_neurons)
        .combine(dropout_fc1)
        .combine(fc2_neurons)
        .combine(dropout_fc2)
        .combine(kmer)

    // Run processes
    
    nn_training(
        params.work_dir,
        params.files_dir,
        params.training_set,
        params.validation_set,
        params.testing_set,
        params.all_sets,
        combined_hyperparameters,
        params.epochs,
        params.training_perc,
        params.validation_perc,
        params.test_perc,
        params.subsetting_seed
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
