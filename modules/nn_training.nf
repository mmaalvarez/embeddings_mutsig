process nn_training {

    time = { "${params.hours_training} h" }
    memory = { "${params.GB_training + 10*(task.attempt-1)} GB" }
    queue = { "${params.partition_training}" }
    cpus = 1
    errorStrategy = 'retry'
    maxRetries = 1
    
    // save output
    publishDir "$PWD/embeddings/CNN_models/", pattern: 'best_model_*.pth', mode: 'copy'
    publishDir "$PWD/embeddings/loss_auc_curves/", pattern: 'loss_and_accuracy_curve_*.png', mode: 'copy'
    publishDir "$PWD/embeddings/saved_embeddings_myCNN/", pattern: 'test_embeddings_probs_*.csv', mode: 'copy'    

    input:
    val(work_dir)
    val(files_dir)
    val(training_set)
    val(validation_set)
    val(testing_set)
    val(all_sets)
    tuple val(batch_size),
          val(learning_rate),
          val(patience),
          val(kernel_size_conv1),
          val(out_channels_conv1),
          val(kernel_size_conv2),
          val(out_channels_conv2),
          val(kernel_size_maxpool),          
          val(fc1_neurons),
          val(dropout_fc1),
          val(fc2_neurons),
          val(dropout_fc2),
          val(kmer)
    val(epochs)
    val(training_perc)
    val(validation_perc)
    val(test_perc)
    val(subsetting_seed)

    output:
    path('best_model_*.pth'), emit: CNN_models
    path('loss_and_accuracy_curve_*.png'), emit: loss_auc_curves
    path('test_embeddings_probs_*.csv'), emit: saved_embeddings_myCNN

    script:
    """
    python "${System.env.work_dir}"/scripts/main.py --work_dir ${work_dir} \
                                                    --files_dir ${files_dir} \
                                                    --training_set ${training_set} \
                                                    --validation_set ${validation_set} \
                                                    --testing_set ${testing_set} \
                                                    --all_sets ${all_sets} \
                                                    --batch_size ${batch_size} \
                                                    --learning_rate ${learning_rate} \
                                                    --patience ${patience} \
                                                    --kernel_size_conv1 ${kernel_size_conv1} \
                                                    --out_channels_conv1 ${out_channels_conv1} \
                                                    --kernel_size_conv2 ${kernel_size_conv2} \
                                                    --out_channels_conv2 ${out_channels_conv2} \
                                                    --kernel_size_maxpool ${kernel_size_maxpool} \
                                                    --fc1_neurons ${fc1_neurons} \
                                                    --dropout_fc1 ${dropout_fc1} \
                                                    --fc2_neurons ${fc2_neurons} \
                                                    --dropout_fc2 ${dropout_fc2} \
                                                    --epochs ${epochs} \
                                                    --kmer ${kmer} \
                                                    --training_perc ${training_perc} \
                                                    --validation_perc ${validation_perc} \
                                                    --test_perc ${test_perc} \
                                                    --subsetting_seed ${subsetting_seed}
    """

    stub:
    """
    touch best_model_STUB.pth
    touch loss_and_accuracy_curve_STUB.png
    touch test_embeddings_probs_STUB.csv
    """
}
