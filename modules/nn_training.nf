process nn_training {

    label 'short_low'
    
    // save output
    publishDir "$PWD/embeddings/CNN_models/", pattern: 'best_model_*.pth', mode: 'copy'
    publishDir "$PWD/embeddings/loss_auc_curves/", pattern: 'loss_and_accuracy_curve_2conv_*.png', mode: 'copy'
    publishDir "$PWD/embeddings/saved_embeddings_myCNN/", pattern: 'test_embeddings_probs_conv1_*.csv', mode: 'copy'    

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
          val(fc1_neurons),
          val(fc2_neurons),
          val(dropout1_rate),
          val(dropout2_rate),
          val(kernel_size1),
          val(kernel_size2),
          val(kernel_size3)

    output:
    path('best_model_*.pth'), emit: CNN_models
    path('loss_and_accuracy_curve_2conv_*.png'), emit: loss_auc_curves
    path('test_embeddings_probs_conv1_*.csv'), emit: saved_embeddings_myCNN

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
                                                    --fc1_neurons ${fc1_neurons} \
                                                    --fc2_neurons ${fc2_neurons} \
                                                    --dropout1_rate ${dropout1_rate} \
                                                    --dropout2_rate ${dropout2_rate} \
                                                    --kernel_size1 ${kernel_size1} \
                                                    --kernel_size2 ${kernel_size2} \
                                                    --kernel_size3 ${kernel_size3}
    """
}
