process process1 {

    label 'short_low'
    
    input:
    val(input_file)
    path(sample_list)

    output:
    path('chr*_preprocessed')

    """
    python "${System.env.work_dir}"/scripts/XXXX.py ${input_file} ${sample_list}
    """
}

process process2 {

    label 'short_low'
    
    input:
    val(input_file)
    path(sample_list)

    output:
    path('chr*_preprocessed')

    """
    python "${System.env.work_dir}"/scripts/XXX.py ${input_file} ${sample_list}
    """
}
