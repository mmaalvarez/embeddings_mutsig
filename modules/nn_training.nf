process extract_samples_from_vcfs {

    label 'process_medium'
    
    input:
    val(input_file)
    path(sample_list)

    output:
    path('chr*_preprocessed')

    """
    bash "${System.env.work_dir}"/scripts/0_extract_samples_from_vcfs.sh ${input_file} ${sample_list}
    """
}

process join_vcfs_plink_QC_LD_PCA {

    label 'process_medium'
    
    input:
    val(bcftools_command)
    path(preprocessed_files_list)
    path(high_LD_regions)
    path(good_mappability_regions)

    output:
    tuple path('passed_biallelic_autosomal_snps_cancer_samples_noHighLDregions_CRG75_geno_mind.bed'),
          path('passed_biallelic_autosomal_snps_cancer_samples_noHighLDregions_CRG75_geno_mind.bim'),
          path('passed_biallelic_autosomal_snps_cancer_samples_noHighLDregions_CRG75_geno_mind.fam'), emit: plink_files
    path('LD_pruned.eigenvec'), emit: eigenvec

    """
    bash "${System.env.work_dir}"/scripts/1_join_vcfs_plink_QC_LD_PCA.sh ${bcftools_command} ${preprocessed_files_list} ${high_LD_regions} ${good_mappability_regions}
    """
}
