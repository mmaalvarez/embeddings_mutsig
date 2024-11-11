#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import all processes from modules
include { extract_samples_from_vcfs } from './modules/vcf_processing'
include { join_vcfs_plink_QC_LD_PCA } from './modules/vcf_processing'
include { ancestry_clustering } from './modules/ancestry'
include { by_cohort_QC } from './modules/qc'
include { by_cohort_rm_high_het_samples } from './modules/qc'
include { by_cohort_set_missing_snps_as_hom_major_alleles } from './modules/qc'
include { by_cohort_min_n_tumors } from './modules/qc'
include { by_cohort_QC_2 } from './modules/qc'
include { by_cohort_high_IBD } from './modules/qc'
include { by_cohort_QC_final_set_samples } from './modules/qc'
include { by_cohort_rint } from './modules/cohort_analysis'
include { by_cohort_parse_covariates_and_split_by_tumor } from './modules/cohort_analysis'
include { by_cohort_by_tumor_maf_filter_final_set_SNPs } from './modules/cohort_analysis'
include { by_cohort_final_set_SNPs } from './modules/cohort_analysis'
include { by_cohort_GWAS_clumping } from './modules/cohort_analysis'

workflow {
    // Input channels
    input_files_ch = Channel
        .fromPath(params.input_files_list)
        .splitCsv(header:false)
        .map { it[0] }  // get first column

    // 1. Initial VCF Processing
    extract_samples_from_vcfs(
        input_files_ch,
        file(params.sample_list)
    )

    // Collect all preprocessed files
    preprocessed_files = extract_samples_from_vcfs.out.collect()
    
    // 2. Join VCFs and perform initial QC
    join_vcfs_plink_QC_LD_PCA(
        params.bcftools_command,
        preprocessed_files,
        file(params.high_LD_regions),
        file(params.good_mappability_regions)
    )

    // 3. Ancestry clustering
    ancestry_clustering(
        file(params.sample_metadata),
        params.n_PCs_ancestry_clustering,
        join_vcfs_plink_QC_LD_PCA.out.eigenvec,
        params.fraction_ancestry_outliers_trimmed,
        params.samples_cohort,
        params.restriction_factor
    )

    // Create channel for cohort-specific analysis
    samples_in_cohorts = ancestry_clustering.out.samples_in_cohorts
        .splitCsv(header:true, sep:' ')
        .map{ row -> tuple(row.cohort_id, row.samples_in_cohort) }

    // 4. Cohort-specific QC
    by_cohort_QC(
        join_vcfs_plink_QC_LD_PCA.out.plink_files,
        samples_in_cohorts
    )

    // 5. Remove high heterozygosity samples
    by_cohort_rm_high_het_samples(
        by_cohort_QC.out
    )

    // 6. Set missing SNPs as homozygous major alleles
    by_cohort_set_missing_snps_as_hom_major_alleles(
        by_cohort_rm_high_het_samples.out.het_outliers_plink_input
    )

    // 7. Filter by minimum number of tumors
    by_cohort_min_n_tumors(
        by_cohort_set_missing_snps_as_hom_major_alleles.out,
        params.min_samples_cohort_tumor,
        file(params.sample_metadata)
    )

    // 8. Second round of QC
    by_cohort_QC_2(
        by_cohort_min_n_tumors.out
    )

    // 9. High IBD filtering
    by_cohort_high_IBD(
        by_cohort_QC_2.out,
        params.highIBDTh
    )

    // 10. Final set of samples QC
    by_cohort_QC_final_set_samples(
        by_cohort_high_IBD.out.IBD_related
    )

    // 11. Rank-based inverse normal transformation
    by_cohort_rint(
        by_cohort_QC_final_set_samples.out.final_set_samples,
        file(params.sample_metadata),
        file(params.original_continuous_phenotype)
    )

    // 12. Parse covariates and split by tumor
    by_cohort_parse_covariates_and_split_by_tumor(
		file(params.n_PCs_covariates),
        by_cohort_QC_final_set_samples.out.final_set_samples,
        file(params.sample_metadata)
    )

	// split channels by tumor type
	by_cohort_parse_covariates_and_split_by_tumor.out
	  .flatMap { cohort_id, age_sex_tumor_pcs, tumor_types ->
		tumor_types.readLines().withIndex().collect { tumor_type, tumor_type_index ->
		  tuple(cohort_id, age_sex_tumor_pcs, "${cohort_id}_${tumor_type_index}", tumor_type)
		}
	  }
	  .set { age_sex_tumor_pcs_split }

    // 13. MAF filtering for final SNP set
    by_cohort_by_tumor_maf_filter_final_set_SNPs(
        by_cohort_set_missing_snps_as_hom_major_alleles.out
            .join(by_cohort_QC_final_set_samples.out.final_set_samples)
			.combine(age_sex_tumor_pcs_split)
    )

	// regroup all tumor .bim files within cohorts
	by_cohort_by_tumor_maf_filter_final_set_SNPs.out.final_set_SNPs_tumor
		.map { cohort_id, tumor_id, tumor_type, bed, bim, fam -> 
		    [cohort_id, bim] 
		}
		.groupTuple(by: 0)
		.set { tumor_bims_grouped_by_cohort }

    // 14. Generate final set of SNPs
    by_cohort_final_set_SNPs(
        tumor_bims_grouped_by_cohort
    )

    // 15. GWAS + clumping
    by_cohort_GWAS_clumping(
		by_cohort_set_missing_snps_as_hom_major_alleles.out
			.combine(by_cohort_rint.out)
			.combine(by_cohort_parse_covariates_and_split_by_tumor.out)
			.combine(by_cohort_final_set_SNPs.out)
    )
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
