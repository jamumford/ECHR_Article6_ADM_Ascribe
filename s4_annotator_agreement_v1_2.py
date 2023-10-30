# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 18:00:20 2022

@author: Truffles

This script will provide inter-annotator analysis for both violation and non-violation cases. 
Output consists of: a csv file of counts and proportions for each label, and a csv file with 
an inter-annotator score for each annotator. 

Change log:
v1_2 = Added ability to analysis kappa scores by ADM node (and other meta-nodes).
v1_1 = Adjusted to accommodate new json formatting with case_ids as objects.
v1_0 = Functioning code.
"""

import json
import os
import re


def compute_fleiss_kappa(matrix):
    """
    Compute Fleiss' Kappa for the given matrix with variable raters per subject.
    
    Args:
    - matrix (list of list of int): A matrix where each row represents a case and each column 
      represents a classification. The values in the matrix are the counts of the classifications.
      
    Returns:
    - float: The computed Fleiss' Kappa value.
    """
    num_subjects = len(matrix)
    num_categories = len(matrix[0])
    
    # Calculate the observed agreement, P
    P = sum([(sum([i**2 for i in item]) - sum(item)) / (sum(item) * (sum(item) - 1)) for item in matrix]) / num_subjects
    print("P:", P)
    
    # Calculate the expected agreement, Pe
    Pe = sum([(sum([matrix[j][i] for j in range(num_subjects)]) / sum(sum(row) for row in matrix))**2 
              for i in range(num_categories)])
    print("Pe:", Pe)
    
    if Pe == 1:
        kappa = "No variance in category selection!"
    else:
        # Compute Fleiss' Kappa
        kappa = (P - Pe) / (1 - Pe)
    
    return kappa


def load_json(analysis_type):
    analysis_dir = 'Analysis'
    vio_annotation_file = os.path.join(analysis_dir, f"{analysis_type}_vio_stats.json")
    non_annotation_file = os.path.join(analysis_dir, f"{analysis_type}_non_stats.json")
    with open(vio_annotation_file, "r") as file:
        vio_json = [json.loads(line) for line in file]
    with open(non_annotation_file, "r") as file:
        non_json = [json.loads(line) for line in file]
    combined_json = vio_json + non_json
    return {'Violations': vio_json, 'No-violations': non_json, 'Combined': combined_json}
    

def process_json(json_data, subject_analysis):  
    # Extract three-valued tuples as subjects
    matrix_data = []
    subject_dict = {}
    for case in json_data:
        for case_id, subjects in case.items():
            for subject, counts in subjects.items():
                if subject_analysis:
                    if subject not in subject_dict.keys():
                        subject_dict[subject] = []
                    subject_dict[subject].append(counts)
                else:
                    # Ignore 'CRIMINAL LIMB?' and 'FLAGGED?'
                    if subject not in ['CRIMINAL LIMB?', 'FLAGGED?']:
                        if isinstance(counts, list) and len(counts) == 3:  # Check if it's a three-valued tuple
                            matrix_data.append(counts)
    if subject_analysis:
        for subject in subject_dict.keys():
            print("Node:", subject)
            # Filter out subjects (rows) with only one rater from matrix_data
            filtered_matrix_data = [row for row in subject_dict[subject] if sum(row) > 1]
            
            #lprint("filtered_matrix_data:", filtered_matrix_data)
            # Compute Fleiss' Kappa for the data
            fleiss_kappa_value = compute_fleiss_kappa(filtered_matrix_data)
            print(f"Fleiss' Kappa Value: {fleiss_kappa_value}", '\n')
    else:
        # Filter out subjects (rows) with only one rater from matrix_data
        filtered_matrix_data = [row for row in matrix_data if sum(row) > 1]
        
        # Compute Fleiss' Kappa for the data
        fleiss_kappa_value = compute_fleiss_kappa(filtered_matrix_data)
        print(f"Fleiss' Kappa Value: {fleiss_kappa_value}", '\n')
    return


def main(analysis_type, subject_analysis):
    # Load and process the JSON data
    print('Analysis Groups:', analysis_type, '\n')
    json_datasets = load_json(analysis_type)
    #print(json_datasets)
    for key, dataset in json_datasets.items():
        print("Outcome type:", key)
        process_json(dataset, subject_analysis)
    return


if __name__ == "__main__":
    analysis_type = None
    while analysis_type not in ["Overall", "Domain", "General"]:
        analysis_type = input("Please enter the analysis type (Overall, Domain, General): ")
    subject_analysis = None
    while subject_analysis not in ["Yes", "No"]:
        subject_analysis = input("Please enter if analysis is at ADM node and meta-data level (Yes, No): ")
    subject_analysis = subject_analysis == "Yes"
    
    main(analysis_type, subject_analysis)
