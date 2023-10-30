# -*- coding: utf-8 -*-
"""
Created on Thur Nov 10 03:12:20 2022

@author: Truffles

This script offers a comprehensive analysis of annotated data, focusing on both violation (vio) 
and non-violation (non) cases. 

The script aggregates annotations to produce overall ascription percentages 
for each label. The results are visualised as horizontal bar plots, showcasing the extent to which 
different labels are ascribed across all cases.

The output primarily consists of visual plots saved as PNG files, depicting various facets of the annotation process. 

Change log
v2_1 = Established modular functionality to accept different input json files (Overall, Domain, General).
v2_0 = Switched to accommodate json input, and anonymised data.
v1_0 = Functional code.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def get_sum_values(non_objects, vio_objects, ordered_labels):

    # Initialize a dictionary to store the sum values for each unique nested key
    sum_values = {key: [0, 0, 0] for key in ordered_labels}
    for obj in non_objects:
        for main_key in obj:
            for nested_key in obj[main_key]:
                if nested_key in ordered_labels:
                    for i in range(3):
                        sum_values[nested_key][i] += obj[main_key][nested_key][i]
    for obj in vio_objects:
        for main_key in obj:
            for nested_key in obj[main_key]:
                if nested_key in ordered_labels:
                    for i in range(3):
                        sum_values[nested_key][i] += obj[main_key][nested_key][i]
    
    return sum_values


def process_data(analysis_type, verdict_analysis, analysis_dir):

    # Load the JSON files
    if verdict_analysis not in ["Vio", "Non", "Both"]:
        raise ValueError(f"Invalid verdict_analysis: {verdict_analysis}.")
    
    non_objects = []
    ordered_labels = None
    keys_to_exclude = {"CRIMINAL LIMB?", "FLAGGED?"}
    
    if verdict_analysis in ["Non", "Both"]:
        with open(os.path.join(analysis_dir, analysis_type + "_non_stats.json"), 'r') as f1:
            for line in f1:
                try:
                    obj = json.loads(line)
                    non_objects.append(obj)
                except json.JSONDecodeError:
                    pass
        ordered_labels = [key for key in non_objects[0][list(non_objects[0].keys())[0]] if key not in keys_to_exclude]  
      
    vio_objects = []
    if verdict_analysis in ["Vio", "Both"]:
        with open(os.path.join(analysis_dir, analysis_type + "_vio_stats.json"), 'r') as f2:
            for line in f2:
                try:
                    obj = json.loads(line)
                    vio_objects.append(obj)
                except json.JSONDecodeError:
                    pass
        if ordered_labels == None:
            ordered_labels = [key for key in vio_objects[0][list(vio_objects[0].keys())[0]] if key not in keys_to_exclude]
    
    return non_objects, vio_objects, ordered_labels
    

def make_plot(sum_values, ordered_labels, analysis_dir, analysis_type, verdict_analysis):

    save_dir = os.path.join(analysis_dir, "Ascription_plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Normalize the x-axis values
    labels = list(sum_values.keys())
    stack1_values = [sum_values[key][0] for key in labels]
    stack2_values = [sum_values[key][1] for key in labels]
    stack3_values = [sum_values[key][2] for key in labels]
    total_values = [stack1 + stack2 + stack3 for stack1, stack2, stack3 in zip(stack1_values, stack2_values, stack3_values)]
    max_value = max(total_values)
    scaling_factor = 100 / max_value
    scaled_stack1_values = [value * scaling_factor for value in stack1_values]
    scaled_stack2_values = [value * scaling_factor for value in stack2_values]
    
    # Order keys as in first JSON object
    ordered_stack1_values = [scaled_stack1_values[labels.index(key)] for key in ordered_labels]
    ordered_stack2_values = [scaled_stack2_values[labels.index(key)] for key in ordered_labels]
    
    # Create the bar chart
    plt.figure(figsize=(12, 10))
    r_ordered = np.arange(len(ordered_labels))
    plt.barh(r_ordered, ordered_stack1_values, color='b', edgecolor='white', label='Pos Ascribe')
    plt.barh(r_ordered, ordered_stack2_values, color='r', edgecolor='white', left=ordered_stack1_values, label='Neg Ascribe')
    plt.xlim(0, 100)
    #plt.ylabel('Keys', fontweight='bold')
    plt.xlabel('Ascription %', fontweight='bold')
    plt.yticks(r_ordered, ordered_labels)
    #plt.title('Horizontal Stacked Bar Chart of Normalized Summed Values')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, analysis_type + '_' + verdict_analysis + '_distribution.png'))


def main(analysis_type, verdict_analysis):
    # Load and process the JSON data
    analysis_dir = "Analysis"
    print('Analysis Groups:', analysis_type)    
    print('Verdicts Plotted:', verdict_analysis, '\n')
    non_objects, vio_objects, ordered_labels = process_data(analysis_type, verdict_analysis, analysis_dir)
    sum_values = get_sum_values(non_objects, vio_objects, ordered_labels)
    make_plot(sum_values, ordered_labels, analysis_dir, analysis_type, verdict_analysis)
    return


if __name__ == "__main__":
    analysis_type = None
    while analysis_type not in ["Overall", "Domain", "General"]:
        analysis_type = input("Please enter the analysis type (Overall, Domain, General): ")
    verdict_analysis = None
    while verdict_analysis not in ["Vio", "Non", "Both"]:
        verdict_analysis = input("Please enter if analysis is at ADM node and meta-data level (Vio, Non, Both): ")
    
    main(analysis_type, verdict_analysis)
    