# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 18:00:20 2022

@author: Truffles

This script takes the student annotations and generates an input dataset for the AI model,
in which the factor ascription weights are determined by the proportion of the annotators
ascribing that factor. This version specifically saves the pickle file via protocol 4
in order to align with the libraries in the Barkla cluster.

Change log:
v2_0 = Adapts to accept json file input and output.
v1_1 = Functional code.
"""


import numpy as np
import os
import pandas as pd
import json
    

def gen_ascribe_df(outcome, ascribe_type):
    """
    Loads the "ascribed" annotated data from the specified directory, given the outcome and ascription groups involved.

    Parameters:
    - outcome (str): The specific outcome for which the annotation data is needed ("vio" or "non").
    - ascribe_type (str): Type of ascription ("Overall" or "Domain" or "Non-domain").

    Returns:
    - df (pandas DataFrame): The processed annotation data after removing certain unnecessary columns.
    """
    ascribe_dir = os.path.join("Analysis")
    file_name = os.path.join(ascribe_dir, f"{ascribe_type}_{outcome}_stats.json")
    annotations = {}
    with open(file_name, 'r') as file:
        for line in file:
            data = json.loads(line)
            annotations.update(data)
    return annotations


def main(ascribe_type):
    """
    Processes annotated data for each provided outcome to determine the "quality" of annotations.

    Functionality:
    - For each outcome, it loads the associated annotation data.
    - Counts how many columns in the annotation data have no annotations.
    - Normalises each labeling based on the total number of annotators.
    - Saves the processed annotation data both as a Pickle file and as an Excel file.
    """
    outcomes = ["non", "vio"]
    for outcome in outcomes:
        count_empty = 0
        annotations = gen_ascribe_df(outcome, ascribe_type)          
        annotations = normalise_data(annotations)
    return annotations


def normalise_data(annotations):

    for case_id in annotations.keys():
        for node in annotations[case_id].keys():
            node_values = annotations[case_id][node]
            num_annotators = sum(node_values)
            if num_annotators != 0:
                # Access the list and update each element
                for i in range(len(node_values)):
                    node_values[i] /= num_annotators
                    
    return annotations


if __name__ == "__main__":
    ascribe_type = None
    while ascribe_type not in ["Overall", "Domain", "General"]:
        ascribe_type = input("Please enter the analysis type (Overall, Domain, General): ")
    main(ascribe_type)