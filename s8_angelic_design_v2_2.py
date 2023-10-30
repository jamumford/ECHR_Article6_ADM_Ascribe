# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 01:03:20 2023

@author: Truffles

This script takes the student annotations and generates an input dataset for the AI model,
in which the factor ascription weights are determined by the proportion of the annotators
ascribing that factor. 

Change log
v2_2 = Changed def backward_pass to stop at a particular level of abstraction from
    [leaf, intermediate, issue, outcome]
v2_1 = Changed def backward_pass so that weights are always passed back from parents unless
    node already has a non-empty weight from the annotated data set.
v2_0 = Altered def backward_pass so that the three weights in the array are kept separate
    with no merging of the positive and neutral ascriptions. Also amended 
    def forward_pass so that acceptance of a node is determined by its children if and only
    if it has no prescribed acceptance before input to the function.
v1_0 = Functional code that allows for forward and backwards passes with the assumption
    that neutral ascription should be treated as positive ascription.
"""
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import s7_gen_training_data_v2_0 as norm_annotate


"""
def create_AD imports the legal nodes from the csv file in the same directory.
"""
def create_AD(filename):

    ## Identifies the children for a particular node.
    def identify_children(row):
        if row['Type'] == 'Leaf':
            return []
        else:
            children = A_df.loc[A_df['Parent'] == row['Factor'], 'Factor'].tolist()
        return children   
    
    ## Reads the base csv file and adds a 'Children' column containing each
    ## node's children.      
    A_df = pd.read_csv(filename)
    A_df['Children'] = A_df.apply(identify_children, axis=1)   
    return A_df


"""
def imported_acceptance imports the initial acceptance values for the legal nodes.
"""
def imported_acceptance(outcome):  
    annotations = {} 
    file_name = os.path.join("datasets",f"Overall_{outcome}_stats.json")
    with open(file_name, "r") as file:
        for line in file:
            data = json.loads(line)
            annotations.update(data)
    normalised_annotations = norm_annotate.normalise_data(annotations)
    return normalised_annotations


"""
def backward_pass propagates weights to leaves via a recursive inner function that takes
the weight of the parent and passes it down in accordance with the parent's acceptance
condition and its number of children.
"""
def backward_pass(A_df, instance_df, case_id, abstraction):
    
    ## The recursive internal function that passes the positive and negative
    ## ascription weights to a node in accordance with its parent's weights,
    ## acceptance condition, and number of children.
    def propagate_weights(node):
    
        ## Checking base case where the node has an array of three weights
        ## ascribed by the annotators for the given case. The first weight [0] 
        ## is positive ascription, the second weight [1] is negative ascription, 
        ## and the third weight [2] is neutral ascription.
        if node in instance_df.keys():
            weight = instance_df[node]
            # Convert weight to a numpy array if it's not already
            weight = np.array(weight)
            if not np.isnan(weight).any():
                return weight
        
        ## Main recursion that looks up the parent's weights and passes them
        ## to the node accordingly.
        parent = A_df.loc[A_df['Factor'] == node, 'Parent'].iloc[0]
        parent_weight = propagate_weights(parent)
        num_weights = len(parent_weight)
        assert num_weights == 3
        parent_children = A_df.loc[A_df['Factor'] == parent, 'Children'].iloc[0]
        siblings = len(parent_children)
        parent_condition = A_df.loc[A_df['Factor'] == parent, 'Conditional'].iloc[0]
        assert parent_condition in ['AND', 'OR', 'NAND']
        disjunct = 2**(siblings - 1) / (2**siblings - 1)
        if parent_condition == 'AND':
            annotation = np.array([parent_weight[0], parent_weight[1] * disjunct, parent_weight[2]])
        elif parent_condition == 'OR':
            annotation = np.array([parent_weight[0] * disjunct, parent_weight[1], parent_weight[2]])
        elif parent_condition == 'NAND':
            annotation = np.array([parent_weight[1], parent_weight[0] * disjunct, parent_weight[2]])
        return annotation
    
    ## Identifies the leaf nodes to ascribe and then runs the recursive inner
    ## function def propagate_weights to produce the classification weights.
    nodes = A_df.loc[A_df['Type'] == abstraction, 'Factor'].tolist()
    targets = dict()
    for node in nodes:
        annotation = propagate_weights(node)
        #print("annotation type:", type(annotation), '\n')
        targets.update({node: annotation})          
    return targets


"""
def forward_pass processes the acceptance of the leaf nodes and propagates
the acceptance forward through all levels of the angelic design framework
in an iterative manner, for all nodes that do not already have a defined
acceptance value.
"""
def forward_pass(A_df):

    ## Deriving the acceptance of a node from its prescribed value, or
    ## from the acceptance of its children in accordance with its
    ## acceptance condition.
    def level_up(node):
        result = A_df.loc[A_df['Factor'] == node, 'Acceptance'].iloc[0]
        if result in [1, 0]:
            return result
        condition = A_df.loc[A_df['Factor'] == node, 'Conditional'].iloc[0]
        assert condition in ['AND', 'OR', 'NAND']
        children = A_df.loc[A_df['Factor'] == node, 'Children'].iloc[0]
        children_rows = A_df.loc[A_df['Factor'].isin(children)]
        if condition == 'AND':
            result = (children_rows['Acceptance'] == 1).all()
        elif condition == 'OR':
            result = (children_rows['Acceptance'] == 1).any()
        else:
            result = not (children_rows['Acceptance'] == 1).all()
        return result     

    ## Identifying the level categories via the type description of each node
    ## and then running through each level in turn to determine the acceptance
    ## of the nodes.
    intermediates = A_df.loc[A_df['Type'] == 'Intermediate', 'Factor'].tolist()
    issues = A_df.loc[A_df['Type'] == 'Issue', 'Factor'].tolist()
    outcomes = A_df.loc[A_df['Type'] == 'Outcome', 'Factor'].tolist()
    for intermediate in intermediates:
        A_df.loc[A_df['Factor'] == intermediate, 'Acceptance'] = level_up(intermediate)
    for issue in issues:
        A_df.loc[A_df['Factor'] == issue, 'Acceptance'] = level_up(issue)
    for outcome in outcomes:
        A_df.loc[A_df['Factor'] == outcome, 'Acceptance'] = level_up(outcome)
    return A_df
   
