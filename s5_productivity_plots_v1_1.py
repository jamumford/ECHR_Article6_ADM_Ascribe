# -*- coding: utf-8 -*-
"""
Created on Thur Nov 10 03:12:20 2022
@author: Truffles

This script offers a comprehensive analysis of group productivity of annotated data, focusing 
on both violation (vio) and non-violation (non) cases. The output primarily consists of 
visual plots saved as PNG files, depicting various facets of the annotation process.

Change log
v1_1 = Improved presentation of the code.
v1_0 = Functioning code.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Ensure integer markers


def module_student_plot(outcome, results_dir, student_df):
    """Generate and save module-specific student distribution plots."""
    metric = 'Vio_cases_per_hour' if outcome == "vio" else 'Non_cases_per_hour'
    
    working_df = student_df[['Group', metric]].copy()
    domain_df = working_df[working_df['Group'] == 'Domain']
    general_df = working_df[working_df['Group'] == 'General']

    data = {
        'Domain': domain_df[metric].tolist(),
        'General': general_df[metric].tolist()
    }

    fig, ax = plt.subplots()
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys())
    plt.ylim(0, 22)
    plt.ylabel("Cases per hour")
    plt.savefig(os.path.join(results_dir, f"{outcome}_student_distribution.png"))
    return
    

def overall_student_plot(outcome_types, results_dir, student_df):
    """Generate and save an overall student distribution plot."""
    columns_to_plot = ['Vio_cases_per_hour', 'Non_cases_per_hour']
    student_df[columns_to_plot].plot.box()
    
    plt.ylim(0, 22)
    plt.ylabel("Cases per hour")
    plt.xticks([1, 2], ['Violations', 'Non-violations'])
    plt.savefig(os.path.join(results_dir, 'Overall_student_distribution.png'))
    return


def umbrella_plot(outcome_types):
    """Generate various plots based on outcome types."""
    student_dir = os.path.join("Analysis", "Students")
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)
    student_file = os.path.join("Analysis", "Annotator_stats.csv")
    student_df = pd.read_csv(student_file)
    
    overall_student_plot(outcome_types, student_dir, student_df)
    
    for outcome in outcome_types:
        module_student_plot(outcome, student_dir, student_df)
    return


if __name__ == "__main__":
    umbrella_plot(["vio", "non"])
