# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:48:03 2024

@author: Casey Rodgers

References:
- https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
- https://www.w3schools.com/python/python_ml_confusion_matrix.asp
- https://medium.com/@gubrani.sanya2/evaluating-multi-class-classification-model-using-confusion-matrix-in-python-4d9344084dfa

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics



## Create stacked bar charts
def tag_stackbar_charts(bd_df, chosen_prop):
    """
    Plot the tags on a stacked bar chart. Assemble buildings by a chosen property.
    
    bd_df: pd dataframe of building results info
    chosen_prop: Chosen property to assemble the buildings by. Can be either:
        Building Type, Seismic Design Level, Inspection Type, Existing Tag
    
    Creates several stacked bar charts to plot the probability of getting a tag.
    """
    ## Sort dataframe by chosen property
    bd_df = bd_df.sort_values(by=[chosen_prop])  # Sort dataframe
    bd_df = bd_df.reset_index(drop=True)  # Reset index
    #print(bd_df["P(G)"])
    
    ## Initialize List to hold values and build_ids
    y_list = []
    build_ids_list = []
    
    ## Go through each building and create a plot for each unique chosen property
    for i in range(len(bd_df.index)):
        ## Collect building probs and add it to y
        curr_build_prob = [bd_df["P(G)"][i], bd_df["P(Y)"][i], bd_df["P(R)"][i]]
        y_list.append(curr_build_prob)
        
        ## Add build ids to list
        build_ids_list.append(bd_df["Building ID"][i])
        
        ## If at end of list or next element has different value for chosen property
        if (i+1 == len(bd_df.index) or not bd_df[chosen_prop][i] == bd_df[chosen_prop][i+1]):
            ## Set up y values
            y_arr = np.array(y_list)  # Turn y into a np array
            
            ## Plot the stacked bar graph
            # Create figure
            fig, ax = plt.subplots()  # Create figure
            
            # Set tag options, width, color, and initialize bottom values
            tag_opt = ["Green", "Yellow", "Red"]
            width = 1 / (np.power(len(y_arr), 1/10)) * 0.13
            #width = 0.2
            #print(width)
            colors = ["#009e73", "#f0e442", "#d55e00"]
            bottom = np.zeros(len(build_ids_list))
            
            # Set x_bar locations
            ax.set_xlim(0, 1)
            x_pos = np.linspace(1, len(y_arr), len(y_arr)) / (len(y_arr) + 1)
            #print(x_pos)
            
            # Go through Green, Yellow, and Red Probs
            for j in range(3):
                # Create bars, update bottom values, and add labels
                p = ax.bar(x_pos, y_arr[:, j], width, label=tag_opt[j], bottom=bottom, color=colors[j])
                bottom += y_arr[:, j]
                #ax.bar_label(p, label_type='center')
                
            # Set title and axes labels
            ax.set_title(chosen_prop + ": " + str(bd_df[chosen_prop][i]))
            ax.set_xlabel("Buliding ID")
            ax.set_ylabel("P(Tag_k)")
            
            # Set y limits
            ax.set_ylim(0, 1)
            
            # Add legend and grid lines
            ax.legend(loc="upper right")
            ax.minorticks_off()
            plt.xticks(x_pos, build_ids_list)
            ax.grid(visible=True, which="major", axis="y")
            ax.grid(visible=True, which="minor", axis="y", linewidth=0.25)
            
            # Show figure
            fig_name = 'plots/' + bd_df[chosen_prop][i] + '.png'
            fig_name = fig_name.replace('->', '_')
            plt.savefig(fig_name, format='png', dpi=300)
            plt.show()
            
            ## Reset y list and build_ids_list
            y_list = []
            build_ids_list = []




## Convert Tag to corresponding int
def tag_to_int(tag_str):
    """
    Convert a string tag to its corresponding int.
    Green = 0, Yellow = 1, Red = 2
    
    tag_str: Tag of type string. "Green", "Yellow", "Red"
    
    Return corresonding int.
    """
    if tag_str == "Green":
        return 0
    elif tag_str == "Yellow":
        return 1
    elif tag_str == "Red":
        return 2
    else:
        return -1
        
    
    
## Get existing tags list, final tags list, and predicted tags list
def get_tag_lists(bd_df, conf_prop, conf_item):
    """
    Get existing tags list, final tags list, and predicted tags list
    
    bd_df: pd dataframe of building results info
    conf_prop: chosen property to include in confusion matrix. If == All, then include all
    conf_item: chosen item of chosen property to include in the confustion matrix.
    
    Return existing tags list, final tags list, and predicted tags list
    """
    ## Initialize list of existing tags, final tags, and predicted tags
    exist_tags_list = []
    final_tags_list = []
    pred_tags_list = []
    
    ## Go through and collect tags
    for i in range(len(bd_df.index)):
        # If chosen property of buliding does not equal to chosen item, then skip
        if not conf_prop == "All":
            if not bd_df[conf_prop][i] == conf_item:
                continue
            
        # Get existing tag
        curr_exist_tag = bd_df["Existing Tag"][i]
        
        # Skip if existing tag is "None"
        if curr_exist_tag == "None":
            continue
        
        # If building changes tag, then add original tag to exist tag and new tag to final tag
        if "->" in curr_exist_tag:
            [tag1, tag2] = curr_exist_tag.split(" -> ")
            exist_tags_list.append(tag_to_int(tag1))
            final_tags_list.append(tag_to_int(tag2))
            
        # If building tag is green, yellow, or red, then add corresponding int to the list
        else:
            curr_int = tag_to_int(curr_exist_tag)
            exist_tags_list.append(curr_int)
            final_tags_list.append(curr_int)
            
        # Add predicted tag to list
        pred_tags_list.append(bd_df["Output Tag"][i])
        
    ## Return lists
    return exist_tags_list, final_tags_list, pred_tags_list



## Create confusion matrix
def create_conf_mat(tags_true, tags_pred, title_str):
    """
    Make confusion matrix to show prediction accuracy. Also give confusion matrix stats
    
    tags_true: True tags int list
    tags_pred: Predicted tags int list
    
    Plot confusion matrix and print confusion matrix stats
    """        
    ## Build confusion matrix for true tag vs predicted tag
    conf_mat1 = metrics.confusion_matrix(tags_true, tags_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat1, display_labels = ["G", "Y", "R"])
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(title_str)
    
    fig_name = 'plots/' + title_str + '.png'
    fig_name = fig_name.replace(':', '_')
    plt.savefig(fig_name, format='png', dpi=300)
    plt.show()
    
    ## Get statics for confusion matrix
    # Accuracy: How often the model is correct. (True P + True N) / Total Predictions
    accuracy = metrics.accuracy_score(tags_true, tags_pred)
    
    # Precision: What % is truly positive? (True P) / (True P + False P)
    #precision = metrics.precision_score(tags_true, tags_pred, average=None)
    
    # Sensitivity (Recall): What % is predicted positive? How good the model is at predicting positives.
    # True P / (True P + False N)
    #recall = metrics.recall_score(tags_true, tags_pred, average=None)
    
    # F-score: "Harmonic mean" of precision and sensitivity
    # 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
    #F1_score = metrics.f1_score(tags_true, tags_pred, average=None)
    
    # Classification Report
    cr = metrics.classification_report(tags_true, tags_pred, target_names=["G", "Y", "R"])

    
    ## Print scores
    print(title_str)
    #print(f"Accuracy: {accuracy}")
    #print(f"Precision: {precision}")
    #print(f"Recall: {recall}")
    #print(f"F1 Score: {F1_score}")
    print(cr)





## Plot tags
def plot_tags(tot_build_df_fp, chosen_prop, conf_prop, conf_item):
    """
    Plot the tags on a stacked bar chart. Assemble buildings by a chosen property.
    Make confusion matrix to show prediction accuracy.
    
    tot_build_df_fp: Building Output dataframe filepath
    chosen_prop: Chosen property to assemble the buildings by. Can be either:
        Building Type, Seismic Design Level, Inspection Type, Existing Tag
    conf_prop: chosen property to include in confusion matrix. If == All, then include all
    conf_item: chosen item of chosen property to include in the confustion matrix.
    
    Creates several stacked bar charts to plot the probability of getting a tag.
    Creates a confusion matrix to show predicted vs true tags for buildings that have
        an existing tag.
    """
    ## Load excel and replace blank spots with "None"
    bd_df = pd.read_excel(tot_build_df_fp, index_col=0)  # Load file
    bd_df = bd_df.replace(np.nan, "None")
    #print(bd_df[chosen_prop])
    
    ## Create stacked bar charts
    tag_stackbar_charts(bd_df, chosen_prop)
    
    ## Get list of tags
    exist_tags_list, final_tags_list, pred_tags_list = get_tag_lists(bd_df, conf_prop, conf_item)
    
    ## Create confusion matrices
    create_conf_mat(exist_tags_list, pred_tags_list, conf_item + ": Original Existing Tag vs Predicted Tag")
    create_conf_mat(final_tags_list, pred_tags_list, conf_item + ": Final Existing Tag vs Predicted Tag")
    
   