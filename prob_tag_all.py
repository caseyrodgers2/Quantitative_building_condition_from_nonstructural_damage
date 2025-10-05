# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:05:59 2024

@author: Casey Rodgers

Minor Coding References:
- https://stackoverflow.com/questions/49677313/skip-specific-set-of-columns-when-reading-excel-frame-pandas
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hazus_lite
import prob_tag_building


class prob_tag_all:
    
    ## Initialize
    def __init__(self, damage_fp, fragility_db_fp, hazus_build_fp):
        ## Load clean files
        self.fr_df, self.dm_df, self.bd_df, self.hazus1 = self.load_clean_files(damage_fp, fragility_db_fp, hazus_build_fp)
        
        
    """ Load & Clean Files """
    ## Load & Clean Files
    def load_clean_files(self, damage_fp, fragility_db_fp, hazus_build_fp):
        """
        Load & clean fragility database, given damages, and given buildings
        
        damage_fp:  Damages excel file path. Two sheets:
                             1.) Each row is one element. Element name, damage 
                                state number.
                             2.) Probability of getting a certain tag for a given 
                                location & structural system type.
        fragility_db_fp:   Fragility database file path. Excel file from FEMA P-58.
        hazus_build_fp:    Hazus building capacity filepath
        
        Returns FEMA P-58 fragility dataframe, damages dataframe, buildings dataframe,
            hazus_lite obj
        """
        ## Load & Clean Fragility database
        fr_df = pd.read_excel(fragility_db_fp, sheet_name="Summary", header=2)  # Load file
        fr_df.drop(fr_df.columns[fr_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)  # Delete unnamed cols
        fr_df.drop([0, 1], axis=0, inplace=True) # Delete first two rows that are empty
        #print(fr_df)
        
        ## Load damages file
        dm_df = pd.read_excel(damage_fp, sheet_name="damages")  # Load file
        #dm_df = dm_df.sort_values('NISTIR Classification')  # Sort table
        #dm_df = dm_df.reset_index(drop=True)  # Reset indices to match sorted table
        #print(dm_df)
        
        ## Load buildings file
        bd_df = pd.read_excel(damage_fp, sheet_name="buildings")  # Load file
        
        ## Load location tag probabilities
        hazus1 = hazus_lite.hazus_lite(hazus_build_fp)
        
        return fr_df, dm_df, bd_df, hazus1
        
    
    
    
    """ Main Function """
    ## Calculate Probability tags for all buildings in the file
    def calc_prob_tags_for_all(self, g_info):
        """
        Calculate the Probability of getting a certain tag for all buidings in
        the file.
        """
        ## All Buildings Dataframe
        tot_output_df = ""                  # Use for building components
        tot_build_df = self.bd_df.copy()    # Use for overall building condition
        
        ## Initialize overall building tag
        build_prob_tag_arr = np.zeros((len(self.bd_df.index), 3))
        build_tag_arr = np.zeros((len(self.bd_df.index)))
        build_prior_arr = np.zeros((len(self.bd_df.index), 3))
        build_prior_tag_arr = np.zeros((len(self.bd_df.index)))
        
        ## Go through all buildings
        for i in range(len(self.bd_df.index)):
            # Get building info
            b_info = self.bd_df.loc[i]
            #print(b_info)
            
            # Get current damage dataframe
            curr_bd_id = b_info["Building ID"]
            curr_dm_df = self.dm_df[self.dm_df["Building ID"] == curr_bd_id]
            curr_dm_df = curr_dm_df.reset_index(drop=True)
            #print(curr_dm_df)
            
            # Create prob_tag_building object
            curr_ptb = prob_tag_building.prob_tag_building(curr_dm_df, b_info, g_info, self.fr_df, self.hazus1)
            
            # Get component prob tags, output df, overall build prob tag, tag, prior model
            prob_tag, output_df, build_prob_tag, build_tag, prior_arr = curr_ptb.get_prob_tag_n_tags()
            #print(prob_tag)
            #print(np.sum(prob_tag, axis=1))
            #print(build_prob_tag)
            #print(build_tag)
            
            # Add overall building probs and tags to list
            build_prob_tag_arr[i, :] = build_prob_tag
            build_tag_arr[i] = build_tag
            
            # Add overall prior building probs and prior tags to list
            build_prior_arr[i, :] = prior_arr.T
            build_prior_tag_arr[i] = np.argmax(prior_arr)
            
            # Add df to main dataframe
            if i == 0:
                tot_output_df = output_df
            else:
                tot_output_df = pd.concat([tot_output_df, output_df])
                
        
        ## Add prior building tag probabilities and prior building tag to the dataframe
        tot_build_df.insert(len(tot_build_df.columns), "Prior P(G)", build_prior_arr[:, 0])
        tot_build_df.insert(len(tot_build_df.columns), "Prior P(Y)", build_prior_arr[:, 1])
        tot_build_df.insert(len(tot_build_df.columns), "Prior P(R)", build_prior_arr[:, 2])
        tot_build_df.insert(len(tot_build_df.columns), "Prior Tag", build_prior_tag_arr)
        
        ## Add building tag probabilities and building tag to the dataframe
        tot_build_df.insert(len(tot_build_df.columns), "P(G)", build_prob_tag_arr[:, 0])
        tot_build_df.insert(len(tot_build_df.columns), "P(Y)", build_prob_tag_arr[:, 1])
        tot_build_df.insert(len(tot_build_df.columns), "P(R)", build_prob_tag_arr[:, 2])
        tot_build_df.insert(len(tot_build_df.columns), "Output Tag", build_tag_arr)
        
        ## Write dataframes to files
        tot_output_df.to_excel("outputs/all_building_components.xlsx")
        tot_build_df.to_excel("outputs/all_building_overall.xlsx")
        
        
       