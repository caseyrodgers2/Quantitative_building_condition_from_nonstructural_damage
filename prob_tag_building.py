# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19  4:32:48 2024

@author: Casey Rodgers

References:
- FEMA P-58. Seismic Performance Assessment of Buildings Volume 1 – Methodology Second Edition. 
    https://femap58.atcouncil.org/documents/fema-p-58/24-fema-p-58-volume-1-methodology-second-edition/file
    
Minor Coding References:
- https://stackoverflow.com/questions/20025882/add-a-string-prefix-to-each-value-in-a-pandas-string-column
- https://saturncloud.io/blog/how-to-select-rows-from-a-dataframe-based-on-list-values-in-a-column-in-pandas/
- https://www.geeksforgeeks.org/python-append-suffix-prefix-to-strings-in-list/
    
    
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hazus_lite


class prob_tag_building:
    
    ## Initialize
    def __init__(self, curr_dm_df, b_info, g_info, fr_df, hazus1):
        """
        curr_dm_df:      Given damaged elements database
        b_info:     Building info Data Series.
        g_info:     Ground motion info list. Includes (in this order):
                    pga, pgv, sa03, sa10, mag
        fr_df:      FEMA P-58 fragility dataframe
        hazus1:     hazus_lite obj
        """
        ## Set variables
        self.curr_dm_df = curr_dm_df
        self.b_info = b_info
        self.g_info = g_info
        self.fr_df = fr_df
        self.hazus1 = hazus1
        
        # Create a copy of dm_df for the output_df
        self.output_df = curr_dm_df.copy()
        
        ## Initialize Hazus nonstructural theta and beta arrays
        self.ns_theta_arr = np.zeros((4, 1))
        self.ns_beta_arr = np.zeros((4, 1))
        
        

    """ PDF & CDF functions """
    
    ## PDF Function
    def pdf (self, D, theta, B):
        """
        Calculates the fragility pdf at D given theta and B for a specific 
        component.
        
        D:      Specified damage parameter
        theta:  Median demand
        B:      Dispersion beta, which indicates uncertainty that the damage state 
                  will initiate at this value of demand
        
        Returns the fragility pdf at D given theta and B for a specific component.
        """
        part1 = 1 / (np.sqrt(2*np.pi))
        part2 = np.exp(-1/2*(np.log(D/theta)/B)**2)
        part3 = 1 / (D * B)
        return part1 * part2 * part3
    
    
    
    ## CDF Function 
    def cdf (self, D, theta, B):
        """
        Calculates the fragility cdf at D given theta and B for a specific 
        component, using the direct formula.
        
        D:      Specified damage parameter
        theta:  Median demand
        B:      Dispersion beta, which indicates uncertainty that the damage state 
                  will initiate at this value of demand
        
        Returns the fragility cdf at D given theta and B for a specific component.
        """
        part1 = np.log(D/theta) / (B * np.sqrt(2))
        return 1/2 * (1 + sp.special.erf(part1))
    
    
    
    ## CDF Function 2
    def cdf2 (self, theta, B, bounds):
        """
        Calculates the fragility cdf given theta and B for a specific component, 
        using the integral with bounds "bounds" of the pdf.
        
        theta:  Median demand
        B:      Dispersion beta, which indicates uncertainty that the damage state 
                  will initiate at this value of demand
        bounds: Boundaries [a, b] for the integral. Starting demand a to ending 
                  demand b
        
        Returns the fragility cdf given theta and B for a specific component.
        """
        return sp.integrate.quad(self.pdf, bounds[0], bounds[1], args=(theta,B))[0]
    
    
    
    ## CDF Function for an Array
    def cdf2_arr (self, theta, B, bounds_arr):
        """
        Calculates the fragility cdf for multiple values given theta and B for a 
        specific component.
        
        theta:      Median demand
        B:          Dispersion beta, which indicates uncertainty that the damage 
                      state will initiate at this value of demand
        bounds_arr: Boundaries [a_arr, b_arr] (size N x 2) array for the integral. 
                      a_arr is an array of starting demands. b_arr is an array of 
                      ending demands.
        
        Returns the fragility cdf given theta and B for a specific component.
        """
        # Initialize result array
        N = bounds_arr.shape[0]          # Number of values
        result = np.zeros(N)
        
        # Go through and calculate cdf for each demand
        for i in range(N):
            result[i] = self.cdf2(theta, B, bounds_arr[i, :])
        return result
    
    
    
    ## Function: CDF_j(x) - CDF_j+1(x)
    def fun_cdfj_cdfj1(self, D, theta, B, theta1, B1):
        """
        Function: CDF_j(x) - CDF_j+1(x)
        
        theta:      For j, median demand
        B:          For j, dispersion beta, which indicates uncertainty that the damage 
                      state will initiate at this value of demand
        theta1:     For j+1, median demand
        B1:         For j+1, dispersion beta, which indicates uncertainty that the damage 
                      state will initiate at this value of demand
    
        Returns the function for CDF_j(x) - CDF_j+1(x)
        """
        ## Result
        result = 0
        
        ## If theta1 or B1 is nan, then j+1 state does not exist
        if np.any(np.isnan(theta1)) or np.any(np.isnan(B1)):
            result = self.cdf(D, theta, B)
        ## If theta or B is nan, then j state does not exist (no damage)
        elif np.any(np.isnan(theta)) or np.any(np.isnan(B)):
            result = 1 - self.cdf(D, theta1, B1)
        ## If j and j+1 states exist, then:
        else:
            result = self.cdf(D, theta, B) - self.cdf(D, theta1, B1)
        
        ## Clip result so >= 0
        result = np.clip(result, a_min=0, a_max=None)
        return result
    
    
    
    
    """ Extract Damage State Values from Fragility Database """
    
    ## Extract desired damage state values of given suffix from elem_df
    def extract_vals(self, ds_arr, suffix, elem_df):
        """
        Extract desired damage state values of given suffix
        
        ds_arr:      Array of strings
        suffix:     Suffix to add to all elements in array
        elem_df:    Dataframe of elements that are in the damage file
        
        Returns desired values
        """
        ## Get column names
        col_names = np.array(["DS " + s + suffix for s in ds_arr])  # Get col names by adding suffix
        
        ## Initialize vals and prepare names
        vals = np.zeros(len(ds_arr))
        elem_names = self.output_df['NISTIR Classification']
        
        ## Go through and extract desired values
        for i in range(len(ds_arr)):
            ## If it is a Hazus component
            if elem_names[i] == "Hazus":
                # Get damage state (-1 to shift it match ns arrays)
                ds = int(ds_arr[i]) - 1
                
                # Check if it's for the None damage state or past complete, then put nan
                if ds == -1 or ds == len(self.ns_theta_arr):
                    vals[i] = np.nan
                
                # Return median, beta, or probability
                elif suffix == ", Median Demand":
                    vals[i] = self.ns_theta_arr[ds]
                    
                elif suffix == ", Total Dispersion (Beta)":
                    vals[i] = self.ns_beta_arr[ds]
                    
                elif suffix == ", Probability":
                    vals[i] = 1
                
            ## If it is in the fragility database
            else:
                elem = elem_df.loc[elem_names[i]]  # Get element
                # Check if column exists in element. If so, set value. If not, set nan
                if col_names[i] in elem.keys():
                    curr_val = elem[col_names[i]]  # Current value
                    # If curr_val == "By User", then it's precast cladding and set
                    # value depending on construction year FEMA P-58 vol 2 Section 7.4
                    if curr_val == "By User":
                        # Median Demand
                        if suffix == ", Median Demand":
                            # FEMA P-58 Table 7-6
                            if self.b_info["Construction Year"] <= 1994:
                                curr_val = 0.012
                            else:
                                curr_val = 0.02
                        # Dispersion.
                        elif suffix == ", Total Dispersion (Beta)":
                            curr_val = 0.5
                        # Probability
                        elif suffix == ", Probability":
                            curr_val = 1
                    # Set value to curr_val
                    vals[i] = curr_val
                else:
                    vals[i] = np.nan
        
        ## Return values
        return vals
    
    
    
    ## Extract theta's, B's, and probabilities for given damage states
    def extract_DS_th_b_pr(self, elem_df, curr_or_next):
        """
        Extract damage state's theta, B, and probability values.
        
        elem_df:        Element dataframe that consists of fragility data only for
                            the elements that are damaged.
        curr_or_next:   Do you want current damage state values or next damage
                            state values? 0 - Curr. 1 - Next.
        
        Returns theta_vals, B_vals, prob_vals
        """
        dam_states_str = ''     # Initialize damage states str
        ## Extract damage state string names
        if curr_or_next == 0:
            # Current (j)
            dam_states_str = np.char.mod("%d", self.output_df['Damage State'])  # Damage states strings list
            beg_str = 'Curr'    # Name of column
        else:
            # Next (j+1)
            next_dam_states = np.array(self.output_df['Damage State']) + 1  # Next damage states
            dam_states_str = np.char.mod("%d", next_dam_states)  # Convert to a str array
            beg_str = 'Next'
        
        ## Extract curr damage state theta's, B's, and probabilities
        theta_vals = self.extract_vals(dam_states_str, ", Median Demand", elem_df)
        B_vals = self.extract_vals(dam_states_str, ", Total Dispersion (Beta)", elem_df)
        prob_vals = self.extract_vals(dam_states_str, ", Probability", elem_df)
        
        ## Add columns for Theta, B, and Probability of curr DS for all elements
        self.output_df.insert(len(self.output_df.columns), beg_str + 'Median Demand', theta_vals)
        self.output_df.insert(len(self.output_df.columns), beg_str + 'Dispersion', B_vals)
        self.output_df.insert(len(self.output_df.columns), beg_str + 'Probability', prob_vals)
        
        return theta_vals, B_vals, prob_vals
    
    
    
    ## Extract current & next damage state values and put them in an array
    def extract_curr_next_ds(self):
        """
        Extract current & next damage state values and return value arrays
        
        curr_dm_df:      Given damaged elements database
        b_info:     Building info Data Series.
        
        Returns nx2 theta array, nx2 Beta values array, nx2 probability values array
        """
        ## Extract rows for the elements that are in the damage file
        elem_mask = self.fr_df['NISTIR Classification'].isin(self.curr_dm_df['NISTIR Classification'])
        elem_df = self.fr_df[elem_mask]
        elem_df = elem_df.sort_values('NISTIR Classification')
        elem_df.set_index('NISTIR Classification', inplace=True)
        #print(elem_df)
        
        ## Extract Current & Next Damage State properties
        curr_theta_vals, curr_B_vals, curr_prob_vals = self.extract_DS_th_b_pr(elem_df, 0)  # Current
        next_theta_vals, next_B_vals, next_prob_vals = self.extract_DS_th_b_pr(elem_df, 1)  # Next
        
        ## Stack current and next values together. Convert to float64 type
        theta_arr = np.vstack((curr_theta_vals, next_theta_vals)).T.astype('float64')
        B_arr = np.vstack((curr_B_vals, next_B_vals)).T.astype('float64')
        prob_arr = np.vstack((curr_prob_vals, next_prob_vals)).T.astype('float64')
        
        return theta_arr, B_arr, prob_arr
        
    
    
    
    """ Plot Fragility Curves """
    
    ## Plot CDF or PDF curves with tag boundaries shown
    def plot_frag_curve(self, theta_arr, B_arr, b_g, b_y, b_r, cdf_or_pdf):
        """
        Plot fragility cdf or pdf curves for n elements.
        
        theta_arr:  Array of thetas nx2. 1st col is curr. 2nd col is next.
        B_arr:      Array of Betas nx2. 1st col is curr. 2nd col is next.
        b_g:    Green tag highest boundary
        b_y:    Yellow tag highest boundary
        b_r:    Red tag highest boundary
        cdf_or_pdf: Plot either cdf or pdf. 0 for cdf and 1 for pdf
        
        Returns Array of Probabilities nx1 of damage state ji given tag k
        """
        ## Preparing values
        n = np.shape(theta_arr)[0]  # Get num of elements
        
        for i in range(n):
            ## x values
            x = np.linspace(0.0001, b_r[i], num=int(b_r[i]*1000))  # Get x values
            
            # Initialize
            b_vals = [0, 1]         # y vals for boundary lines
            beg_title_str = "Fragility CDF: "
            
            # Get and plot curr DS y values
            y = np.zeros(x.shape)  # Initialize y
            if cdf_or_pdf == 0:
                # CDF
                y = self.cdf(x, theta_arr[i,0], B_arr[i,0])  # Curr DS
            else:
                # PDF
                y = self.pdf(x, theta_arr[i,0], B_arr[i,0])  # Curr DS
                b_vals[1] = np.max(y)
                beg_title_str = "Fragility PDF: "
            plt.plot(x, y)
            
            # Get and plot next DS y values
            if not np.isnan(theta_arr[i, 1]):
                y2 = np.zeros(x.shape)  # Initialize y2
                if cdf_or_pdf == 0:
                    # CDF
                    y2 = self.cdf(x, theta_arr[i,1], B_arr[i,1])  # Next DS
                else:
                    y2 = self.pdf(x, theta_arr[i,1], B_arr[i,1])  # Next DS
                plt.plot(x, y2)
                
            # Get and plot boundary x vals for vertical lines
            x_gr = [b_g[i], b_g[i]]
            x_ye = [b_y[i], b_y[i]]
            x_re = [b_r[i], b_r[i]]
            plt.plot(x_gr, b_vals, 'g')
            plt.plot(x_ye, b_vals, 'y')
            plt.plot(x_re, b_vals, 'r')
            
            # Get plot title
            title_str = self.output_df['Component Name'][i]
            title_str = title_str[0:35]  # Trim string
            
            # Format plot
            plt.grid(True)
            plt.title(beg_title_str + title_str)
            plt.ylabel("P(DS i >= DS ij | X = x)")
            plt.xlabel("Story Drift Ratio")
            plt.legend(["Current", "Next"])
            plt.show()
            
            
            
            
    """ Calculate Probabilities """
    
    ## Calculate Probability of damage state ji given tag k
    def prob_dsji_tagk(self, theta_arr, B_arr, bounds, y_bounds):
        """
        Calculate Probability of damage state ji given tag k
        
        theta_arr:  Array of thetas nx2. 1st col is curr. 2nd col is next.
        B_arr:      Array of Betas nx2. 1st col is curr. 2nd col is next.
        bounds:    Tag bounds as a 2x1 array
        y_bounds:  Yellow tag bounds. Equals 0 when not for a red tag.
                    A 2x1 array when for a red tag.
        
        Returns Array of Probabilities nx1 of damage state ji given tag k
        """
        ## Create resulting probabilities array
        n = np.shape(theta_arr)[0]  # Number of components (n)
        result_arr = np.zeros((n, 1))   # Results array    
        b_arr = np.zeros((n))        # Upper bound b array    
        
        for i in range(n):
            ## If b = np.inf, then set b to be when next cdf first equals threshold
            b = bounds[1]       # Bound b
            if b == np.inf:
                thres = 0.999   # Threshold value for where to put b
                x_b = np.linspace(0.0001, 1, num=1000)  # x values
                y_cdf = np.array([])    # Initialize y values
                
                # If next cdf doesn't exist, then use curr cdf
                if np.isnan(theta_arr[i,1]):
                    y_cdf = self.cdf(x_b, theta_arr[i, 0], B_arr[i, 0])   # y values
                else:
                    y_cdf = self.cdf(x_b, theta_arr[i, 1], B_arr[i, 1])   # y values
                
                # Find first x value that is >= threshold
                b = x_b[np.argmax(y_cdf>=thres)]
                
                # Check if lower than highest yellow tag bound
                if b <= y_bounds[1]:
                    b = y_bounds[1] * 2
                    
            ## Calculate total probabiity space: 1 * (b-a)
            total_prob = b - bounds[0]
            
            ## Calculate integrals ∫_𝑎^𝑏 (𝐹_𝑗𝑖 (𝑥)− 𝐹_(𝑗𝑖+1) (x))𝑑𝑥
            args_set = (theta_arr[i,0], B_arr[i,0], theta_arr[i,1], B_arr[i,1])  # Prepare args
            Fji_Fji1 = sp.integrate.quad(self.fun_cdfj_cdfj1, bounds[0], b, args=args_set)[0]  # Integrate
            result_arr[i, 0] = Fji_Fji1 / total_prob  # Divide by total prob space
            b_arr[i] = b
        
        ## Return result array
        return result_arr, b_arr
        
    
    
    ## Calculate Tag Probability for each Element
    def calc_tag_prob (self):
        """
        Calculates the tag probability for each nonstructural element
        
        Returns:
            - Prior model tag array
            - a table of nonstructural element tag probabilities. Each row is
            one nonstructural element. Col 1: Green. Col 2: Yellow. Col 3: Red.
        """
        ## Get tag array and bounds
        tag_arr, bounds_arr, self.ns_theta_arr, self.ns_beta_arr = self.hazus1.calc_prob_tags(self.b_info, self.g_info)
        tag_arr = np.expand_dims(tag_arr, axis=1)
        
        ## Tag Story Drift Ratio Cutoffs
        bounds_g = bounds_arr[0, :]
        bounds_y = bounds_arr[1, :]
        bounds_r = bounds_arr[2, :]
        
        ## Extract curr & next DS values into arrays for given elements
        theta_arr, B_arr, prob_arr = self.extract_curr_next_ds()
        #print(f"theta_arr: {theta_arr}")
        #print(f"B_arr: {B_arr}")
        
        ## Calculate Probability of damage state ji given tag k
        prob_ds_g, b_g = self.prob_dsji_tagk(theta_arr, B_arr, bounds_g, 0)
        prob_ds_y, b_y = self.prob_dsji_tagk(theta_arr, B_arr, bounds_y, 0)
        prob_ds_r, b_r = self.prob_dsji_tagk(theta_arr, B_arr, bounds_r, bounds_y)
        prob_ds_tag = np.hstack((prob_ds_g, prob_ds_y, prob_ds_r))
        #print(f"prob_ds_tag: {prob_ds_tag}")
        
        ## Plot fragility curves
        #self.plot_frag_curve(theta_arr, B_arr, b_g, b_y, b_r, 0)
        #self.plot_frag_curve(theta_arr, B_arr, b_g, b_y, b_r, 1)
    
        ## Calculate Probability of Damage State ji
        prob_ds = prob_ds_tag @ tag_arr
        #print(f"prob_ds: {prob_ds}")
        
        ## Prepare tag_arr and prob_ds to be the proper shape
        tag_arr2 = np.repeat(tag_arr.T, prob_ds.shape[0], axis=0)
        prob_ds2 = np.repeat(prob_ds, 3, axis=1)
    
        ## Calculate Probability of Tag k given DS ji
        prob_tag_ds = prob_ds_tag * tag_arr2 / prob_ds2
        #print(f"prob_tag_ds: {prob_tag_ds}")
        
        #print(np.sum(prob_tag_ds, axis=1))
        
        #print(np.prod(prob_tag_ds, axis=0))
        #print(np.sum(np.prod(prob_tag_ds, axis=0)))
        
        #print(np.sum(prob_tag_ds, axis=0))
        #print(np.sum(prob_tag_ds, axis=0) / prob_tag_ds.shape[0])
        
        ## Add results to output excel
        output_tag_arr = np.repeat(tag_arr, prob_ds_g.shape[0], axis=1).T
        self.output_df.insert(len(self.output_df.columns), "P(Green)", output_tag_arr[:, 0])
        self.output_df.insert(len(self.output_df.columns), "P(Yellow)", output_tag_arr[:, 1])
        self.output_df.insert(len(self.output_df.columns), "P(Red)", output_tag_arr[:, 2])
        
        self.output_df.insert(len(self.output_df.columns), "P(DS|Green)", prob_ds_g)
        self.output_df.insert(len(self.output_df.columns), "P(DS|Yellow)", prob_ds_y)
        self.output_df.insert(len(self.output_df.columns), "P(DS|Red)", prob_ds_r)
        
        self.output_df.insert(len(self.output_df.columns), "P(DS)", prob_ds)
        
        self.output_df.insert(len(self.output_df.columns), "P(Green | DS)", prob_tag_ds[:, 0])
        self.output_df.insert(len(self.output_df.columns), "P(Yellow | DS)", prob_tag_ds[:, 1])
        self.output_df.insert(len(self.output_df.columns), "P(Red | DS)", prob_tag_ds[:, 2])
        
        ## Get output tag
        tags = np.argmax(prob_tag_ds, axis=1)  # Get index of max value
        self.output_df.insert(len(self.output_df.columns), "Output Tag", tags)
        #output_df.to_excel("outputs/" + b_info["Building ID"] + ".xlsx")
        
        ## Return results
        return tag_arr, prob_tag_ds
    
    
    
    ## Get overall building tag
    def get_overall_build_tag(self, prob_tag):
        """
        Get overall building tag. Use worst case component.
        
        prob_tag: Np.array of nonstructural element tag probabilities. Each row is
            one nonstructural element. Col 1: Green. Col 2: Yellow. Col 3: Red.
            
        Return 3x1 np.array of overall building tag probabilities and overall building tag.
        """
        ## Get overall building tag (worst tag)
        comp_tags = np.argmax(prob_tag, axis=1)  # Get component tags
        #print(comp_tags)
        build_tag = np.max(comp_tags)
        
        ## Get all components that have that build_tag
        prob_tag_final = prob_tag[comp_tags == build_tag]
        #print(f"prob:{prob_tag_final}")
        
        ## Get worst component (highest probability in:
        #  Y+R for G and Y tags, and R for R tags)
        prob_overall_ind = 0
        if build_tag == 0 or build_tag == 1:
            prob_overall_ind = np.argmax(prob_tag_final[:, 1] + prob_tag_final[:, 2])
        else:
            prob_overall_ind = np.argmax(prob_tag_final[:, 2])
        
        prob_overall = prob_tag_final[prob_overall_ind, :]
        
        ## Return overall building tag probabilities and overall building tag
        return prob_overall, build_tag
    
    
    
    """ Main Function """
    def get_prob_tag_n_tags(self):
        """
        Get component probabilities of tags, output df, overall building probabilities of tags,
        and overall building tag.
            
        Return component probabilities of tags, output df, overall building probabilities of tags,
        overall building tag, and prior model tags
        """
        ## Get component probabilities of tags and output df
        prior_arr, prob_tag_ds = self.calc_tag_prob()
        
        ## Get overall building probabilities of tags and overall building tag
        prob_overall, build_tag = self.get_overall_build_tag(prob_tag_ds)
        
        ## Return outputs
        return prob_tag_ds, self.output_df, prob_overall, build_tag, prior_arr
        
        
