# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:56:40 2024

@author: Casey Rodgers

References:
- Hazus FEMA Hazus Earthquake Model Technical Manual 5-1

Minor Coding References:
- https://stackoverflow.com/questions/28766692/how-to-find-the-intersection-of-two-graphs
- https://stackoverflow.com/questions/25439243/find-the-area-between-two-curves-plotted-in-matplotlib-fill-between-area

"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon


class hazus_lite:
    
    ## Initialize
    def __init__(self, hazus_build_fp, dt = 1e-4):
        self.dt = dt       # dt (s) for curves capacity spectrum method
        
        ## Building Capacity Databases
        self.bc_hi_df = pd.read_excel(hazus_build_fp, sheet_name="High", index_col=1)
        self.bc_mod_df = pd.read_excel(hazus_build_fp, sheet_name="Moderate", index_col=1) 
        self.bc_low_df = pd.read_excel(hazus_build_fp, sheet_name="Low", index_col=1) 
        self.bc_pre_df = pd.read_excel(hazus_build_fp, sheet_name="Pre", index_col=1) 
        
        
        
    """ Capacity Spectrum Method Functions """      
        
    ## Create Response Spectrum
    def create_resp_spec(self, b_id, pga, pgv, sa03, sa10, mag, B_eff, B_tvd, B_tav):
        """
        Creates a standardized response spectrum following Hazus 5.1 Section 4.1.3.2. Standard Shape
        of the Response Spectra, given several ground motion characteristics at 
        a select location pga, pgv, sa03, sa10, mag. Characteristics must be taken
        from a ShakeMap. Includes adjusted damping and hysteretic energy dissipation.
        
        b_id: Building id as a str
        pga: Peak Ground Acceleration (g)
        pgv: Peak Ground Velocity (cm/s)
        sa03: Spectral acceleration at a period of 0.3 sec (g)
        sa10: Spectral acceleration at a period of 1.0 sec (g)
        mag: Magnitude
        B_eff: Effective Damping. If == -1, then no adjusted damping
        B_tvd: Effective Damping at period T_vd
        B_tav: Effective Damping at period T_av
        
        Returns the response spectrum for a given earthquake at a specific location.
        The reponse spectrum is returned as two np arrays: one for spectral displacement
        and another for corresponding spectral acceleration.
        """
        ## Calculate Ra and Rv (Hazus Section 5.6.1.)
        # Ra(B_eff), Rv(B_eff)
        Ra_Beff = 1
        Rv_Beff = 1
        
        # Ra(B_tvd), Rv(B_tvd)
        Rv_Btvd = 1
        
        # Ra(B_tav), Rv(B_tav)
        Ra_Btav = 1
        Rv_Btav = 1
        
        if not B_eff == -1:
            # Ra(B_eff), Rv(B_eff)
            Ra_Beff = 2.12 / (3.21 - 0.68 * np.log(B_eff))
            Rv_Beff = 1.65 / (2.31 - 0.41 * np.log(B_eff))
            
            # Rv(B_tvd)
            Rv_Btvd = 1.65 / (2.31 - 0.41 * np.log(B_tvd))
            
            # Ra(B_tav), Rv(B_tav)
            Ra_Btav = 2.12 / (3.21 - 0.68 * np.log(B_tav))
            Rv_Btav = 1.65 / (2.31 - 0.41 * np.log(B_tav))
            
        #print(Ra_Beff, Rv_Beff, Rv_Btvd, Ra_Btav, Rv_Btav)
        
        ## Find Tav and Tvd
        T_av = sa10 / sa03 * (Ra_Btav / Rv_Btav)  # Period where const sp acc and const sp vel meet
        #print(f"new Tav = {T_av}")
        T_vd = 10 ** ((mag - 5) / 2)  # Period where const sp vel and const disp meet
        #print(f"mag: {mag}, T_av: {T_av}, T_vd: {T_vd}")
    
        ## Constant spectral acceleration part (Sa(T) = sa03)
        x1 = np.linspace(0, T_av, int(T_av / self.dt))
        y1 = np.ones(x1.shape) * sa03 / Ra_Beff
        #print(f"x1 = {x1}")
        #print(f"y1 = {y1}")
        
        ## Constant spectral velocity part (Sa(T) = sa10 / T)
        x2 = np.linspace(T_av, T_vd, int((T_vd - T_av) / self.dt))
        y2 = np.ones(x2.shape) * sa10 / x2 / Rv_Beff
        #print(f"x2 = {x2}")
        #print(f"y2 = {y2}")
        
        ## Constant spectral displacement part (Sa(T) = sa10 * Tvd / T^2)
        x3 = np.linspace(T_vd, 2 * T_vd, int((2 * T_vd - T_vd) / self.dt))
        y3 = np.ones(x3.shape) * sa10 * T_vd / x3**2 / Rv_Btvd
        #print(f"x3 = {x3}")
        #print(f"y3 = {y3}")
        
        ## Put it together
        x = np.hstack((x1, x2, x3))
        y = np.hstack((y1, y2, y3))
          
        ## Convert from period to spectral displacement
        x_sd = 9.8 * y * x**2
        
        ## Truncate to a certain value for the graph
        #x_sd = x_sd[x_sd < 50]
        #y = y[0 : x_sd.shape[0]]
        
        ## Plot
        """
        plt.plot(x_sd, y, color="#0077BB", linestyle="dotted")
        #plt.title(f"{b_id} - Response Spectrum: mag={mag}")
        plt.title(f"{b_id} - Response Spectrum")
        plt.ylabel("Spectral Acceleration (g)")
        plt.xlabel("Spectral Displacement (in)")
        plt.xlim(0, 5)
        plt.grid(True)
        plt.show()
        """
        
        ## Return 
        return x_sd, y
    
    
    
    ## Create Building Capacity Curves
    def create_build_cap(self, b_id, Dy, Ay, Du, Au, end_val_mult=2):
        """
        Creates a building capacity curve based on Hazus 5.1 Section 5.4.1. 
        
        b_id: Building id as a str
        Dy: Yield Capacity Point - Spectral Displacement (x)
        Ay: Yield Capacity Point - Spectral Acceleration (y)
        Du: Ultimate Capacity Point - Spectral Displacement (x)
        Au: Ultimate Capacity Point - Spectral Acceleration (y)
        
        Returns the building capacity curve as 2 np arrays: one for spectral displacement
        and another for corresponding spectral acceleration.
        """
        ## Elastic Region (up to yield pt) (Linear)
        x1 = np.linspace(0, Dy, int(Dy / self.dt))
        y1 = Ay/Dy * x1
        
        ## Betwween yield pt and ult pt (Quadratic Bezier Curve)
        t = np.expand_dims(np.linspace(0, 1, int(1 / self.dt)), axis=1)  # t for bezier curve
        
        p0 = np.expand_dims(np.array([Dy, Ay]), axis=0)  # Endpoint 1
        p1 = np.expand_dims(np.array([Au*Dy/Ay, Au]), axis=0)  # Intersection Pt btwn 2 tangents
        p2 = np.expand_dims(np.array([Du, Au]), axis=0)  # Endpoint 2
                            
        xy2 = (1 - t)**2 @ p0 + (2 * (1 - t) * t) @ p1 + t**2 @ p2
        
        ## Past the ult pt (horizontal line)
        x3 = np.linspace(Du, end_val_mult * Du, int((end_val_mult * Du - Du) / self.dt))
        y3 = np.ones(x3.shape) * Au
        
        ## Combine Together
        x = np.hstack((x1, xy2[:, 0], x3))
        y = np.hstack((y1, xy2[:, 1], y3))
        
        ## Plot
        """
        plt.plot(x, y, color="#EE7733", linestyle="solid")
        #plt.title(f"{b_id} - Capacity Curve: Dy={Dy}in, Ay={Ay}in, Du={Du}in, Au={Du}in")
        plt.title(f"{b_id} - Capacity Curve")
        plt.ylabel("Spectral Acceleration (g)")
        plt.xlabel("Spectral Displacement (in)")
        plt.grid(True)
        plt.xlim(0, 5)
        plt.show()
        """
        
        ## Return
        return x, y
    
    
    
    ## Calculate Peak Building Response
    def calc_pbr(self, b_id, b_cap, g_info, B_eff, B_tvd, B_tav):
        """
        Calculates the peak building response, spectral acceleration and spectral
        displacement following Hazus 5.1, using a method similar to the capacity
        spectrum method.
        
        b_id: Building id as a str
        b_cap: Building capacity property list. Includes (in this order): 
            Dy: Yield Capacity Point - Spectral Displacement (x)
            Ay: Yield Capacity Point - Spectral Acceleration (y)
            Du: Ultimate Capacity Point - Spectral Displacement (x)
            Au: Ultimate Capacity Point - Spectral Acceleration (y)            
        g_info: Ground motion info list. Includes (in this order):
            pga: Peak Ground Acceleration (g)
            pgv: Peak Ground Velocity (cm/s)
            sa03: Spectral acceleration at a period of 0.3 sec (g)
            sa10: Spectral acceleration at a period of 1.0 sec (g)
            mag: Magnitude
        B_eff: Effective Damping. If == -1, then no adjusted damping
        B_tvd: Effective Damping at period T_vd
        B_tav: Effective Damping at period T_av
        
        Returns the peak building response in the form of spectral acceleration and
        spectral displacement.
        """     
        ## Create response spectrum and capacity curve
        rs_x, rs_y = self.create_resp_spec(b_id, g_info[0], g_info[1], g_info[2], g_info[3], g_info[4], B_eff, B_tvd, B_tav)
        cc_x, cc_y = self.create_build_cap(b_id, b_cap[0], b_cap[1], b_cap[2], b_cap[3])
        
        ## Interpolate the Capacity curve to have the same x values as the Response spectrum
        cc_y = np.interp(rs_x, cc_x, cc_y)
        cc_x = rs_x
        
        ## Find the intersection pt btwn the response spectrum and capacity curve
        idx = np.argwhere(np.diff(np.sign(rs_y - cc_y))).flatten()
        pb_sd = rs_x[idx]  # Get peak building spectral displacement (in)
        pb_sa = rs_y[idx]  # Get peak building spectral acceleration (g)
        
        ## Truncate arrays to 2 times the intersection point
        rs_x = rs_x[0 : 2 * idx[0]]
        rs_y = rs_y[0 : 2 * idx[0]]
        cc_x = cc_x[0 : 2 * idx[0]]
        cc_y = cc_y[0 : 2 * idx[0]]
        
        ## Plot the reponse spectrum and capacity curve
        """
        plt.plot(rs_x, rs_y, color="#33BBEE", linestyle="dashed")
        plt.plot(cc_x, cc_y, color="#EE7733", linestyle="solid")
        #plt.plot(np.ones(1000) * pb_sd, np.linspace(0, pb_sa, 1000), color='mediumpurple')
        plt.title(f"{b_id} - Capcity Spectrum Method")
        plt.ylabel("Spectral Acceleration (g)")
        plt.xlabel("Spectral Displacement (in)")
        plt.legend(labels=["Response Spectrum", "Capacity Curve"])
        plt.grid(True)
        plt.show()
        """
        
        ## Return peak building spectral acceleration and displacement
        return pb_sd[0], pb_sa[0]
    
    
    
    ## Calculate Building Response (includes effective damping)
    def calc_br_wdamp(self, b_id, b_cap, g_info, kappa, Be):
        """
        Calculates the peak building response, spectral acceleration and spectral
        displacement following Hazus 5.1, using a method similar to the capacity
        spectrum method.
        
        b_id: Building id as a str
        b_cap: Building capacity property list. Includes (in this order): 
            Dy: Yield Capacity Point - Spectral Displacement (x)
            Ay: Yield Capacity Point - Spectral Acceleration (y)
            Du: Ultimate Capacity Point - Spectral Displacement (x)
            Au: Ultimate Capacity Point - Spectral Acceleration (y)            
        g_info: Ground motion info list. Includes (in this order):
            pga: Peak Ground Acceleration (g)
            pgv: Peak Ground Velocity (cm/s)
            sa03: Spectral acceleration at a period of 0.3 sec (g)
            sa10: Spectral acceleration at a period of 1.0 sec (g)
            mag: Magnitude
        kappa: Degradation factor that defines the effective amount of hysteretic damping
            as a function of earthquake duration
        B_e: Elastic Damping Coefficient
        
        Returns the building response in the form of spectral acceleration and
        spectral displacement with accommodations for damping.
        """     
        ## Calculate peak building response without damping
        pb_sd, pb_sa = self.calc_pbr(b_id, b_cap, g_info, -1, -1, -1)
        
        ## Calculate Effective Damping
        #print(f"kappa={kappa}")
        #print(f"pb_sd={pb_sd}, pb_sa={pb_sa}")
        area, A_pb = self.get_hyst_area(b_id, b_cap, pb_sd, pb_sa)
        B_eff = Be + kappa * area / (2 * np.pi * pb_sd * A_pb) * 100
        #print(f"B_eff={B_eff}")
        
        ## Calculate Effective Damping at Tav
        T_av = g_info[3] / g_info[2]  # Period where const sp acc and const sp vel meet
        Sa_tav = g_info[2]  # Spec. Accel at T
        Sd_tav = 9.8 * Sa_tav * T_av ** 2
        
        #print(f"Sd_tav={Sd_tav}")
        #area_tav, A_tav = self.get_hyst_area(b_id, b_cap, Sd_tav, -1)
        B_tav = B_eff
        #print(f"B_tav={B_tav}")
        
        ## Calculate Effective Damping at Tvd
        T_vd = 10 ** ((g_info[4] - 5) / 2)  # Period where const sp vel and const disp meet
        Sa_tvd = g_info[3] / T_vd  # Spec. Accel. at T_vd
        Sd_tvd = 9.8 * Sa_tvd * T_vd ** 2  # Spec. Displ. at T_vd
        
        #print(f"Sd_tvd={Sd_tvd}")
        area_tvd, A_tvd = self.get_hyst_area(b_id, b_cap, Sd_tvd, -1)
        B_tvd = Be + kappa * area_tvd / (2 * np.pi * Sd_tvd * A_tvd) * 100
        #print(f"B_tvd={B_tvd}")
        
        ## Calculate peak building response without damping
        pb_sd, pb_sa = self.calc_pbr(b_id, b_cap, g_info, B_eff, B_tvd, B_tav)
        
        #return B_eff, B_tav, B_tvd
        #print(f"pb_sd={pb_sd}, pb_sa={pb_sa}")
        return pb_sd, pb_sa
        
        
    
    """ Get Building Capacity Dataframe """
    
    ### Get building capacity dataframe based on seismic design level.
    def get_build_df(self, b_info):
        """
        Get building capacity dataframe based on seismic design level.
        
        b_info: Building info Data Series.
            
        Returns building capacity dataframe based on seismic design level, which
            is based on construction year and UBC zone.
        """
        # Get construction year
        constr_yr = b_info["Construction Year"]
        
        # Return seismic design level dataframe based on constr year
        # Values for UBC Zone 4
        if constr_yr > 1975:
            return self.bc_hi_df
        
        elif constr_yr <= 1975 and constr_yr >= 1941:
            return self.bc_mod_df
        
        elif constr_yr < 1941 and b_info["Building Type"] == "W1":
            return self.bc_mod_df
        
        else:
            return self.bc_pre_df
        
        
        
    ## Calculate Hysteresis Loop Area 
    def get_hyst_area(self, b_id, b_cap, D, A):
        """
        Calculate Hysteresis Loop Area, following Hazus 5.1 Section 5.6.1.1.
        Symmetrical push pull of the buliding capacity curve up to peak positive
        and negative displacements.
        
        b_id: Building ID as a string
        b_cap: Building capacity property list. Includes (in this order): 
            Dy: Yield Capacity Point - Spectral Displacement (x)
            Ay: Yield Capacity Point - Spectral Acceleration (y)
            Du: Ultimate Capacity Point - Spectral Displacement (x)
            Au: Ultimate Capacity Point - Spectral Acceleration (y)
        D: Building spectral displacement response
        A: Building spectral acceleration response. If == -1, then calculate A
            from D using the capacity curve
        
        Returns:
        area: Area of Hysteresis Loop
        """
        ## Get building capacity curve
        end_val_mult = D / b_cap[2] + 2  # Make sure curve has enough values
        cc_x, cc_y = self.create_build_cap(b_id, b_cap[0], b_cap[1], b_cap[2], b_cap[3], end_val_mult)
        
        ## Calculate A if A == -1
        if A == -1:
            A = np.interp(D, cc_x, cc_y)
            
        ## Take curve up to displacement D
        cc_x = cc_x[cc_x <= D]
        cc_y = cc_y[0 : len(cc_x)]
        
        ## Interpolate it to have a consistent dt
        last_x = cc_x[len(cc_x) - 1]  # Last x value
        new_cc_x = np.linspace(0, last_x, int(last_x / self.dt))  # New x array
        new_cc_y = np.interp(new_cc_x, cc_x, cc_y)  # New y array
        
        ## Slopes of the sides of the trapezoid
        m1 = b_cap[1] / b_cap[0]  # m1 = Ay / Dy
        
        y_end = new_cc_y[len(new_cc_y)-1]  # Last y val
        y_end1 = new_cc_y[len(new_cc_y)-2] # Second to last y val
        x_end = new_cc_x[len(new_cc_x)-1]  # Last x val
        x_end1 = new_cc_x[len(new_cc_x)-2] # Second to last x val
        
        m2 = (y_end - y_end1) / (x_end - x_end1)
        
        ## Find the 4 pts of the quadrilateral
        # Adjust if D > Du
        new_D = D
        new_A = A
        if D > b_cap[2]:
            new_D = b_cap[2]
            new_A = b_cap[3]
        
        # Tips of Hysteresis loop
        # Adjusted for Ultmate capacity
        p0 = np.expand_dims(np.array([-new_D, -new_A]), axis=0)  # Southwest most point
        p2 = np.expand_dims(np.array([new_D, new_A]), axis=0)  # Northeast most point
        
        # Original
        p0_2 = np.expand_dims(np.array([-D, -A]), axis=0)  # Southwest most point
        p2_2 = np.expand_dims(np.array([D, A]), axis=0)  # Northeast most point
        
        # Intersection of slopes from tips
        x_num = 2 * new_A - (m2 + m1) * new_D  # Numerator for fraction that equals 
        
        x1 = x_num / (m1 - m2)  # Point 1 x
        y1 = m1 * x1 + (-new_A + m1 * new_D)  # Point 1 y
        #y1_2 = m2 * x1 + (new_A - m2 * new_D)
        
        x3 = x_num / (m2 - m1)  # Point 3 x
        y3 = m2 * x3 + (-new_A + m2 * new_D)  # Point 3 y
        #y3_2 = m1 * x3 + (new_A - m1 * new_D) 
        
        p1 = np.expand_dims(np.array([x1, y1]), axis=0)  # Northwest most point
        p3 = np.expand_dims(np.array([x3, y3]), axis=0) # Southeast most point
        
        ## Get the curves that make up the two curves of the loop
        t = np.expand_dims(np.linspace(0, 1, int(1 / self.dt)), axis=1)  # t for bezier curve
        
        # Top Curve
        xy_top = (1 - t)**2 @ p0_2 + (2 * (1 - t) * t) @ p1 + t**2 @ p2
        
        # Bottom Curve
        xy_bot = (1 - t)**2 @ p2_2 + (2 * (1 - t) * t) @ p3 + t**2 @ p0
        
        ## Properly put together polygon
        xy = 0
        if D > b_cap[2]:
            top_hor_line = np.array([[b_cap[2], b_cap[3]], [D, A]])
            bot_hor_line = np.array([[-b_cap[2], -b_cap[3]], [-D, -A]])
            xy = np.vstack((xy_top, top_hor_line, xy_bot, bot_hor_line))
        else:
            xy = np.vstack((xy_top, xy_bot))
        
        ## Get area
        polygon = Polygon(xy)
        area = polygon.area
        #print(f"area={area}")
        
        """
        ## Calculate Area based on simple quadrilateral
        # Horizontal top line
        line1 = [[x1, y1], [D, A]]  # Horizontal top line
        line2 = [[D, A], [x3, y3]]  # Slanted east line
        line3 = [[x3, y3], [-D, -A]]  # Horizontal bottom line
        line4 = [[-D, -A], [x1, y1]]  # Slanted west line
        
        # Create polygon
        xy2 = np.vstack((line1, line2, line3, line4))
        polygon2 = Polygon(xy2)
        area3 = polygon2.area
        #print(f"area3={area3}")
        """
        
        # Plot the Curve
        """
        plt.plot(new_cc_x, new_cc_y)
        plt.plot(xy[:, 0], xy[:, 1], "r")
        #plt.plot(xy2[:, 0], xy2[:, 1], "g")
        
        #plt.title(f"Hysteresis Loop: D={D}, A={A}")
        plt.title(f"{b_id}: Hysteresis Loop")
        plt.ylabel("Spectral Acceleration (g)")
        plt.xlabel("Spectral Displacement (in)")
        plt.legend(["Capacity Curve", "Outline of Area"])
        plt.grid(True)
        plt.show()
        """
        
        return area, A
            
    
    
    ## Get building properties from Hazus data excel
    def get_build_prop(self, b_info, mag):
        """
        Get building capacity properties from Hazus data excel for given building
        
        b_info: Building info Data Series.
        mag: Ground motion magnitude
        
        Returns:
        b_cap: Building capacity property list. Includes (in this order): 
            Dy: Yield Capacity Point - Spectral Displacement (x)
            Ay: Yield Capacity Point - Spectral Acceleration (y)
            Du: Ultimate Capacity Point - Spectral Displacement (x)
            Au: Ultimate Capacity Point - Spectral Acceleration (y)
        theta_arr: Np array of thetas from Hazus for each damage state
        beta_arr: Np array of betas from Hazus for each damage state
        bounds_arr: Np 3x2 array of tag thresholds
        kappa: Degradation factor that defines the effective amount of hysteretic damping
            as a function of earthquake duration
        B_e: Elastic Damping Coefficient
        ns_theta_arr: Np array of thetas from Hazus for each damage state for nonstructural components
        ns_beta_arr: Np array of betas from Hazus for each damage state for nonstructural components
        """
        ## Get the building capacity curve dataframe. 
        # Load the correct seismic design level dataframe.
        # Set the Building Type column to be the index column
        bc_df = self.get_build_df(b_info)
        
        ## Extract Row for Building Type
        bc_row = bc_df.loc[b_info["Building Type"]]
        
        ## Extract Building Capacity Values
        b_cap = [bc_row["Dy (in)"], bc_row["Ay (g)"], bc_row["Du (in)"], bc_row["Au (g)"]]
        
        ## Extract Building Damage State theta, beta, and bounds values
        ds_names = ["Slight", "Moderate", "Extensive", "Complete"] # Ds names
        theta_arr = np.zeros(len(ds_names))  # Initialize array for thetas
        beta_arr = np.zeros(len(ds_names))  # Initialize array for betas
        thres_arr = np.zeros(len(ds_names)) # Initialize array for thresholds
        
        # Go through and add theta and beta values to the array
        for i in range(len(ds_names)):
            theta_arr[i] = bc_row[ds_names[i] + " Median"]
            beta_arr[i] = bc_row[ds_names[i] + " Beta"]
            thres_arr[i] = bc_row[ds_names[i] + " Interstory Drift Threshold"]
            
        # Bounds array
        # Green: [0, extensive threshold]
        # Yellow: [extensive threshold, complete threshold]
        # Red: [complete threshold, np.inf]
        bounds_arr = np.array([[0, thres_arr[2]], [thres_arr[2], thres_arr[3]], [thres_arr[3], np.inf]])
        
        ## Effective Damping
        # Get Kappa
        kappa = 0
        if mag <= 5.5:
            kappa = bc_row["Kappa / Short Dur."]
        elif mag >= 7.5:
            kappa = bc_row["Kappa / Long Dur."]
        else:
            kappa = bc_row["Kappa / Medium Dur."]
        
        # Damping
        Be = bc_row["Damping"]
        
        ## Nonstructural Damage State Theta and Beta values
        # Initialize arrays
        ns_theta_arr = np.zeros(len(ds_names))  # Initialize array for thetas
        ns_beta_arr = np.zeros(len(ds_names))   # Initialize array for betas
        
        for i in range(len(ds_names)):
            ns_theta_arr[i] = bc_row["NS " + ds_names[i] + " Median"]
            ns_beta_arr[i] = bc_row["NS " + ds_names[i] + " Beta"]
        
        ## Return values
        return b_cap, theta_arr, beta_arr, bounds_arr, kappa, Be, ns_theta_arr, ns_beta_arr
    
    
    
    """ Plot CDF Curves """
    
    def plot_cdf_curves(self, b_id, theta_arr, beta_arr, pb_sd):
        """
        Plots cdf curves.
        
        b_id: Building ID as a string
        theta_arr: Np array of thetas from Hazus for each damage state
        beta_arr: Np array of betas from Hazus for each damage state
        pb_sd: Peak building spectral displacement
        """
        ## Set up x
        x = np.linspace(0.01, pb_sd * 2, int(pb_sd * 200))
        
        ## Set up x and y for vertical line at p_sd
        x_line = np.ones(1000) * pb_sd
        y_line = np.linspace(0, 1, 1000)
        
        ## Colors and Linestyles
        colors = ["#CC6677", "#88CCEE", "#DDCC77", "#117733"]
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        
        # Find y and plot
        for i in range(4):
            y = self.prob_ds_sd(x, theta_arr[i], beta_arr[i])
            plt.plot(x, y, color=colors[i], linestyle=linestyles[i])
            
        plt.plot(x_line, y_line, color='#AA4499', linestyle=(0, (5, 10)))
        plt.xlabel("Spectral Displacement, SD (m)")
        plt.ylabel("Probability ds >= DS | SD")
        plt.title(f"{b_id} - Damage State Fragility Curves")
        plt.legend(labels=["Slight", "Moderate", "Extensive", "Complete"])
        plt.grid(True)
        plt.show()
        
        
        
    """ Building Tag Probability """
    
    ## Calculate the probability of being in a damage state given spectral displacement
    def prob_ds_sd(self, S, theta, B):
        """
        Calculates the fragility cdf at spectral displacement S given theta and B 
        for a specific building type, height, and seismic design level.
        
        S:      Specified spectral displacement
        theta:  Median spectral displacement
        B:      Dispersion beta, which indicates uncertainty that the damage state 
                  will initiate at this value of spectral displacement
        
        Returns the fragility cdf at S given theta and B for a specific building type,
        height, and seismic design level.
        """
        part1 = np.log(S/theta) / (B * np.sqrt(2))
        return 1/2 * (1 + sp.special.erf(part1))
        
        
    
    ## Calculate the probabilities of receiving a certain tag
    def calc_prob_tags(self, b_info, g_info):
        """
        Calculates the probability of receiving a certain tag based on Hazus methodology.
        
        b_info: Building info Data Series.
        g_info: Ground motion info list. Includes (in this order):
            pga, pgv, sa03, sa10, mag
        
        Returns a 3x1 np array of tag probabilities, 3x2 np array of thresholds for damage,
            np array of thetas for nonstructural components, np array of betas for nonstructural components
        """
        ## Get building properties
        b_id = b_info["Building ID"]
        b_cap, theta_arr, beta_arr, bounds_arr, kappa, Be, ns_theta_arr, ns_beta_arr = self.get_build_prop(b_info, g_info[4])
        #print(b_cap, theta_arr, beta_arr)
        
        ## Get peak building response: spectral acceleration and spectral displacement
        #pb_sd, pb_sa = self.calc_pbr(b_id, b_cap, g_info, -1, -1, -1)
        pb_sd, pb_sa = self.calc_br_wdamp(b_id, b_cap, g_info, kappa, Be)
        #print(pb_sd, pb_sa)
        
        ## Calculate Probability of >= each damage state (including no damage)
        p_gte_ds = np.ones(len(theta_arr) + 1)  # Initialize p_ds array 
        p_gte_ds[1:p_gte_ds.shape[0]] = self.prob_ds_sd(pb_sd, theta_arr, beta_arr)
        #print(p_gte_ds)
            
        ## Calculate Probability of = each damage state (including no damage)
        # P(DS_j) = F(DS_j) - F(DS_j+1)
        p_ds = np.diff(p_gte_ds) * -1
        p_ds = np.clip(p_ds, a_min=0, a_max=None)
        p_ds = np.append(p_ds, p_gte_ds[4])
        #print(f"p_ds={p_ds}")
        #print(p_ds)
        
        ## Put it in terms of ATC-20 tags: green, yellow, red
        # No damage, slight, and moderate = green
        # Extensive = yellow
        # Complete = green
        p_ds_atc = np.array([p_ds[0] + p_ds[1] + p_ds[2], p_ds[3], p_ds[4]])
        #print(f"p_ds_atc={p_ds_atc}")
        
        ## Plot damage state cdf fragility curves
        #self.plot_cdf_curves(b_id, theta_arr, beta_arr, pb_sd)
        
        return p_ds_atc, bounds_arr, ns_theta_arr, ns_beta_arr
        
        