import numpy
import pandas as pd

def plot_trace(df, nrays:int = 100000, ntrace:int=100, show_sun_vector:bool=False):
        """
        Creates and (optionally) displays a 3D scatter and trace plot. This
        function requires that the Python package `plotly` be installed. 

        Parameters
        ------------
        nrays : int
            Number of individual rays to include in the scatter plot. Very 
            large values may render slowly.
        ntrace : int 
            Number of rays for which traces will be displayed. Large values
            may render slowly
        show_sun_vector : bool
            Flag indicating whether the sun vector should be rendered on the plot
        """

        print("Generating 3D trace plots")
        # Plotting with plotly
        try:
            import plotly.graph_objects as go
        except:
            raise RuntimeError("Missing library: plotly. \n Trace plotting requires the Plotly library to be installed. [$ pip install plotly]")

        # Choose how many points to plot. 
        nn = min(nrays, len(df))
        inds = numpy.random.choice(range(len(df)), size=nn, replace=False)
        
        # Data for a three-dimensional line. Randomly choose points if fewer than the full amount are desired.
        loc_x = df.loc_x.values[inds]
        loc_y = df.loc_y.values[inds]
        loc_z = df.loc_z.values[inds]
        stage = df.stage.values[inds]
        raynum = df.number.values[inds]

        # Generate the 3D scatter plot
        layout = go.Layout(scene=dict(aspectmode='data'))

        if len(list(set(stage))) > 1:
            md = dict( size=0.75, color=stage, colorscale='jet', opacity=0.7, ) 
        else:
            md = dict( size=0.75, color='black', opacity=0.7, ) 

        fig = go.Figure(data=go.Scatter3d(x=loc_x, y=loc_y, z=loc_z, mode='markers', marker=md ), layout=layout )

        # Generate line traces for a subset of randomly selected rays
        for i in numpy.random.choice(raynum, size=50, replace=False):
            dfr = df[df.number == i]    #find all rays numbered 'i'
            ray_x = dfr.loc_x 
            ray_y = dfr.loc_y
            ray_z = dfr.loc_z
            fig.add_trace(go.Scatter3d(x=ray_x, y=ray_y, z=ray_z, mode='lines', line=dict(color='black', width=0.5)))
        # Add a trace for the sun vector
        if show_sun_vector:
            tmp = df[df.stage==1].iloc[0]  #sun is coming from the cos vector of the elements in the first stage. Just take the first.
            sun_vec = numpy.array([-tmp.cos_x,-tmp.cos_y,-tmp.cos_z])  #negative of the vector
            # scale the vector based on the overall size of the sun bounding box
            sunrange = numpy.array([df.loc_x.max()-df.loc_x.min(), df.loc_y.max()-df.loc_y.min(), df.loc_z.max()-df.loc_z.min()])
            # sun_scale = ((self.sunstats['xmax']-self.sunstats['xmin'])**2 + (self.sunstats['ymax']-self.sunstats['ymin'])**2)**.5 *0.75
            # sun_scale = min([sun_scale, df.loc_x.max()])
            sun_vec *= (sunrange*sun_vec).max()
            fig.add_trace(go.Scatter3d(x=[0,sun_vec[0]], y=[0,sun_vec[1]], z=[0,sun_vec[2]], mode='lines', line=dict(color='orange', width=3)))
            fig.add_trace(go.Scatter3d(x=[0,sun_vec[0]], y=[0,sun_vec[1]], z=[0,0], mode='lines', line=dict(color='gray', width=2)))

        fig.update_layout(showlegend=False)
        fig.show()


if __name__ == "__main__":
     
    df = pd.read_csv('test_output_new_sun_model_v15.csv')

    print(df)
    plot_trace(df, nrays=100000)