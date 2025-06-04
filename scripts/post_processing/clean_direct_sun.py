import numpy as np
import csv
import pandas as pd

# remove hitpoints on the receiver coming directly from the sun for CPU comparison 
def clean_receiver_direct_sun_hits(filename, output_filename):
    df = pd.read_csv(filename)

    # Group by 'number' and filter out groups that contain 'stage' 2 but not 'stage' 1
    valid_numbers = df.groupby('number').filter(lambda group: 1 in group['stage'].values or 2 not in group['stage'].values)['number'].unique()

    # Keep only rows where 'number' is in the valid numbers
    df_cleaned = df[df['number'].isin(valid_numbers)]

    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(output_filename, index=False)

    print(f"Cleaned data saved to {output_filename}")

if __name__ == "__main__":
    folder_dir = "C:/Users/allie/Documents/SolTrace/hit_point_data/"

    filename = folder_dir + "cylinder_rec_tilted_gpu.csv"
    output_filename = folder_dir + "cylinder_rec_tilted_gpu_cleaned.csv"

    clean_receiver_direct_sun_hits(filename, output_filename)

