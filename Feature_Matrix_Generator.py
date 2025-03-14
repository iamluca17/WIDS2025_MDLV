import numpy as np
import pandas as pd



class Feature_Matrix_Generator:
    
    def __init__(self):

        pass

    # Extract X and Y from column names
    def extract_coords_from_name(self, col_name):
        """Extract row (X) and column (Y) indices from 'xthrow_ythcolumn' format."""
        parts = col_name.split("_")
        x = int(parts[0][:-5]) + 1 # Remove "throw" suffix and convert to int and +1 Assuming zero-based indexing
        y = int(parts[1][:-8])  # Remove "thcolumn" suffix and convert to int
        
        return x, y

    # Get unique X and Y values to determine matrix size
    def shape(self, df):
        coords = [self.extract_coords_from_name(col) for col in df.columns]
        x_max = max(x for x, y in coords)
        y_max = max(y for x, y in coords)

        return x_max, y_max

    # Convert each row into a 2D matrix
    def gen_connectome_matrix(self, df):
        connectome_matrices = []

        x_max, y_max = self.shape(df)

        for i, row in df.iterrows():
            matrix = np.zeros((x_max, y_max))  # Initialize empty matrix
            for col, value in row.items():
                x, y = self.extract_coords_from_name(col)
                x = x - 1
                y = y - 1
                matrix[x, y] = value  # Fill in matrix
            connectome_matrices.append(matrix)

        # Convert to NumPy array (shape: [num_samples, x_max, y_max, 1])
        return np.array(connectome_matrices).reshape(len(df), x_max, y_max, 1)


