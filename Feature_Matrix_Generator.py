import numpy as np
import pandas as pd



class Feature_Matrix_Generator:
    
    def __init__(self):

        pass

    # Extract X and Y from column names
    def extract_coords_from_name(self, col_name):
        """Extract row (X) and column (Y) indices from 'xthrow_ythcolumn' format."""
        parts = col_name.split("_")
        x = int(parts[0][:-5]) # Remove "throw" suffix and convert to int and +1 Assuming zero-based indexing
        y = int(parts[1][:-8]) # Remove "thcolumn" suffix and convert to int
        
        return x, y

    # Get unique X and Y values to determine matrix size
    def shape(self, df):
        coords = [self.extract_coords_from_name(col) for col in df.columns]
        x_max = max(x for x, y in coords)+1
        y_max = max(y for x, y in coords)+1

        return x_max, y_max

    # Convert each row into a 2D matrix
    def gen_connectome_matrix(self, df):
        connectome_matrices = []

        df.reset_index(drop=True, inplace=True)

        x_max, y_max = self.shape(df)

        for i, row in df.iterrows():
            matrix = np.zeros((x_max+1, y_max))  # Initialize empty matrix
            for col, value in row.items():
                x, y = self.extract_coords_from_name(col)
                matrix[x][y] = value  # Fill in matrix

            connectome_matrices.append(matrix.copy())

            if __debug__:
                if not np.array_equal(connectome_matrices[i],matrix):
                    raise Exception(f"matrix mismatch for sample {i}")
                
                for col, value in row.items():
                    x, y = self.extract_coords_from_name(col)
                    if connectome_matrices[i][x,y] != value:
                        raise Exception(f"matrix mismatch for sample {i} {value} vs {connectome_matrices[i][x,y]}")

        # Convert to NumPy array (shape: [num_samples, x_max, y_max, 1])
        connectome_matrices_with_channel = np.array(connectome_matrices).reshape(len(df), x_max+1, y_max, 1)
        #connectome_matrices_no_channel = np.array(connectome_matrices).reshape(len(df), x_max, y_max)

        return connectome_matrices_with_channel


