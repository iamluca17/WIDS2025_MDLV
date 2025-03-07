import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path

class Parquet_Processor:
    """
    A utility class that converts .csv and .xlsx files in a specified folder (and its subdirectories) to .parquet format,
    and merges the generated parquet files into training and testing datasets.

    The class searches for .csv and .xlsx files, excluding hidden directories (those starting with a dot),
    and converts each file to a corresponding .parquet file in the same directory.
    After conversion, it merges parquet files starting with 'train' and 'test' respectively into two separate DataFrames:
    df_train and df_test.
    *Note, found out if you have a dataframe with categorical columns, when saving as parquet and loading the parquet, the
    category dtype will be lost and the affected columns will revert to their underlying dtype (int, float, object etc).

    Attributes
    ----------
    folder : str or Path
        The root folder containing the files to convert.
    overwrite : bool, optional
        Flag indicating  whether to overwrite existing parquet files. If not provided, methods will use their default value

    Methods
    -------
    convert()
        Converts all .csv and .xlsx files in the folder to .parquet files and returns their paths.
    merge_train_test_parquets()
        Merges parquet files starting with 'train' and 'test' respectively into separate DataFrames, df_train and df_test.
    find_csv_xlsx_files()
        Recursively searches for .csv and .xlsx files in the folder and its subdirectories,
        excluding hidden directories.
    optimize_dataframe()
        Optimizes the data types of a DataFrame to reduce memory usage while maintaining data fidelity.


    Examples
    --------
    >>> processor = Parquet_Processor('/path/to/folder')
    >>> parquet_paths = processor.convert()
    >>> df_train, df_test = processor.merge_train_test_parquets()
    """

    def __init__(self, folder: Union[str, Path], overwrite: Optional[bool] = False):
        """
        Initializes the Folder_Parquet_Processor class.
        """
        self.folder = Path(folder)
        self.overwrite = overwrite

    def convert(self, extensions: Optional[list[str]] = None, overwrite: Optional[bool] = None):
        """
        Converts all files in the folder that match the provided extensions to .parquet format,
        and returns a list of the paths to the generated .parquet files.

        Parameters
        ----------
        extensions : list of str, optional
            The file extensions to filter by.
            If not provided, defaults to ['.csv', '.xlsx'].
        overwrite : bool, optional
            Whether to overwrite existing .parquet files. If not provided, defaults to the
            class-level overwrite attribute.

        Returns
        -------
        list of Path
            A list of Path objects pointing to the generated .parquet files.
        """
        if overwrite is None:
            overwrite = self.overwrite

        if extensions is None:
            extensions = ['.csv', '.xlsx']

       # Dictionary mapping file extensions to pandas read functions
        read_functions = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
            '.json': pd.read_json,
            '.feather': pd.read_feather
        }

        # Validate extensions
        valid_extensions = [ext for ext in extensions if ext in read_functions]
        invalid_extensions = [ext for ext in extensions if ext not in read_functions]

        # If no valid extensions, raise an error
        if not valid_extensions:
            raise ValueError(f"No valid extensions provided. Supported extensions are: {list(read_functions.keys())}. Program will now exit.")

        # If some extensions are invalid, print a warning
        if invalid_extensions:
            print(f"Warning: Unsupported file extensions: {invalid_extensions}. Supported extensions are: {list(read_functions.keys())}. Program will continue with the supported extensions provided: {valid_extensions}")


        raw_files = self.find_files_by_extension(valid_extensions)
        parquet_paths = []

        if not raw_files:
            raise FileNotFoundError(fr"No files found with the following extensions in the folder {self.folder} or its subdirectories: {valid_extensions}")

        for raw_file in raw_files:
            parquet_file = self.folder / raw_file.with_suffix(".parquet")

            # Skip conversion if the .parquet file already exists
            if parquet_file.exists() and not overwrite:
                print(fr"Skipping converting {raw_file}, .parquet file already exists.")
                parquet_paths.append(parquet_file)
                continue

             # Identify the appropriate read function based on file extension
            file_extension = raw_file.suffix.lower()
            read_func = read_functions[file_extension]
            try:
                df = read_func(raw_file)
                optimized_df = self.optimize_dataframe(df)
                optimized_df.to_parquet(parquet_file)
                print(fr"Converted {raw_file} to {parquet_file}")
                parquet_paths.append(parquet_file)
            except Exception as e:
                print(fr"Error converting {raw_file}: {e}")

        return parquet_paths

    def find_files_by_extension(self, extensions: list[str]) -> list[Path]:
        """
        Recursively searches for files in the folder and its subdirectories, excluding hidden directories,
        that match the provided extensions.

        Parameters
        ----------
        extensions : list of str
            The extensions to filter by (e.g., ['.csv', '.xlsx']).

        Returns
        -------
        list of Path
            A list of Path objects pointing to the matching files.
        """
        folder = self.folder
        files = []

        for path in folder.rglob("*"):
            if path.is_dir() and path.name.startswith("."):
                continue
            if path.is_file() and path.suffix in extensions and (path.name.startswith('TRAIN') or path.name.startswith('TEST')):
                files.append(path)

        return files

    def merge_train_test_parquets(self, paths: Optional[list[Path]] = None, overwrite: Optional[bool] = None):
        """
        Merges all .parquet files that start with 'train' into a single DataFrame df_train,
        and all .parquet files starting with 'test' into a separate DataFrame df_test, by the 'participant_id' column.
        If a list of file paths is provided, those paths will be used. Otherwise, the function will search for all
        .parquet files in the folder. Merged DataFrames will be saved to parquet
        files (train_merged.parquet and test_merged.parquet) to avoid redundant merging in future calls.

        Parameters
        ----------
        paths : list of Path, optional
            A list of Paths to parquet files. If None, the method will search for files in the folder.
        overwrite : bool, optional
            Whether to overwrite existing .parquet files. If not provided, defaults to the
            class-level overwrite attribute.

        Returns
        -------
        tuple
            A tuple containing two DataFrames: df_train and df_test.
        """
        if overwrite is None:
            overwrite = self.overwrite

        merged_train_path = self.folder / "train_merged.parquet"
        merged_test_path = self.folder / "test_merged.parquet"

        # If the merged parquet files exist, read them and return
        if merged_train_path.exists() and merged_test_path.exists() and not overwrite:
            print("Merging already done, reading the merged files.")
            df_train = pd.read_parquet(merged_train_path)
            df_test = pd.read_parquet(merged_test_path)
            return df_train, df_test

        # If paths is None, retrieve all .parquet files in the folder
        if paths is None:
            paths = list(self.folder.rglob("*.parquet"))

        train_files = []
        test_files = []

        # Filter the files into train and test based on the filename
        for path in paths:
            if path.name.lower().startswith('train'):
                train_files.append(path)
            elif path.name.lower().startswith('test'):
                test_files.append(path)

        # Merge the train files on 'participant_id'
        if train_files:
            df_train = pd.read_parquet(train_files[0])  # Start with the first train file
            for file in train_files[1:]:
                df_train = pd.merge(df_train, pd.read_parquet(file), on='participant_id', how='inner')  # Merge subsequent files
        else:
            df_train = pd.DataFrame()  # Empty DataFrame if no train files

        # Merge the test files on 'participant_id'
        if test_files:
            df_test = pd.read_parquet(test_files[0])  # Start with the first test file
            for file in test_files[1:]:
                df_test = pd.merge(df_test, pd.read_parquet(file), on='participant_id', how='inner')  # Merge subsequent files
        else:
            df_test = pd.DataFrame()  # Empty DataFrame if no test files

        # Save the merged DataFrames to parquet
        if not df_train.empty:
            df_train.to_parquet(merged_train_path)
            print(fr"Successfully merged train files. Saved as {merged_train_path}.")
        if not df_test.empty:
            df_test.to_parquet(merged_test_path)
            print(fr"Successfully merged test files. Saved as {merged_test_path}.")


        return df_train, df_test

    def optimize_dataframe(self, df: pd.DataFrame, tolerance: float = .01) -> pd.DataFrame:
        """
        Optimizes the data types of a DataFrame to reduce memory usage while maintaining data fidelity.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to optimize.
        tolerance : float, optional
            The maximum allowable percentage of information loss (e.g., 0.01 for 1%). Default is 0.01

        Returns
        -------
        pd.DataFrame
            - The optimized DataFrame.
        """
        original_memory = df.memory_usage(deep=True).sum()

        for col in df.columns:
            col_data = df[col]

            if pd.api.types.is_integer_dtype(col_data):
                df[col] = pd.to_numeric(col_data, downcast='integer')

            elif pd.api.types.is_float_dtype(col_data):
                downcast = col_data.astype("float32")
                max_diff = (col_data - downcast).abs().max()
                allowed_diff = tolerance * col_data.abs().max()
                if max_diff <= allowed_diff:
                    df[col] = downcast

            # Optimize object/string columns with few unique values
            elif pd.api.types.is_object_dtype(col_data):
                unique_count = col_data.nunique()
                total_count = len(col_data)
                if unique_count / total_count < 0.2:  # Arbitrary threshold for low cardinality
                    df[col] = col_data.astype('category')

        optimized_memory = df.memory_usage(deep=True).sum()
        reduction_percentage = ((original_memory - optimized_memory) / original_memory) * 100
        print(f'Finished optimizing the data types, achieved memory reduction of {reduction_percentage}%')

        return df