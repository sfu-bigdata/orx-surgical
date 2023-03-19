# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Load the tables
ben_cols=['DESYNPUF_ID', 'SP_RA_OA', 'BENE_BIRTH_DT', 'BENE_SEX_IDENT_CD']
ip_cols=['DESYNPUF_ID','CLM_FROM_DT','CLM_ID','CLM_DRG_CD','ICD9_DGNS_CD_1','ICD9_DGNS_CD_2',
        'ICD9_DGNS_CD_3','ICD9_DGNS_CD_4','ICD9_DGNS_CD_5','ICD9_DGNS_CD_6','ICD9_DGNS_CD_7',
        'ICD9_DGNS_CD_8','ICD9_DGNS_CD_9','ICD9_DGNS_CD_10','ICD9_PRCDR_CD_1','ICD9_PRCDR_CD_2',
        'ICD9_PRCDR_CD_3','ICD9_PRCDR_CD_4','ICD9_PRCDR_CD_5','ICD9_PRCDR_CD_6']
pde_cols=['DESYNPUF_ID', 'PROD_SRVC_ID']

# rows = 100000

ben = pd.read_csv("data/cms/ben.csv",
                  usecols=ben_cols)
ip = pd.read_csv("data/cms/ip.csv",
                 usecols=ip_cols)
# pde = pd.read_csv("data/cms/pde.csv",
#                   usecols=pde_cols)
dx = pd.read_csv("data/cms/dx.csv")
pcs = pd.read_csv("data/cms/pcs.csv")

col_num = 6


def get_arthritis_patient_data(beneficiaries_df, inpatient_df, start_year, end_year):
  """
  Process data to identify cases of patients with Rheumatoid Arthritis or Osteoarthritis.

  Args:
      beneficiaries_df (pandas.DataFrame): Dataframe containing beneficiaries data
      inpatient_df (pandas.DataFrame): Dataframe containing inpatient data
      start_year (int): Start year to filter data
      end_year (int): End year to filter data

  Returns:
      pandas.DataFrame: Processed dataframe containing cases of patients with Rheumatoid Arthritis or Osteoarthritis.
  """

  # Merge beneficiaries and inpatient dataframes on DESYNPUF_ID
  merged_df = pd.merge(beneficiaries_df, inpatient_df, on='DESYNPUF_ID', how='inner')

  # Extract year from the Year column
  merged_df['CLM_FROM_DT'] = pd.to_numeric(merged_df['CLM_FROM_DT'], errors='coerce')

  # Remove any missing values in CLM_FROM_DT
  merged_df.dropna(subset=['CLM_FROM_DT'], inplace=True)
  merged_df['CLM_FROM_DT'] = merged_df['CLM_FROM_DT'].astype(int)

  # Extract year from the CLM_FROM_DT column
  merged_df['Year'] = pd.to_datetime(merged_df['CLM_FROM_DT'], format='%Y%m%d', errors='coerce').dt.year
  merged_df.insert(4, "Year", merged_df.pop("Year"))  

  # Filter data to only include years between start_year and end_year
  merged_df = merged_df[(merged_df['Year'] >= start_year) & (merged_df['Year'] <= end_year)]
  
  # Filter data to only include cases with Rheumatoid Arthritis/Osteoarthritis diagnosis
  merged_df = merged_df[merged_df['SP_RA_OA'] == 1]
  
  # Remove TJR surgeries in the first two years
  merged_df = merged_df[~(merged_df['Year'].isin([start_year, start_year + 1]) & merged_df['CLM_DRG_CD'].isin(['469', '470']))]

  # Filter data to only include patients fully enrolled in the selected years
  full_enrollment_years = {year for year in range(start_year, end_year + 1)}
  merged_df = merged_df[merged_df.groupby('DESYNPUF_ID')['Year'].transform(lambda x: set(x) >= full_enrollment_years)]
  
  # calculate age in years
  merged_df['Age'] = (pd.to_datetime(merged_df['CLM_FROM_DT'],
                                     format='%Y%m%d') - pd.to_datetime(merged_df['BENE_BIRTH_DT'],
                                                                       format='%Y%m%d')).dt.days / 365
  merged_df.insert(3, 'Age', merged_df.pop('Age'))

  return merged_df


def clean_medical_codes_table(table):
    """
    Cleans up a medical codes table by removing trailing quotes and
    periods from the entries, renaming columns to remove quotes, and
    replacing empty values with "None".

    Args:
    - table: a pandas DataFrame representing the medical codes table

    Returns:
    - A cleaned up version of the medical codes table
    """

    # Clean up each column in the table
    for column in table.columns:
        table[column] = table[column].apply(
            lambda x: x.strip("'").split(".")[0] if isinstance(x, str) else x
        )
        table.rename(columns={column: column.replace("'", "")}, inplace=True)

    # Replace all empty values with "None"
    table.replace(r"^\s*$", "None", regex=True, inplace=True)

    # Clean up first column (ICD code)
    table.iloc[:, 0] = table.iloc[:, 0].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    
    return table


def process_diagnosis_and_procedure_tables(dx_table, pcs_table):
    """
    Processes a diagnosis table and a procedure table by cleaning them up
    using the clean_medical_codes_table function, renaming columns to match,
    and creating dictionaries mapping codes to categories.

    Args:
    - dx_table: a pandas DataFrame representing the diagnosis table
    - pcs_table: a pandas DataFrame representing the procedure table

    Returns:
    - Four variables:
      - dx_dict: a dictionary mapping ICD9 codes to CCS categories for diagnoses
      - dx_categories: an array of unique CCS categories for diagnoses
      - pcs_dict: a dictionary mapping ICD9 codes to CCS categories for procedures
      - pcs_categories: an array of unique CCS categories for procedures
    """

    # Process diagnoses table
    dx_table = clean_medical_codes_table(dx_table)
    dx_table = dx_table.rename(
        columns={"ICD-9-CM CODE": "ICD9", "CCS CATEGORY": "CCS"}
    )
    dx_dict = dict(zip(dx_table["ICD9"], dx_table["CCS"]))
    dx_categories = dx_table["CCS"].unique()

    # Process procedures table
    pcs_table = clean_medical_codes_table(pcs_table)
    pcs_table = pcs_table.rename(
        columns={"ICD-9-CM CODE": "ICD9", "CCS CATEGORY": "CCS"}
    )
    pcs_dict = dict(zip(pcs_table["ICD9"], pcs_table["CCS"]))
    pcs_categories = pcs_table["CCS"].unique()

    return dx_dict, dx_categories, pcs_dict, pcs_categories


def split_data(data, dx_dict, pcs_dict, dx_cols_indices, pcs_cols_indices, target_col):
    """
    Preprocesses the data by creating new columns for dx and pcs codes, merging with original data, dropping 
    unnecessary columns and extracting the target variable.
    
    Args:
    - data: pandas DataFrame, the data to be preprocessed
    - dx_dict: dictionary, a mapping of ICD codes to CCS categories for diagnoses
    - pcs_dict: dictionary, a mapping of ICD codes to CCS categories for procedures
    - dx_cols_indices: tuple, a tuple of the start and end indices for the columns containing the ICD diagnosis codes
    - pcs_cols_indices: tuple, a tuple of the start and end indices for the columns containing the ICD procedure codes
    - target_col: str, the name of the target column
    
    Returns:
    - x_data: pandas DataFrame, the preprocessed claims data with the target variable removed
    - y_data: pandas DataFrame, the target variable for the preprocessed claims data
    """
    
    # Create new columns for dx and pcs codes
    dx_cols = data.iloc[:, dx_cols_indices[0]:dx_cols_indices[1]]
    new_dx_cols = dx_cols.applymap(lambda x: dx_dict.get(x, 0) if x in dx_dict else 0)
    
    pcs_cols = data.iloc[:, pcs_cols_indices[0]:pcs_cols_indices[1]]
    new_pcs_cols = pcs_cols.applymap(lambda x: pcs_dict.get(x, 0) if x in pcs_dict else 0)
  
    # Merge the original data with new dx and pcs columns
    merged_data = pd.concat([data.iloc[:, :dx_cols_indices[0]], new_dx_cols, new_pcs_cols, data.iloc[:, pcs_cols_indices[1]:]], axis=1)
  
    # Drop unnecessary columns and extract the target variable
    x_data = merged_data.drop(["SP_RA_OA"], axis=1)
    x_data[target_col] = x_data[target_col].apply(lambda x: 1 if x in ['469', '470'] else 0)
  
    y_data = x_data[['DESYNPUF_ID', "CLM_ID" , 'Year', target_col]]

    return x_data, y_data


dx_cols = (col_num+3, col_num+13)
pcs_cols = (col_num+13, col_num+19)
target_col = 'CLM_DRG_CD'


def binarize_categorical_columns(data, start_col_index, end_col_index, icd9_to_ccs_dict):
    """
    Binarize the categorical variables in a pandas DataFrame into a one-hot encoded numpy array.
    
    Parameters:
    data (pandas.DataFrame): The input data
    start_col_index (int): The index of the first column to binarize
    end_col_index (int): The index of the last column to binarize (exclusive)
    icd9_to_ccs_dict (dict): The dictionary mapping ICD9 codes to CCS categories
    
    Returns:
    numpy.ndarray: The binarized data in a numpy ndarray
    """
    # Convert the columns specified by start_col_index and end_col_index to integers
    category_columns = data.iloc[:, start_col_index:end_col_index].astype(int)
    # Convert the values in the category_columns dataframe to a numpy array
    category_values = np.array(category_columns.values, dtype=int)
    # Get the unique integer codes from the icd9_to_ccs_dict values
    unique_category_codes = np.array(list(set(icd9_to_ccs_dict.values())), dtype=int)
    # Initialize the output array with zeros and the desired shape
    output = np.zeros((category_values.shape[0], unique_category_codes.shape[0]), dtype=int)

    # For each row in the category_values array
    for i, row in enumerate(category_values):
        # Check which unique category codes are present in the row and update the output
        output[i] = np.isin(unique_category_codes, row).astype(int)

    return output


def create_code_dataframe(raw_data, diagnosis_codes, procedure_codes, target_variable):
    """
    Preprocess the data by binarizing, adding headers, and concatenating
    
    Parameters:
    raw_data (pandas.DataFrame): The raw data
    diagnosis_codes (numpy.ndarray): The binarized data for diagnosis codes
    procedure_codes (numpy.ndarray): The binarized data for procedure codes
    target_variable (pandas.DataFrame): The target variable
    
    Returns:
    pandas.DataFrame: The preprocessed data
    """
    # Add headers to the binarized data
    diagnosis_headers = [f'Diagnosis Code {i}' for i in range(1, diagnosis_codes.shape[1] + 1)]
    procedure_headers = [f'Procedure Code {i}' for i in range(1, procedure_codes.shape[1] + 1)]
    diagnosis_df = pd.DataFrame(diagnosis_codes, columns=diagnosis_headers)
    procedure_df = pd.DataFrame(procedure_codes, columns=procedure_headers)
    
    # Reset index for target variable
    target = target_variable['CLM_DRG_CD'].reset_index(drop=True).astype(int)
    
    # Concatenate the DataFrames
    preprocessed_data = pd.concat([raw_data.iloc[:, :col_num+1].reset_index(drop=True), diagnosis_df, procedure_df, target], axis=1)
    
    return preprocessed_data


def aggregate_occurrence_vector_encoding(data, start_col_index):
    """
    Prepares the input data for a machine learning model by grouping the data by DESYNPUF_ID and Year,
    aggregating the maximum value of each code column, and flattening the code columns in groups of 3 rows.
    
    Parameters:
    data (pandas.DataFrame): The input data
    start_col_index (int): The index of the first code column
    
    Returns:
    tuple: A tuple containing the x_input and y_input for the model
    """
    # Determine the end column index of the code columns
    end_col_index = data.shape[1]
    # Get the names of all code columns
    code_columns = list(data.columns[start_col_index:end_col_index])
    code_columns[:0] = ['Age', 'BENE_SEX_IDENT_CD']
    # Group the data by DESYNPUF_ID and Year, and aggregate the maximum value of each code column
    data_grouped = data.groupby(['DESYNPUF_ID', 'Year'], as_index=False)[code_columns].agg('max')
    # Get the data for the target variable, CLM_DRG_CD
    data_target = data_grouped[['DESYNPUF_ID', 'Year', 'CLM_DRG_CD']]
    # Prepare the x_input by flattening the code columns in groups of 3 rows
    rows_per_sample = 3
    flattened_code_columns = np.concatenate([data_grouped.iloc[i:i+rows_per_sample, 2:end_col_index].values.flatten() 
                                              for i in range(0, len(data_grouped), rows_per_sample)])
    print(flattened_code_columns.shape)
    x_input = flattened_code_columns.reshape(-1, len(code_columns) * rows_per_sample)
    # Prepare the y_input by selecting the values of CLM_DRG_CD for Year equal to 2010
    y_input = data_target.loc[data_target['Year'] == '2010', 'CLM_DRG_CD'].values
    
    return x_input, y_input


def multi_hot_encoding(rows, data, start_col):
    """
    Encode the data in a multi-hot format for input into a model
    
    Parameters:
    data (pandas.DataFrame): The input data
    start_col (int): The start column index of the Code columns
    
    Returns:
    tuple: A tuple containing the x_input and y_input for the model
    """

    # sort the dataframe by the "CLM_DRG_CD" column in descending order
    data = data.sort_values('CLM_DRG_CD', ascending=False)
    data = data.iloc[:rows]
    
    # shuffle the entire dataframe randomly
    data = data.sample(frac=1).reset_index(drop=True)

    # Convert the CLM_FROM_DT column to datetime format and extract the day of year
    data['CLM_FROM_DT'] = pd.to_datetime(data['CLM_FROM_DT'], format='%Y%m%d')
    data['DayOfYear'] = data['CLM_FROM_DT'].dt.dayofyear
    data.drop('CLM_FROM_DT', axis=1, inplace=True)
    data.insert(4, 'DayOfYear', data.pop('DayOfYear'))
    
    # Determine the end column index of the Code columns
    end_col = data.shape[1]
    
    # Prepare the x_input by encoding the Code columns in a multi-hot format
    matrix_shape = data.iloc[:, start_col:-1].shape
    x = np.zeros((matrix_shape[0],366,matrix_shape[-1]))

    for record in range(data.shape[0]):
      try:
          x[record][(data['DayOfYear'].iloc[record])-1] = data.iloc[record, start_col:-1].values
      except Exception as e:
          print(f"Error occurred at record {record}: {e}")

    
    # Prepare the y by selecting the values of CLM_DRG_CD for Year equal to 2010
    y = data['CLM_DRG_CD'].values
    
    return np.array(x.astype("float32")), np.array(y.astype("float32")), np.array(data[['Age', 'BENE_SEX_IDENT_CD']].values).astype('float32')


# This function takes several arguments, including beneficiary data, inpatient data, diagnosis and procedure tables, and start and end years
def get_aov(beneficiaries_df=ben, inpatient_df=ip, dx = dx, pcs= pcs, start_year=2008, end_year=2010, random_state = 42):
    # First, it gets arthritis patient data using the beneficiary and inpatient data for the specified time period
    data = get_arthritis_patient_data(beneficiaries_df=ben, inpatient_df=ip, start_year=2008, end_year=2010)
    # It then processes the diagnosis and procedure tables to create dictionaries of codes
    dx_dict, dx_codes, pcs_dict, pcs_codes = process_diagnosis_and_procedure_tables(dx, pcs)
    # The data is split into input data (x_data) and target data (y_data) based on the diagnosis and procedure codes
    x_data, y_data = split_data(data, dx_dict, pcs_dict, dx_cols, pcs_cols, target_col)
    # The categorical columns for diagnosis and procedure codes are binarized
    dxb = binarize_categorical_columns(data = x_data, start_col_index = col_num+3, end_col_index = col_num+13, icd9_to_ccs_dict = dx_dict)
    pcsb = binarize_categorical_columns(data = x_data, start_col_index = col_num+13, end_col_index = x_data.shape[1], icd9_to_ccs_dict = pcs_dict)
    # The binarized data is combined with the original data to create a processed dataframe
    processed_data = create_code_dataframe(x_data, dxb, pcsb, y_data)
    # The data is multi-hot encoded and split into training and testing sets
    input_data , target_data = aggregate_occurrence_vector_encoding(data = processed_data, start_col_index = col_num-4)
    train_input, test_input, train_target, test_target = train_test_split(input_data,
                    target_data, test_size=0.2, stratify=target_data, random_state= random_state)
    # The training and testing data is returned
    return train_input.astype('float32'), test_input.astype('float32'), train_target.astype('float32'), test_target.astype('float32')


# This function takes several arguments, including beneficiary data, inpatient data, diagnosis and procedure tables, and start and end years
def get_mhe(beneficiaries_df=ben, inpatient_df=ip, dx = dx, pcs= pcs, start_year=2008, end_year=2010, rows = 10000, random_state = 42):
    # First, it gets arthritis patient data using the beneficiary and inpatient data for the specified time period
    data = get_arthritis_patient_data(beneficiaries_df=ben, inpatient_df=ip, start_year=2008, end_year=2010)
    # It then processes the diagnosis and procedure tables to create dictionaries of codes
    dx_dict, dx_codes, pcs_dict, pcs_codes = process_diagnosis_and_procedure_tables(dx, pcs)
    # The data is split into input data (x_data) and target data (y_data) based on the diagnosis and procedure codes
    x_data, y_data = split_data(data, dx_dict, pcs_dict, dx_cols, pcs_cols, target_col)
    # The categorical columns for diagnosis and procedure codes are binarized
    dxb = binarize_categorical_columns(data = x_data, start_col_index = col_num+3, end_col_index = col_num+13, icd9_to_ccs_dict = dx_dict)
    pcsb = binarize_categorical_columns(data = x_data, start_col_index = col_num+13, end_col_index = x_data.shape[1], icd9_to_ccs_dict = pcs_dict)
    # The binarized data is combined with the original data to create a processed dataframe
    processed_data = create_code_dataframe(x_data, dxb, pcsb, y_data)
    # The data is multi-hot encoded and split into training and testing sets
    input_data , target_data, dv = multi_hot_encoding(rows, data = processed_data, start_col = col_num+1)
    train_input, test_input, train_target, test_target = train_test_split(input_data,
                    target_data, test_size=0.2, stratify=target_data, random_state= random_state)
    train_dv, test_dv, _, __ = train_test_split(dv,
                    target_data, test_size=0.2, stratify=target_data, random_state= random_state)
    # The training and testing data is returned
    return train_input.astype('float32'), test_input.astype('float32'), train_target.astype('float32'), test_target.astype('float32'), train_dv.astype('float32'), test_dv.astype('float32')