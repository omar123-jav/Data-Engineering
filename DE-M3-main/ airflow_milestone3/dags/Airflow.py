from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from Dashboard import dashboard
###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import datetime
from sqlalchemy import create_engine


###

#######################################


def percent_missing(df):
    percent_nan = 100 * df.isna().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values(ascending=False)
    return percent_nan


def contain_missing_value(x, feature_name):
    no_feature = "no " + feature_name.split("_")[0].split(" ")[0]
    missing_values = ["missing", "unknown", "none", no_feature]
    for st in missing_values:
        if st in str(x).lower():
            return True
    return False


def detect_outlier_using_Zscore(feature):
    """
    For a given feature using Z-Score method, records considered as an outlier is returned

    Parameters
    __________
    feature: feature column (pd.Series)

    Returns
    _______
    outliers: dataframe containing records considered as oultiers (pd.DataFrame)
    """
    z = np.abs(stats.zscore(feature))
    outlier_entries = z > 3
    outliers = feature[outlier_entries]

    return outliers


def detect_outlier_using_IQR(feature):
    """
    For a given feature using internal quartile range method, records considered as an outlier is returned

    Parameters
    __________
    feature: feature column (pd.Series)

    Returns
    _______
    outliers: dataframe containing records considered as oultiers (pd.DataFrame)
    """
    Q1 = feature.quantile(0.25)
    Q3 = feature.quantile(0.75)
    IQR = Q3 - Q1
    cut_off = IQR * 1.5
    lower = Q1 - cut_off
    upper = Q3 + cut_off
    outliers = feature[(feature < lower) | (feature > upper)]
    return outliers


def group_low_frequencies(feature, thr=5):
    """
    For a given feature, reconstruct feature with least 5% frequent observations grouped together

    Parameters
    __________
    feature (pd.Series): feature from the data
    thr (float): threshold

    Returns
    _______
    observations_less_than_threshold (list): observations bellow thr
    label (str): label to replace returned observations (exists/other)
    """
    threshold = thr / 100
    freq = (feature.value_counts().sort_values()) / len(feature)
    observations_less_than_threshold = list(freq.cumsum().loc[lambda x: x <= threshold].index)
    if len(observations_less_than_threshold) != (feature.nunique() - 1):
        label = "others"
    else:

        label = "exists"
    return observations_less_than_threshold, label


def outlier_report(feature, method='z'):
    """
    Prints number of outliers, outliers values ,count of each value & percentage of the outliers

    Parameters
    __________
    feature: feature column (pd.Series)
    method: method to detect outliers either z or iqr (str) (default = z)

    Returns
    _______
    None
    """
    if method == 'z':
        outliers = detect_outlier_using_Zscore(feature)
        print("Method: Z-score")
    elif method == 'iqr':
        outliers = detect_outlier_using_IQR(feature)
        print("Method: IQR")
    count_of_outliers = len(outliers)
    column_size = len(feature)

    percentage_of_outliers = count_of_outliers / column_size * 100
    ouliers_distinct_values = outliers.unique()

    value_counts_df = pd.DataFrame(outliers.value_counts())
    value_counts_df.columns = ["count"]

    value_counts_df["percentage"] = value_counts_df["count"].apply(
        lambda x: int(x / column_size * 100 * 100000) / 100000)

    print("Count Of Outliers: ", count_of_outliers, "from", column_size)
    print("Outliers Unique Values", np.sort(ouliers_distinct_values))
    print("Percentage Of Outliers: ", int(percentage_of_outliers * 1000) / 1000, "%")

    return value_counts_df


def outliers_bellow_th(report, th=0.5):
    """
    outputs observation values to be dropped

    Parameter
    _________
    report (pd.DataFrame): output from report function
    th (float): threshold (default = 0.5)

    Return
    ______
    res (list): list of observations
    """

    cumulative_sum = report.sort_index(ascending=False)["percentage"].cumsum()
    res = cumulative_sum[cumulative_sum <= th].index.to_list()
    return res


def one_hot_encode(feature):
    """
    Returns the result of applying one hot encoding to a single feature

    Paramaters
    __________
    feature: feature to be converted (pd.series)
    colum_labels: new column labels (dict)

    Returns
    ________
    encoded_df: (pd.DataFrame)
    """
    encoded_df = pd.get_dummies(feature, prefix=feature.name)
    return encoded_df


def use_label_encoding(df, feature_name):
    """
    Label Encode Specific Feature

    Parameters
    __________
    feature_name (str): feature name

    Returns
    _______
    None
    """
    label_encode = preprocessing.LabelEncoder()
    encoded_feature = label_encode.fit_transform(df[feature_name])
    df[feature_name] = encoded_feature
    return df


def encode_features_based_on_threshold(df, col_to_encode, threshold):
    """
    determines the way of encoding a column based on its cardinality

    Parameters
    _________
    col_to_encode (list): list of column names
    threshold (floar): threshold

    Return
    ______
    None

    """
    df_temp = df
    for i in col_to_encode:
        if (len(df_temp[i].unique()) > threshold or len(df_temp[i].unique()) == 2):
            df_temp = use_label_encoding(df_temp, i)
        else:
            df_temp = pd.concat([df_temp.drop(i, axis=1), one_hot_encode(df_temp[i])], axis=1)
    return df_temp


def label_encoder(feature, values_map):
    """
    Returns the result of applying one hot encoding to a single feature

    Paramaters
    __________
    feature: feature to be converted (pd.series)
    values_map: Values labels (dict)

    Returns
    ________
    feature_encoded: (pd.Series)
    """
    feature_encoded = feature.map(values_map)
    # feature_unique_values = feature.unique()
    return feature_encoded


def max_columns(df, max_columns=19):
    """
    for a given dataframe, compute suitable threshold according to
    features cardinality to limit the number of created columns

    Parameters
    __________
    df (pd.DataFrame): dataframe
    max_columns (int): maximum number of columns (default = 10)

    Return
    ______
    th (int): threshold
    """

    th = 0
    df_obj_nunique = df.select_dtypes(['object']).nunique().sort_values()
    for n in df_obj_nunique:
        if n > 2:
            max_columns -= n
            if max_columns < 0:
                if th == n:
                    th -= 1
                break
            th = n
    return th


def data_transformation(filename):
    df = pd.read_csv(filename)
    df_cleaned = clean(df).set_index("accident_index")
    df_discretized = discretize(df_cleaned)
    df_encoded = encoding(df_discretized)
    df_final = addweekendandweather(df_discretized, df_encoded)
    df_final.to_csv("/opt/airflow/data/Processed_2011_Accidents_UK.csv")

def clean(df):
    df_redundant_removed = df.drop(["location_easting_osgr", "location_northing_osgr"], axis=1).copy()
    df_structured = df_redundant_removed.drop(["accident_reference"], axis=1).copy()
    df_structured.lsoa_of_accident_location = df_structured.lsoa_of_accident_location.astype(str)

    # df_structured["local_authority_district"] = df_structured["local_authority_district"].str.replace(r',', '').replace(
    #     r'\.', '')
    # df_structured["local_authority_highway"] = df_structured["local_authority_highway"].str.replace(r',', '').replace(
    #     r'\.', '')
    # df_structured["local_authority_ons_district"] = df_structured["local_authority_ons_district"].str.replace(r',',
    #                                                                                                           '').replace(
    #     r'\.', '')
    months = []
    days = []
    # years=[]
    col = df_structured['date']
    for d in col:
        date = d.split('/')
        months.append(date[1])
        # years.append(date[2])
        days.append(date[0])
    df_structured['Month'] = months
    df_structured['Day'] = days

    df_structured['Month'] = df_structured['Month'].astype(int)
    df_structured['Day'] = df_structured['Day'].astype(int)
    df_structured.drop(columns='date', axis=1, inplace=True)
    hours = []
    minutes = []
    col = df_structured['time']
    for t in col:
        time = t.split(':')
        hours.append(time[0])
        # years.append(date[2])
        minutes.append(time[1])
    df_structured['Hour'] = hours
    df_structured['Minute'] = minutes

    df_structured['Hour'] = df_structured['Hour'].astype(int)
    df_structured['Minute'] = df_structured['Minute'].astype(int)
    df_structured.drop(columns='time', axis=1, inplace=True)
    equivalent_to_nan = ["-1", -1]
    features_to_not_replace_nan = ["longitude", "latitude"]
    df_accidents_missing_nan = df_structured.drop(features_to_not_replace_nan, axis=1).replace(
        equivalent_to_nan, np.nan)
    df_accidents_missing_nan[features_to_not_replace_nan] = df[features_to_not_replace_nan]
    for feature in df_accidents_missing_nan.columns:
        df_accidents_missing_nan[feature] = df_accidents_missing_nan[feature].apply(
            lambda x: np.nan if contain_missing_value(x, feature) else x)

    df_accidents_missing_nan = df_accidents_missing_nan.applymap(
        lambda x: np.nan if "missing" in str(x).lower() or "unknown" in str(x).lower() else x)
    percent_nan = percent_missing(df_accidents_missing_nan)
    drop_columns = percent_nan[percent_nan > 85].index
    df_accidents_drop = df_accidents_missing_nan.drop(columns=drop_columns)
    df_accidents_drop_fill = df_accidents_drop.copy()

    df_accidents_drop_fill['road_type'] = df_accidents_drop.groupby('local_authority_district')['road_type'].apply(
        lambda val: val.fillna(val.mode().iloc[0]))

    df_accidents_drop_fill['weather_conditions'] = \
        df_accidents_drop_fill.groupby(['Month', 'local_authority_district'])['weather_conditions'].apply(
            lambda val: val.fillna(val.mode().iloc[0]))
    df_accidents_drop_complete = df_accidents_drop_fill.copy()
    df_accidents_drop_complete.replace(
        'first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ', '0',
        inplace=True)
    df_accidents_drop_complete['road_surface_conditions'] = df_accidents_drop_complete.groupby(['weather_conditions'])[
        'road_surface_conditions'].apply(lambda val: val.fillna(val.mode().iloc[0]))
    df_accidents_drop_complete['light_conditions'] = df_accidents_drop_complete.groupby(['Hour'])[
        'light_conditions'].apply(lambda val: val.fillna(val.mode().iloc[0]))
    df_accidents_drop_complete['trunk_road_flag'] = df_accidents_drop_complete.groupby(['first_road_class'])[
        'trunk_road_flag'].apply(lambda val: val.fillna(val.mode().iloc[0]))
    df_accidents_drop_complete["local_authority_highway"].fillna(df_accidents_drop_complete["local_authority_district"],
                                                                 inplace=True)
    percent_nan = percent_missing(df_accidents_drop_complete)
    drop_rows = percent_nan[percent_nan < 1].index
    df_accidents_drop_complete = df_accidents_drop_complete.dropna(subset=drop_rows, axis=0)
    for col in percent_nan.index:
        if df_accidents_drop_complete[col].dtype == 'int' or df_accidents_drop_complete[col].dtype == 'float':
            df_accidents_drop_complete[col].fillna(-1, inplace=True)
        else:
            values = df_accidents_drop_complete[col].unique()
            element = values[0]
            firstchar = element[0]
            if firstchar >= '0' and firstchar <= '9':
                df_accidents_drop_complete[col].fillna('-1', inplace=True)
            else:
                df_accidents_drop_complete[col].fillna('None', inplace=True)
    df_accidents_duplicates = df_accidents_drop_complete.copy()

    df_accidents_duplicates.drop_duplicates(inplace=True)
    for col in df_accidents_duplicates:
        if df_accidents_duplicates[col].dtype == 'object':
            condition = pd.to_numeric(df_accidents_duplicates[col], errors='coerce').notnull().all()
            if condition:
                flag = True
                for n in df_accidents_duplicates[col]:
                    condition1 = n.isdigit()
                    if condition1 == False:
                        flag = False
                        break
                if flag == True:
                    df_accidents_duplicates[col] = df_accidents_duplicates[col].astype(int)

                else:
                    df_accidents_duplicates[col] = df_accidents_duplicates[col].astype(float)
    df_accidents_outliers = df_accidents_duplicates.copy()
    vehicles_report = outlier_report(df_accidents_outliers.number_of_vehicles)
    casualties_report = outlier_report(df_accidents_outliers.number_of_casualties, method='iqr')
    indices_to_be_dropped = set()

    vehicles_outliers = outliers_bellow_th(vehicles_report)

    indices_to_be_dropped |= set(
        df_accidents_outliers[df_accidents_outliers["number_of_vehicles"].isin(vehicles_outliers)].index.tolist())

    casualities_outliers = outliers_bellow_th(casualties_report)

    indices_to_be_dropped |= set(
        df_accidents_outliers[df_accidents_outliers["number_of_casualties"].isin(casualities_outliers)].index.tolist())
    df_accidents_outliers = df_accidents_outliers.drop(list(indices_to_be_dropped))
    # df_accidents_outliers.to_csv("/opt/airflow/data/accidents_clean.csv")
    # df_accidents_outliers.to_csv("accidents.csv")
    return df_accidents_outliers


def discretize(df):
    df_accidents_discretization = df.copy()

    week_number = df_accidents_discretization.apply(
        lambda row: datetime.datetime(row["accident_year"], row["Month"], row["Day"]).isocalendar()[1], axis=1)
    df_accidents_discretization["week_number"] = week_number
    features_to_group_low_freq = ["pedestrian_crossing_human_control", "special_conditions_at_site",
                                  "carriageway_hazards"]
    for feature in features_to_group_low_freq:
        if feature in df_accidents_discretization.columns:
            observations, imputation_label = group_low_frequencies(df_accidents_discretization[feature])
            df_accidents_discretization.loc[df_accidents_discretization[feature].isin(observations), feature] = \
                imputation_label

    # df_accidents_discretization.to_csv("/opt/airflow/data/accidents_discretized.csv",index=False)
    # df_accidents_discretization.to_csv("accidents_discretized.csv",index=False)
    return df_accidents_discretization


def encoding(df):
    # df_to_be_encoded = pd.read_csv(filename,index_col=0)
    df_to_be_encoded = df.copy()
    print(df_to_be_encoded.head())

    df_to_be_encoded['accident_severity'] = label_encoder(df_to_be_encoded['accident_severity'],
                                                          {'Slight': 1, 'Serious': 2, 'Fatal': 3})
    col_to_encode = list(df_to_be_encoded.select_dtypes(['object']).columns.copy())
    print(df_to_be_encoded.head())
    if "lsoa_of_accident_location" in col_to_encode:
        col_to_encode.remove("lsoa_of_accident_location")

    df_encoded = encode_features_based_on_threshold(df_to_be_encoded.copy(), col_to_encode,
                                                    max_columns(df_to_be_encoded))
    print(df_encoded.head())
    # df_encoded.to_csv("accidents_encoded.csv")
    # df_encoded.to_csv("/opt/airflow/data/accidents_encoded.csv")
    return df_encoded


def addweekendandweather(df_accidents_discretization, df_encoded):
    # df_accidents_discretization=pd.read_csv(filename1,index_col=0)
    # df_encoded=pd.read_csv(filename2,index_col=0)
    df_encode = df_encoded.assign(weekend=lambda x: (df_accidents_discretization['day_of_week'] == 'Saturday') | (
            df_accidents_discretization['day_of_week'] == 'Sunday'))
    df_encode['weekend'] = df_encode['weekend'].map({True: 1, False: 0})
    special_weather = lambda x: \
        1 if "rain" in x.lower() else (2 if "snow" in x.lower() else 0)
    df_encode['special_weather'] = df_accidents_discretization['weather_conditions'].apply(special_weather)
    return df_encode
    # df_encode.to_csv("accidents_final.csv")


def lookup(filename1, filename2):
    df_raw = pd.read_csv(filename1, index_col=0)
    df_final = pd.read_csv(filename2, index_col=0)
    df_accidents = df_raw.copy()
    # df_final.set_index("accident_index",inplace=True)
    # df_accidents.set_index("accident_index", inplace=True)
    lookup_v2 = pd.DataFrame(columns=["Feature", "Original Value", "Imputed/Encoded"])

    indices_available = df_final.index.tolist()
    # print(indices_available)
    df_raw_subset = df_accidents.loc[indices_available]
    for col in df_raw.columns:
        if col in df_final.columns:
            # feature_name * number_of_rows (series)
            feature_col = pd.Series(col, index=range(len(indices_available)))
            if (not df_final[col].equals(df_raw_subset[col])):
                # columns value changed
                df_compare = pd.DataFrame(
                    pd.concat([df_final[col], df_raw_subset[col]], axis=1)).drop_duplicates().reset_index(drop=True)

                df_compare["Feature"] = col
                df_compare.columns = ["Imputed/Encoded", "Original Value", "Feature"]

                df_compare = df_compare[df_compare["Imputed/Encoded"] != df_compare["Original Value"]]
                df_compare = df_compare[
                    df_compare["Imputed/Encoded"].astype(str) != df_compare["Original Value"].astype(str)]

                lookup_v2 = pd.concat([lookup_v2, df_compare], axis=0)

    for org in df_final[df_final['weekend'] == 1]['day_of_week'].unique():
        lookup_v2.loc[len(lookup_v2)] = ['weekend', org, 1]

    for org in df_final[df_final['weekend'] == 0]['day_of_week'].unique():
        lookup_v2.loc[len(lookup_v2)] = ['weekend', org, 0]
    for org in df_final[df_final['special_weather'] == 0]["weather_conditions"].unique():
        lookup_v2.loc[len(lookup_v2)] = ['weekspecial_weatherend', org, 0]
    for org in df_final[df_final['special_weather'] == 1]["weather_conditions"].unique():
        lookup_v2.loc[len(lookup_v2)] = ['weekspecial_weatherend', org, 1]
    for org in df_final[df_final['special_weather'] == 2]["weather_conditions"].unique():
        lookup_v2.loc[len(lookup_v2)] = ['weekspecial_weatherend', org, 2]
    lookup_v2 = lookup_v2.sort_values(["Feature", "Imputed/Encoded"]).reset_index(drop=True)
    final_lookup_v2 = pd.DataFrame(
        lookup_v2.groupby(["Feature", "Imputed/Encoded"], as_index=False)["Original Value"].agg(list))
    # final_lookup_v2.to_csv('Lookup_Table.csv', index=False)
    final_lookup_v2.to_csv('/opt/airflow/data/Lookup_Table.csv', index=False)


def MS2(filename1, filename2):
    df_ms1 = pd.read_csv(filename1, index_col=0)
    df_2011_gender = pd.read_csv(filename2)
    df_2011_gender["males"] = np.where(df_2011_gender["sex_of_driver"] == 1, 1, 0)

    df_2011_gender["females"] = np.where(df_2011_gender["sex_of_driver"] == 2, 1, 0)

    df_2011_gender["unknown_gender"] = np.where(df_2011_gender["sex_of_driver"] == 3, 1, 0)
    df_2011_gender_grouped = df_2011_gender.groupby(df_2011_gender["accident_index"])[
        "males", "females", "unknown_gender"].sum()
    df_merged = pd.merge(df_ms1, df_2011_gender_grouped, left_index=True, right_index=True, how="left")
    # df_merged.to_csv('df_ms2.csv')
    df_merged.to_csv("/opt/airflow/data/df_ms2.csv")


def load_to_postgres(filename1, filename2):
    df1 = pd.read_csv(filename1, index_col=0)
    df2 = pd.read_csv(filename2, index_col=0)
    engine = create_engine('postgresql://root:root@pgdatabase5:5432/accidents_pipeline')
    if (engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df1.to_sql(name='UK_Accidents_2011', con=engine, if_exists='replace')
    df2.to_sql(name='lookup_table', con=engine, if_exists='replace')


# MS2('accidents_final.csv','df_2011_gender.csv')


#################################################################################

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}
dag = DAG(
    'accidents_pipeline',
    default_args=default_args,
    description=' pipeline',
)
with DAG(
        dag_id='accidents_pipeline',
        schedule_interval='@once',
        default_args=default_args,
        tags=['accidents-pipeline'],
) as dag:
    data_transformation_task = PythonOperator(
        task_id='MS1_Data_Cleaning_and_Transformation',
        python_callable=data_transformation,
        op_kwargs={
            "filename": '/opt/airflow/data/2011_Accidents_UK.csv'
        },
    )
    lookup_setup_task = PythonOperator(
        task_id='Lookup_Table',
        python_callable=lookup,
        op_kwargs={
            "filename1": "/opt/airflow/data/2011_Accidents_UK.csv",
            "filename2": "/opt/airflow/data/Processed_2011_Accidents_UK.csv"
        },
    )
    MS2_final_task = PythonOperator(
        task_id='MS2_Gender',
        python_callable=MS2,
        op_kwargs={
            "filename1": "/opt/airflow/data/Processed_2011_Accidents_UK.csv",
            "filename2": "/opt/airflow/data/df_2011_gender.csv"
        },
    )
    load_to_postgres_task = PythonOperator(
        task_id='Postgres_Populating_Database',
        python_callable=load_to_postgres,
        op_kwargs={
            "filename1": "/opt/airflow/data/df_ms2.csv",
            "filename2": "/opt/airflow/data/Lookup_Table.csv"
        },
    )
    dashboard_task = PythonOperator(
        task_id='Dashboard',
        python_callable=dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/df_ms2.csv",
            "filename2": "/opt/airflow/data/Lookup_Table.csv"
        },
    )

    data_transformation_task >> lookup_setup_task >> MS2_final_task >> load_to_postgres_task >> dashboard_task
