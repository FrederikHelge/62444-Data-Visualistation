from scipy import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import io


def create_scatterplot(df, x_col, y_col, title, xlabel, ylabel):
    """
    This function creates a scatter plot with a linear regression line from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_col (str): The column in the DataFrame to use for the x-axis.
    y_col (str): The column in the DataFrame to use for the y-axis.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """

    # Create the plot
    plt.figure(figsize=(7, 7))
    sns.regplot(x=df[x_col], y=df[y_col], scatter_kws={"alpha": 0.3})

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()


def get_a_random_chunk_property(data):
    """
    This function only serves an example of fetching some of the properties
    from the data.
    Indeed, all the content in "data" may be useful for your project!
    """

    chunk_index = np.random.choice(len(data))

    date_list = list(data[chunk_index]["near_earth_objects"].keys())

    date = np.random.choice(date_list)

    objects_data = data[chunk_index]["near_earth_objects"][date]

    object_index = np.random.choice(len(objects_data))

    object = objects_data[object_index]

    properties = list(object.keys())
    property = np.random.choice(properties)

    print("date:", date)
    print("NEO name:", object["name"])
    print(f"{property}:", object[property])


# We load the data with this url function, but optimized it in case of SSL issues, which we ran into underway.
def load_data_from_google_drive(url):
    url_processed = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    response = requests.get(url_processed, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))
    return df



# Add time features to datasets for temporal analysis and forecasting.
def add_time_features(df, pickup_col):
    df = df.copy()
    df['pickup_dt']   = pd.to_datetime(df[pickup_col])
    df['hour']        = df['pickup_dt'].dt.hour
    df['day_of_week'] = df['pickup_dt'].dt.day_name()
    df['month']       = df['pickup_dt'].dt.month_name()
    df['date']        = df['pickup_dt'].dt.date
    return df


# Build daily series with continuous calendar (fills missing dates with 0 rides)
def get_daily_rides(df, pickup_col, start='2022-01-01', end='2023-12-31'):
    tmp = df.copy()
    tmp['ds'] = pd.to_datetime(tmp[pickup_col]).dt.tz_localize(None).dt.floor('D')
    tmp = tmp[(tmp['ds'] >= start) & (tmp['ds'] <= end)]

    daily = tmp.groupby('ds').size().rename('y').to_frame()
    full_idx = pd.date_range(start=start, end=end, freq='D')
    daily = daily.reindex(full_idx, fill_value=0).reset_index()
    daily.columns = ['ds', 'y']

    # Prophet expects numeric target
    daily['y'] = daily['y'].astype(float)
    return daily


# Dark themed function to give a consistent look to a couple of plots
def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333355')
    ax.spines['bottom'].set_color('#333355')
    ax.tick_params(colors='#ccccdd')
    ax.xaxis.label.set_color('#ccccdd')
    ax.yaxis.label.set_color('#ccccdd')
    ax.title.set_color('white')
    
    
# We clean our data multiple times in the notebook, so we put the cleaning steps in a function to avoid code repetition.
def clean_taxi_data(df):
    # Downcast to reduce memory pressure on large datasets.
    for col in ['trip_distance', 'fare_amount', 'passenger_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

    mask = (
        (df['trip_distance'] > 0) &
        (df['trip_distance'] <= 50) &
        (df['fare_amount'] > 0) &
        (df['fare_amount'] <= 100) &
        (df['passenger_count'] >= 1) &
        (df['passenger_count'] <= 6)
    )

    cleaned = df.loc[mask]
    print(f"Rows kept: {len(cleaned):,} / {len(df):,} ({len(cleaned)/len(df):.1%})")
    return cleaned.reset_index(drop=True)