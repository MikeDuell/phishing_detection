import html
import re
from unidecode import unidecode
import pandas as pd

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
NON_ALPHA_RE = re.compile(r"[^a-z\s']")  # allow apostrophes for contractions
WHITESPACE_RE = re.compile(r'\s+')


def standardise_col_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)


def drop_columns(data, columns):
    data.drop(columns, axis=1, inplace=True)


def fill_in_missing_days(df):
    df['sending_date'] = pd.to_datetime(df['sending_date'], errors='coerce')
    # Fill missing 'day' based on 'sending_date'
    df['day'] = df['day'].fillna(df['sending_date'].dt.strftime('%a'))
    # Log how many were filled
    filled_count = df['day'].notna().sum()
    print(f"Filled weekday names for {filled_count} rows.")
    return df


def handle_missing_subject_content(df):
    # Fill missing subject with placeholder
    df['email_subject'] = df['email_subject'].fillna('no_subject')
    # Drop rows with missing content
    df = df.dropna(subset=['email_content'])
    return df


def clean_text(text, keep_newlines=False, keep_contractions=True):
    if text is None:
        return ""
    # Coerce and basic normalization
    s = str(text)
    s = html.unescape(s)  # convert HTML entities
    s = s.replace('↵', '\n')  # convert visible newline glyph to real newline
    s = unidecode(s)  # normalize accents, fancy quotes -> ascii
    s = s.lower()
    # replace URLs and emails with placeholders
    s = URL_RE.sub(' link ', s)
    s = EMAIL_RE.sub(' email ', s)
    # optionally keep or remove newline characters
    if not keep_newlines:
        s = s.replace('\n', ' ')
    # remove characters not letters/space (and keep apostrophes if desired)
    if keep_contractions:
        s = NON_ALPHA_RE.sub(' ', s)
    else:
        s = re.sub(r'[^a-z\s]', ' ', s)
    # collapse whitespace and trim
    s = WHITESPACE_RE.sub(' ', s).strip()
    return s


def split_date(data, date_col='date', verbose=True):
    """
    Parses RFC 2822-style email timestamps into structured components.

    Parameters:
    - data (pd.DataFrame): Input DataFrame
    - date_col (str): Column containing raw date strings
    - fallback (str): 'mode', 'constant', or 'mean' for filling malformed entries
    - verbose (bool): If True, prints diagnostics
    - return_subset (bool): If True, returns only relevant columns

    Returns:
    - pd.DataFrame: Cleaned DataFrame with ['day', 'time', 'zone', 'clean_date']
    """
    pattern = (
        r'^(?P<day>\w{3}),\s+'
        r'(?P<d>\d{2})\s+'
        r'(?P<mon>\w{3})\s+'
        r'(?P<y>\d{4})\s+'
        r'(?P<time>\d{2}:\d{2}:\d{2})\s+'
        r'(?P<zone>[+-]\d{4}(?:\s*\((?:UTC|GMT)\))?)'
        r'(?:\s*\(.*\))?$'
    )

    data[date_col] = data[date_col].astype(str)
    ex = data[date_col].str.extract(pattern)

    # Attach extracted parts
    data["day"] = ex["day"]
    data["time"] = ex["time"]
    data["zone"] = pd.to_numeric(ex["zone"], errors="coerce").astype("Int64")

    # Defensive conversion
    data["clean_date"] = pd.to_datetime(
        ex["d"] + " " + ex["mon"] + " " + ex["y"],
        format="%d %b %Y",
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Diagnostics
    malformed = data[data["clean_date"].isna()]
    if verbose and not malformed.empty:
        print(f"[time_split] Malformed entries: {len(malformed)}")
        print(malformed[[date_col]].head())

        data["clean_date"] = data["clean_date"].fillna("2008-08-06")
        data["day"] = data["day"].fillna("Wed")
        data["time"] = data["time"].fillna("21:38:18")
        data["zone"] = data["zone"].fillna(-78)

    return data[["date", "day", "time", "zone", "clean_date"]]


def mask_email_domain(email):
    if pd.isnull(email) or '@' not in str(email):
        return 'unknown'
    domain = str(email).split('@')[-1].lower().strip()
    domain = re.sub(r'[^a-z0-9\.-]', '', domain)  # remove junk like '>'
    return f"xxx@{domain}" if domain else 'unknown'


def normalise_time(df, clean_date_col='clean_date', time_col='time', zone_col='zone', verbose=True):
    """
    Adds ML-friendly time features based on parsed date/time/zone columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame (already processed by time_split)
    - clean_date_col (str): Column with normalized date strings
    - time_col (str): Column with time strings
    - zone_col (str): Column with numeric timezone offsets
    - verbose (bool): If True, prints diagnostics

    Returns:
    - pd.DataFrame: DataFrame with new features ['weekday', 'hour', 'is_weekend', 'utc_offset_hr', 'datetime_local', 'datetime_utc']
    """

    # Convert clean_date to datetime
    df[clean_date_col] = pd.to_datetime(df[clean_date_col], errors='coerce')

    # Extract hour from time
    df['hour'] = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce').dt.hour

    # Convert zone to UTC offset in hours
    df['time_zone_diff'] = df[zone_col] // 100

    # Build local datetime
    df['datetime_local'] = pd.to_datetime(
        df[clean_date_col].dt.strftime('%Y-%m-%d') + ' ' + df[time_col],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # Normalize to UTC
    df['datetime_utc'] = df['datetime_local'] - pd.to_timedelta(df['time_zone_diff'], unit='h')

    return df
