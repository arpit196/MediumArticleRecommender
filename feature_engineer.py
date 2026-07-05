import ast
def clean_tags(tags_data):
    # If it's already a list, join it
    if isinstance(tags_data, list):
        return ", ".join(map(str, tags_data))
    # If it's a string representation of a list, parse it then join
    try:
        tags_list = ast.literal_eval(tags_data)
        return ", ".join(map(str, tags_list))
    except (ValueError, SyntaxError):
        # Fallback if the data is malformed (e.g., plain text or empty)
        return str(tags_data)

def generate_initial_df(articles):
    articles['formatted_tags'] = articles['tags'].apply(clean_tags)
    articles['combined_content'] = articles.apply(
        lambda row: f"{row['formatted_tags']} | {row['title']} | {str(row['text'])[:300]}", 
        axis=1
    )
    return articles

################### Generating valuable metadata, including age, and author embedding for each document ####################
def append_metadata_and_calc_age(df_persona, df_articles, request_time_str="2026-07-03 12:00:00.000000+00:00"):
    """
    Maps author and timestamp from df_articles using row positional index,
    and calculates document age in days.
    
    Parameters:
    df_persona (pd.DataFrame): Your structured LTR dataframe (containing 'article_id')
    df_articles (pd.DataFrame): The master dataframe where position corresponds to 'article_id'
    request_time_str (str): The mock timestamp of the recommendation engine request.
    """
    # 1. Create a copy of the columns we need from the master df
    df_meta = df_articles[['authors', 'timestamp']].copy()
    
    # 2. Expose the dataframe's row positions as a column called 'article_id'
    df_meta = df_meta.reset_index().rename(columns={'index': 'article_id'})
    
    # Ensure timestamps are parsed as standard datetimes
    df_meta['timestamp'] = pd.to_datetime(df_meta['timestamp'], errors='coerce')
    
    # 3. Merge the metadata into the persona dataframe based on row positions
    updated_df = pd.merge(df_persona, df_meta, on='article_id', how='left')
    
    # 4. Calculate Document Age
    request_time = pd.to_datetime(request_time_str)
    time_delta = request_time - updated_df['timestamp']
    
    # Convert total delta into fractional days
    updated_df['document_age_days'] = time_delta.dt.total_seconds() / (24 * 3600)
    updated_df['document_age_days'] = updated_df['document_age_days'].clip(lower=0)
    
    # 5. Engineering feature: Log-transform the age
    updated_df['log_document_age'] = np.log1p(updated_df['document_age_days'])
    
    return updated_df