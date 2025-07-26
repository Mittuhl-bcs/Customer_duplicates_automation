# Import necessary libraries
import pandas as pd
import numpy as np
import re
import jellyfish
import networkx as nx
from itertools import combinations
from rapidfuzz import fuzz

# Read the DataFrame from an Excel file
# Replace 'customer_list.xlsx' with your actual file name
df = pd.read_excel('customer_list.xlsx')
df["email_address"] = df["email_address"].str.lower()


# Ensure all necessary columns are present
required_columns = [
    'customer_id', 'customer_name', 'phys_state', 'phys_city',
    'email_address', 'mail_city', 'mail_postal_code'
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"The following required columns are missing in the input file: {missing_columns}"
    )

# Step 1: Data Preprocessing
def preprocess_customer_name(name):
    if pd.isnull(name):
        return ''
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    suffixes = [
        'llc', 'inc', 'ltd', 'corp', 'co', 'company',
        'incorporated', 'limited'
    ]
    pattern = r'\b(' + '|'.join(suffixes) + r')\b'
    name = re.sub(pattern, '', name)
    name = re.sub(r'\s+', ' ', name)  # Remove extra spaces
    # Expand common abbreviations
    abbreviations = {
        'mech': 'mechanical',
        'tech': 'technology',
        'intl': 'international',
        'svc': 'service'
    }
    abbr_pattern = re.compile(
        r'\b(' + '|'.join(abbreviations.keys()) + r')\b'
    )
    name = abbr_pattern.sub(lambda m: abbreviations[m.group()], name)
    return name.strip()

df['cleaned_customer_name'] = df['customer_name'].apply(preprocess_customer_name)


# Step 2: Blocking using Phonetic Encoding
def double_metaphone(name):
    return jellyfish.metaphone(name)

df['phonetic_code'] = df['cleaned_customer_name'].apply(double_metaphone)

# Create blocks based on phonetic codes
blocks = df.groupby('phonetic_code')

# Step 3: Advanced String Matching and Similarity Scoring
def compute_similarity_scores(block):
    records = block.to_dict('records')
    pairs = []
    for rec1, rec2 in combinations(records, 2):
        idx1 = rec1['customer_id']
        idx2 = rec2['customer_id']
        name1 = rec1['cleaned_customer_name']
        name2 = rec2['cleaned_customer_name']
        # Use multiple similarity measures from rapidfuzz
        ratio = fuzz.token_set_ratio(name1, name2)
        partial_ratio = fuzz.partial_ratio(name1, name2)
        avg_score = (ratio + partial_ratio) / 2
        threshold = 85  # Adjust threshold as needed (out of 100)
        if avg_score >= threshold:
            pairs.append((idx1, idx2))
    return pairs

# Build a graph of similar names
G = nx.Graph()
G.add_nodes_from(df['customer_id'])

for _, block in blocks:
    similar_pairs = compute_similarity_scores(block)
    G.add_edges_from(similar_pairs)

# Step 4: Clustering Similar Customer Names (Primary Groups)
# Identify connected components (clusters) - Primary Groups
primary_clusters = list(nx.connected_components(G))

# Map customer_id to primary_group_id
customer_id_to_primary_group_id = {}
for primary_group_id, cluster in enumerate(primary_clusters):
    for customer_id in cluster:
        customer_id_to_primary_group_id[customer_id] = primary_group_id

df['primary_group_id'] = df['customer_id'].map(customer_id_to_primary_group_id)

# Step 5: Refining Clusters with Additional Attributes (Secondary Groups)
# Within each primary group, refine clusters
refined_clusters = []
secondary_group_id_counter = 0  # Counter for unique secondary_group_ids

for primary_group_id, cluster in enumerate(primary_clusters):
    sub_df = df[df['customer_id'].isin(cluster)]
    # Group by exact matches on email and phys_state within the primary group
    group_keys = ['email_address', 'phys_state']
    grouped = sub_df.groupby(group_keys)
    secondary_group_ids_in_primary_group = []
    for group_key, group in grouped:
        if len(group) > 1 and any(group_key):  # Ensure at least one key is not empty
            secondary_group = set(group['customer_id'])
            refined_clusters.append({
                'primary_group_id': primary_group_id,
                'secondary_group': secondary_group,
                'secondary_group_id': secondary_group_id_counter,
                'priority': 3  # Highest priority
            })
            secondary_group_ids_in_primary_group.append(secondary_group)
            secondary_group_id_counter += 1
    # Handle remaining records not in any secondary group
    grouped_customer_ids = set().union(*secondary_group_ids_in_primary_group)
    remaining_ids = set(cluster) - grouped_customer_ids
    if remaining_ids:
        # Group by city similarity within remaining records
        records = sub_df[sub_df['customer_id'].isin(remaining_ids)].to_dict('records')
        city_graph = nx.Graph()
        city_graph.add_nodes_from(remaining_ids)
        for rec1, rec2 in combinations(records, 2):
            idx1 = rec1['customer_id']
            idx2 = rec2['customer_id']
            city1 = str(rec1['phys_city']).lower()
            city2 = str(rec2['phys_city']).lower()
            if city1 and city2:
                city_score = fuzz.token_set_ratio(city1, city2)
                if city_score >= 85:
                    city_graph.add_edge(idx1, idx2)
        city_clusters = list(nx.connected_components(city_graph))
        for city_cluster in city_clusters:
            refined_clusters.append({
                'primary_group_id': primary_group_id,
                'secondary_group': city_cluster,
                'secondary_group_id': secondary_group_id_counter,
                'priority': 2  # Medium priority
            })
            secondary_group_id_counter += 1
        # Handle any remaining records
        city_grouped_ids = set().union(*city_clusters)
        final_remaining_ids = remaining_ids - city_grouped_ids
        for customer_id in final_remaining_ids:
            refined_clusters.append({
                'primary_group_id': primary_group_id,
                'secondary_group': {customer_id},
                'secondary_group_id': secondary_group_id_counter,
                'priority': 1  # Lower priority
            })
            secondary_group_id_counter += 1

# Create DataFrame from refined_clusters
group_records = []
for cluster_info in refined_clusters:
    for customer_id in cluster_info['secondary_group']:
        group_records.append({
            'customer_id': customer_id,
            'primary_group_id': cluster_info['primary_group_id'],
            'secondary_group_id': cluster_info['secondary_group_id'],
            'priority': cluster_info['priority']
        })

group_df = pd.DataFrame(group_records)

# Merge group information back to the main DataFrame
df = df.merge(group_df, on=['customer_id', 'primary_group_id'], how='left')

# Handle ungrouped records (if any)
df['secondary_group_id'] = df['secondary_group_id'].fillna(-1).astype(int)
df['priority'] = df['priority'].fillna(0).astype(int)

# Step 6: Update the duplicate_customer_ids Columns
# For each primary_group_id, get the list of all customer_ids in that group
primary_group_to_customer_ids = df.groupby('primary_group_id')['customer_id'].apply(list).to_dict()

# For each secondary_group_id, get the list of customer_ids in that secondary group
secondary_group_to_customer_ids = df.groupby('secondary_group_id')['customer_id'].apply(list).to_dict()

# Add primary_group_duplicate_ids column
df['primary_group_duplicate_ids'] = df['primary_group_id'].map(primary_group_to_customer_ids)

# Add duplicate_customer_ids column for secondary groups
df['duplicate_customer_ids'] = df['secondary_group_id'].map(secondary_group_to_customer_ids)

# For records without a secondary group (secondary_group_id == -1), set duplicate_customer_ids to their own customer_id
df.loc[df['secondary_group_id'] == -1, 'duplicate_customer_ids'] = df['customer_id'].apply(lambda x: [x])

# Step 7: Organize the DataFrame
# Sort by primary_group_id, then by priority (descending), then by secondary_group_id
df.sort_values(by=['primary_group_id', 'priority', 'secondary_group_id'], ascending=[True, False, True], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Select desired columns
desired_columns = [
    "customer_id", "customer_name", "duplicate_customer_ids", "corp_address_id", "mail_address1",
    "mail_address2", "mail_city", "mail_state", "mail_postal_code", "phys_address1", "phys_address2",
    "phys_city", "phys_state", "phys_postal_code", "email_address", "customer_salesrep_id",
    "date_acct_opened", "customer_date_created", "r12_sales", "locations"
]

# Ensure all desired columns are present in the DataFrame
missing_columns = [col for col in desired_columns if col not in df.columns]
for col in missing_columns:
    df[col] = np.nan  # or appropriate default value

df_final = df[desired_columns]

# Step 8: Write the result to a new Excel file
# Replace 'output_file.xlsx' with your desired output file name
df_final.to_excel('output_file_refined.xlsx', index=False)
