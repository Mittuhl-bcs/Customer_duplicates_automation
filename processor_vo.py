# Import necessary libraries
import pandas as pd
import numpy as np
import re
import jellyfish
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from itertools import combinations
from rapidfuzz import fuzz
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import string
import BCS_connector
import mailer
from datetime import datetime, timedelta
import domain_based_matcher

# Step 1: Read and Preprocess Data


df = BCS_connector.reader_df()
df.to_excel("D:\\Customer_data_duplication_check\\p21_data\\customer_data.xlsx", index=False)  # Replace with your actual file name

df['customer_id'] = df['customer_id'].astype(int)

# Ensure all necessary columns are present
required_columns = [
    'customer_id', 'customer_name', 'phys_state', 'phys_city',
    'email_address', 'phys_address1', 'mail_city', 'mail_postal_code'
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"The following required columns are missing in the input file: {missing_columns}"
    )

# Data Preprocessing Functions
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # Remove common company suffixes
    suffixes = [
        'llc', 'inc', 'ltd', 'corp', 'co', 'company',
        'incorporated', 'limited', 'corp', 'corporation', 'plc', 'gmbh', 'srl',
        'sa', 'ag', 'kg', 'oy', 'ab', 'as', 'pte', 'pte ltd', 'llp', 'lp'
    ]
    pattern = r'\b(' + '|'.join(suffixes) + r')\b'
    text = re.sub(pattern, '', text)
    # Expand common abbreviations
    abbreviations = {
        'intl': 'international',
        'tech': 'technology',
        'mfg': 'manufacturing',
        'svc': 'service',
        'svcs': 'services',
        'mgmt': 'management',
        'grp': 'group',
        'inst': 'institute',
        'univ': 'university',
        'dept': 'department',
        'deptt': 'department',
        'co': 'company',
        'cos': 'companies',
        'corp': 'corporation',
        'assn': 'association',
        'assoc': 'association',
        'org': 'organization',
        'dept': 'department',
        'hosp': 'hospital',
        'med': 'medical',
        'ctr': 'center',
        'cnt': 'center',
        'cntre': 'centre',
        'ctr': 'centre'
    }
    abbr_pattern = re.compile(r'\b(' + '|'.join(abbreviations.keys()) + r')\b')
    text = abbr_pattern.sub(lambda m: abbreviations[m.group()], text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def preprocess_address(address):
    if pd.isnull(address):
        return ''
    address = address.lower()
    address = re.sub(r'[^\w\s]', '', address)  # Remove punctuation
    # Replace common abbreviations
    abbreviations = {
        'st': 'street',
        'rd': 'road',
        'ave': 'avenue',
        'blvd': 'boulevard',
        'ln': 'lane',
        'dr': 'drive',
        'hwy': 'highway',
        'pkwy': 'parkway',
        'ctr': 'center',
        'ct': 'court',
        'pl': 'place',
        'sq': 'square',
        'trl': 'trail',
        'ter': 'terrace',
        'cir': 'circle',
        'loop': 'loop',
        'apt': 'apartment',
        'ste': 'suite',
        'fl': 'floor',
        'bldg': 'building',
        'unit': 'unit',
        'no': 'number',
        'dept': 'department'
    }
    abbr_pattern = re.compile(r'\b(' + '|'.join(abbreviations.keys()) + r')\b')
    address = abbr_pattern.sub(lambda m: abbreviations[m.group()], address)
    address = re.sub(r'\s+', ' ', address)  # Remove extra spaces
    return address.strip()

def standardize_phone_number(phone):
    if pd.isnull(phone):
        return ''
    phone = re.sub(r'\D', '', str(phone))  # Remove non-digit characters
    return phone

df['cleaned_customer_name'] = df['customer_name'].apply(preprocess_text)
df['cleaned_phys_address1'] = df['phys_address1'].apply(preprocess_address)
df['cleaned_email_address'] = df['email_address'].apply(lambda x: x.lower() if pd.notnull(x) else '')
#df['cleaned_phone_number'] = df['phone_number'].apply(standardize_phone_number)
df['cleaned_phys_city'] = df['phys_city'].apply(preprocess_text)
df['cleaned_phys_state'] = df['phys_state'].apply(preprocess_text)
df['cleaned_mail_postal_code'] = df['mail_postal_code'].apply(lambda x: str(x).strip() if pd.notnull(x) else '')

# Generate Phonetic Codes for Blocking
def double_metaphone(name):
    tokens = name.split()
    metaphone_tokens = [jellyfish.metaphone(token) for token in tokens]
    return ' '.join(metaphone_tokens)

df['phonetic_code'] = df['cleaned_customer_name'].apply(double_metaphone)

# Step 2: Build Blocks
df['blocking_key'] = df['phonetic_code'].apply(lambda x: x[:3])  # Use first 3 characters of phonetic code
blocks = df.groupby('blocking_key')

# Step 3: Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model

# Encode all customer names
print("Encoding customer names...")
df['name_embedding'] = list(model.encode(df['cleaned_customer_name'].tolist(), show_progress_bar=True))

# Step 4: Initialize Graph for Clustering
G = nx.Graph()
G.add_nodes_from(df['customer_id'])

# Step 5: Compare Records within Blocks
print("Comparing records within blocks...")
for block_key, block_df in tqdm(blocks, total=len(blocks)):
    if len(block_df) < 2:
        continue  # Skip blocks with only one record
    records = block_df.to_dict('records')
    for rec1, rec2 in combinations(records, 2):
        idx1 = rec1['customer_id']
        idx2 = rec2['customer_id']
        name1 = rec1['cleaned_customer_name']
        name2 = rec2['cleaned_customer_name']
        tokens1 = name1.split()
        tokens2 = name2.split()
        
        # Phonetic Similarity
        phonetic1 = rec1['phonetic_code']
        phonetic2 = rec2['phonetic_code']
        phonetic_sim = fuzz.ratio(phonetic1, phonetic2) / 100
        
        # Embedding Similarity
        embedding1 = rec1['name_embedding']
        embedding2 = rec2['name_embedding']
        name_cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
        
        # String Similarity Measures
        name_fuzz_ratio = fuzz.token_set_ratio(name1, name2) / 100
        partial_ratio = fuzz.partial_ratio(name1, name2) / 100
        levenshtein_dist = jellyfish.levenshtein_distance(name1, name2)
        max_len = max(len(name1), len(name2))
        levenshtein_ratio = (max_len - levenshtein_dist) / max_len if max_len > 0 else 0
        
        # Token-Based Matching
        token_matches = sum(1 for token in tokens1 if token in tokens2)
        token_match_ratio = token_matches / max(len(tokens1), len(tokens2))
        
        # Address Similarity
        address_sim = fuzz.token_set_ratio(rec1['cleaned_phys_address1'], rec2['cleaned_phys_address1']) / 100
        
        # Email and Phone Match
        email_match = int(rec1['cleaned_email_address'] == rec2['cleaned_email_address'] and rec1['cleaned_email_address'] != '')
        #phone_match = int(rec1['cleaned_phone_number'] == rec2['cleaned_phone_number'] and rec1['cleaned_phone_number'] != '')
        
        # Aggregate Similarity Score
        aggregate_score = (
            0.25 * name_cosine_sim +
            0.15 * name_fuzz_ratio +
            0.15 * phonetic_sim +
            0.1 * partial_ratio +
            0.1 * token_match_ratio +
            0.1 * levenshtein_ratio +
            0.05 * address_sim +
            0.05 * email_match
        )
        
        # Decision Thresholds
        if aggregate_score >= 0.75:
            G.add_edge(idx1, idx2, weight=aggregate_score)

# Step 6: Identify Clusters (Primary Groups)
print("Identifying clusters...")
clusters = list(nx.connected_components(G))

# Assign cluster IDs as primary_group_id
customer_id_to_primary_group_id = {}
for primary_group_id, cluster in enumerate(clusters):
    for customer_id in cluster:
        customer_id_to_primary_group_id[customer_id] = primary_group_id

df['primary_group_id'] = df['customer_id'].map(customer_id_to_primary_group_id).fillna(-1).astype(int)

# Step 7: Refining Clusters with Additional Attributes (Secondary Groups)
print("Refining clusters with secondary grouping...")
refined_clusters = []
secondary_group_id_counter = 0  # Counter for unique secondary_group_ids

for primary_group_id in df['primary_group_id'].unique():
    sub_df = df[df['primary_group_id'] == primary_group_id]
    # Group by exact matches on email_address and phys_state within the primary group
    group_keys = ['cleaned_email_address', 'cleaned_phys_state']
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
    remaining_ids = set(sub_df['customer_id']) - grouped_customer_ids
    if remaining_ids:
        # Group by city similarity within remaining records
        records = sub_df[sub_df['customer_id'].isin(remaining_ids)].to_dict('records')
        city_graph = nx.Graph()
        city_graph.add_nodes_from(remaining_ids)
        for rec1, rec2 in combinations(records, 2):
            idx1 = rec1['customer_id']
            idx2 = rec2['customer_id']
            city1 = rec1['cleaned_phys_city']
            city2 = rec2['cleaned_phys_city']
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

# Step 8: Update the duplicate_customer_ids Columns
# For each primary_group_id, get the list of all customer_ids in that group
primary_group_to_customer_ids = df.groupby('primary_group_id')['customer_id'].apply(list).to_dict()

# For each secondary_group_id, get the list of customer_ids in that secondary group
secondary_group_to_customer_ids = df.groupby('secondary_group_id')['customer_id'].apply(list).to_dict()

# Add primary_group_duplicate_ids column
df['primary_group_duplicate_ids'] = df['primary_group_id'].map(primary_group_to_customer_ids)

# Update duplicate_customer_ids column for secondary groups
df['duplicate_customer_ids'] = df['primary_group_id'].map(primary_group_to_customer_ids)

# For records without a secondary group (secondary_group_id == -1), set duplicate_customer_ids to their own customer_id
df.loc[df['secondary_group_id'] == -1, 'duplicate_customer_ids'] = df['customer_id'].apply(lambda x: [x])

# Step 9: Organize the DataFrame
# Sort by primary_group_id, then by priority (descending), then by secondary_group_id
# Sort by primary_group_id, then by priority (descending), then by secondary_group_id
#  df.sort_values(by=['primary_group_id', 'priority', 'secondary_group_id'], ascending=[True, False, True], inplace=True)

# Reset index
# df.reset_index(drop=True, inplace=True)


df['duplicate_check'] = df.apply(
    lambda row: 'yes' if (
        isinstance(row['primary_group_duplicate_ids'], list) and len(row['primary_group_duplicate_ids']) > 1
    ) or (
        isinstance(row['duplicate_customer_ids'], list) and len(row['duplicate_customer_ids']) > 1
    ) else 'no',
    axis=1
)
df["duplicate_count"] = df.apply(
    lambda row: max(
        len(row['primary_group_duplicate_ids']) if isinstance(row['primary_group_duplicate_ids'], list) else 1,
        len(row['duplicate_customer_ids']) if isinstance(row['duplicate_customer_ids'], list) else 1
    ) if row['duplicate_check'] == 'yes' else 'no duplicates',
    axis=1
)

# Create a custom sort key for alphabetical ordering while keeping duplicate clusters together
def create_alphabetical_sort_key(row):
    customer_name = row['customer_name'].strip().lower() if pd.notnull(row['customer_name']) else 'zzz'
    
    if row['duplicate_check'] == 'yes':
        # For duplicates: sort by customer name, then group duplicates together
        return (customer_name, row['primary_group_id'], row['secondary_group_id'], -row['priority'])
    else:
        # For non-duplicates: just sort alphabetically
        return (customer_name, 999999, 999999, 0)  # Large numbers to put after duplicates of same name

# Apply the custom sorting
df['sort_key'] = df.apply(create_alphabetical_sort_key, axis=1)
df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)

# Convert lists to a string representation before calling unique()
df["temp_primary_group_duplicate_ids"] = df["primary_group_duplicate_ids"].apply(lambda x: str(x) if isinstance(x, list) else x)

# Now you can use unique() safely
dict_pids = {v: i for i, v in enumerate(df["temp_primary_group_duplicate_ids"].unique())}


# Use map() to create the Primary_groupings column
df["primary_groupings"] = df["temp_primary_group_duplicate_ids"].map(dict_pids)


def get_alphabetical_label(index):
    # String.ascii_uppercase is 'A' to 'Z'
    letters = string.ascii_uppercase
    result = []
    while index >= 0:
        result.append(letters[index % 26])
        index = index // 26 - 1
    return ''.join(reversed(result))


# Convert lists to a string representation before calling unique()
df["temp_secondary_group_id"] = df["secondary_group_id"].apply(lambda x: str(x) if isinstance(x, list) else x)

# fopr secondary ids
dict_sids = {v: get_alphabetical_label(i) for i, v in enumerate(df["temp_secondary_group_id"].unique())}

# Use map() to create the Primary_groupings column
df["secondary_groupings"] = df["temp_secondary_group_id"].map(dict_sids)
df["secondary_groupings"] = df["secondary_groupings"].astype(str) + df["primary_groupings"].astype(str)
df['customer_date_created'] = pd.to_datetime(df['customer_date_created'])

# Select desired columns
desired_columns = [
    "customer_id", "customer_name", "duplicate_check", "duplicate_count", "duplicate_customer_ids", "primary_groupings", "secondary_groupings",
    "phys_address1", "phys_address2", "phys_city", "phys_state", "phys_postal_code", "email_address",
    "mail_city", "mail_state", "mail_postal_code", "customer_salesrep_id", "date_acct_opened",
    "customer_date_created", "r12_sales", "locations"
]

# Ensure all desired columns are present in the DataFrame
missing_columns = [col for col in desired_columns if col not in df.columns]
for col in missing_columns:
    df[col] = np.nan  # or appropriate default value

df_final = df[desired_columns]

end_date = datetime.now()
start_date = end_date - timedelta(days=31)

recent_duplicate_ids = df_final[
    (df_final['customer_date_created'] >= start_date) & 
    (df_final['customer_date_created'] <= end_date) & 
    (df_final["duplicate_check"] == "yes")
]['customer_id'].tolist()

# Get all primary group IDs for these recent duplicates
recent_primary_groups = df_final[
    df_final['customer_id'].isin(recent_duplicate_ids)
]['primary_groupings'].unique().tolist()

# Get all records that belong to these primary groups (including older duplicates in the same groups)
df_latest = df_final[
    df_final['primary_groupings'].isin(recent_primary_groups)
].copy()

today = datetime.today()

# Format the date
formatted_date = today.strftime("%d_%b_%Y").lower()

output_path = f"D:\\Customer_data_duplication_check\\processed_data\\customer_duplicates_{formatted_date}.xlsx"

# Save both DataFrames in different sheets
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_final.to_excel(writer, sheet_name='Final_Data', index=False)
    df_latest.to_excel(writer, sheet_name='Latest_Dups', index=False)

# Write to Excel
# df_final.to_excel(f"D:\\Customer_data_duplication_check\\processed_data\\customer_duplicates_{formatted_date}.xlsx", index=False)
# df_latest.to_excel(f"D:\\Customer_data_duplication_check\\processed_data\\customer_duplicates_{formatted_date}.xlsx", index=False)
print(f"Deduplication complete. Results saved to 'customer_duplicates_{formatted_date}.xlsx'.")

domain_df = domain_based_matcher.main()

dom_output_path = f"D:\\Customer_data_duplication_check\\processed_data\\domain_based_customer_duplicates_{formatted_date}.xlsx"

# Save both DataFrames in different sheets
with pd.ExcelWriter(dom_output_path, engine='openpyxl') as writer:
    domain_df.to_excel(writer, sheet_name='Final_Data', index=False)

files_l = [output_path, dom_output_path]
print(f"Domain based matching complete. Results saved to 'domain_based_customer_duplicates_{formatted_date}.xlsx'.")


mailer.sender(files_l)