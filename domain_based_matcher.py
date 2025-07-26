import pandas as pd
import numpy as np
import string
import re


def main():
    # Load your data
    df = pd.read_excel("D:\\Customer_data_duplication_check\\p21_data\\customer_data.xlsx")

    # Convert customer_domain to lowercase
    df['customer_domain'] = df['customer_domain'].str.lower()

    """
    # Define the patterns
    patterns_to_blank = ['gmail', 'yahoo', 'hotmail', 'airmail']
    combined_pattern = re.compile(rf".*({'|'.join(patterns_to_blank)}).*")

    df['customer_domain'] = df['customer_domain'].apply(
        lambda x: '' if pd.notna(x) and combined_pattern.search(x) else x
    )
    """

    # Replace blank strings with NaN in domain
    df['customer_domain'] = df['customer_domain'].replace('', np.nan)

    # Step 1: Identify matched domains
    group_info = {}
    group_id_counter = 1
    alphabet = list(string.ascii_uppercase)

    grouped = df[~df['customer_domain'].isna()].groupby('customer_domain')

    for domain, group in sorted(grouped):  # sorted ensures deterministic order
        if len(group) > 1:
            group_id = f"{group_id_counter}{alphabet[(group_id_counter - 1) % 26]}"
            cust_ids = group['customer_id'].tolist()
            group_info[domain] = {
                'group_id': group_id,
                'matching_ids': cust_ids
            }
            group_id_counter += 1

    # Step 2: Assign match info
    def get_matching_check(row):
        domain = row['customer_domain']
        if pd.isna(domain):
            return 'not matched'
        return 'matched' if domain in group_info else 'not matched'

    def get_group_id(row):
        domain = row['customer_domain']
        if domain in group_info:
            return group_info[domain]['group_id']
        return np.nan

    def get_matching_ids(row):
        domain = row['customer_domain']
        if domain in group_info:
            return group_info[domain]['matching_ids']
        return np.nan

    df['matching_check'] = df.apply(get_matching_check, axis=1)
    df['group_id'] = df.apply(get_group_id, axis=1)
    df['matching_ids'] = df.apply(get_matching_ids, axis=1)

    # Step 3: Assign a cluster ID to each row (either domain or row index)
    df['cluster_id'] = df.apply(lambda row: row['customer_domain'] if pd.notna(row['group_id']) else f'unmatched_{row.name}', axis=1)

    # Step 4: Compute cluster sort name: lowest `cust_name` in that cluster
    cluster_names = df.groupby('cluster_id')['customer_name'].transform(lambda x: x.min().strip().lower() if x.notna().any() else 'zzz')
    df['cluster_sort_name'] = cluster_names

    # Step 5: Sort by cluster_sort_name → this sorts groups + singletons alphabetically
    df_sorted = df.sort_values(by=['cluster_sort_name', 'group_id', 'customer_name'], na_position='last').reset_index(drop=True)

    # Step 6: Drop helper columns if needed
    df_sorted.drop(columns=['cluster_id', 'cluster_sort_name'], inplace=True)

    # Step 7: Save result
    #df_sorted.to_excel("domain_based_matches.xlsx", index=False)

    print("Final sorted output saved. Clusters are now sequential and alphabetically structured by `customer_name`.")

    return df_sorted


if __name__ == "__main__":
    df = main()