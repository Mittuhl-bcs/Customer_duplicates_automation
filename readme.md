## Objective 
This code helps in identifying companies saved in different variations and formats - company (customer duplicates)

## Processor:
- Load data using BCS_connector.reader_df().
- Export raw data to Excel for record-keeping.
- Validate required columns.
- Apply preprocessing:
    - customer_name: Lowercasing, punctuation removal, suffix & abbreviation normalization.
    - address: Similar cleaning and expansion of abbreviations.
    - email_address: Lowercased.
    - phonetic_code: Uses Metaphone to encode names phonetically.
    - Blocking key = first 3 characters of the phonetic code.
- Group customers using the blocking_key to limit comparisons to similar-sounding names, reducing computation time.
- Load the model all-MiniLM-L6-v2 to compute semantic embeddings of customer names.
- For each pair of customers within a block:
    - we Compute these:
        - Phonetic similarity (fuzz.ratio)
        - Sentence embedding cosine similarity
        - Fuzzy text ratios (token_set_ratio, partial_ratio)
        - Levenshtein ratio
        - Token overlap
        - Address similarity
        - Email match
        - Aggregate score from weighted combination of these features.
        - If aggregate_score >= 0.75, connect them in a graph (duplicate candidates).
- Use networkx to extract connected components from the graph.
- Assign each group a primary_group_id.
- For each primary_group_id:
    - Sub-group using:
        - Exact matches on email + state (high priority)
        - High city name similarity (medium priority)
        - Remaining singletons (low priority)
    - Assign:
        - secondary_group_id
        - priority (3 > 2 > 1)
- Merge the secondary_group_id and priority back into main df.
- Create primary_group_duplicate_ids and duplicate_customer_ids as lists of customer_ids per group.
- Mark records with duplicate_check = 'yes' if they belong to a group.
- Sort:
    - Alphabetically by cluster representative name
    - Then by group ID and priority
- Assign:
    - primary_groupings: Numeric cluster labels
    - secondary_groupings: Alphabetical labels like A0, B1, etc.
- Filter customers created in the last 31 days.
    - For those marked as duplicates:
        - Extract the entire group they belong to (even if others are older).
        - Save as df_latest.
- Create Excel file with these and save it
    - Final_Data: All customers with duplicate tagging
    - Latest_Dups: Only recent duplicates and their groups
- Run domain_based_matcher.main() to identify matches by domain patterns.


## Domain_based_matcher.py
- This is for matching the names based on their domains such as (gmail, workaci etc..).