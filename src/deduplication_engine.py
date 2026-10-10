"""
Voter Registry Deduplication & Entity Resolution Engine (Optimized)
===================================================================
Applies:
1. Multi-pass Phonetic Blocking & Fast Heuristic Pre-Filtering.
2. Batched Machine Learning classification on candidate pairs.
3. Disjoint Set Union (DSU) graph clustering to merge duplicate voter IDs.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

from similarity_algorithms import soundex, token_sort_similarity
from ml_matcher import VoterMLMatcher, extract_pairwise_features


class DisjointSetUnion:
    """Disjoint Set Union (DSU) / Union-Find with path compression."""
    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1
            return True
        return False


def run_deduplication(
    registry_csv: str,
    matcher_model_path: str,
    confidence_threshold: float = 0.70,
    output_report_csv: str = None
) -> pd.DataFrame:
    print(f"Loading voter registry from {registry_csv}...", flush=True)
    df = pd.read_csv(registry_csv)
    voter_ids = df["Voter_ID"].tolist()
    
    matcher = VoterMLMatcher(matcher_model_path)
    
    # 1. Blocking: Last name initial + Soundex
    print("Building phonetic blocking index...", flush=True)
    blocks = defaultdict(list)
    for _, row in df.iterrows():
        ln = str(row["Last_Name"])
        block_key = f"{ln[:1].upper()}_{soundex(ln)}"
        blocks[block_key].append(row)
        
    print(f"Formed {len(blocks)} candidate blocks across {len(df)} voters.", flush=True)
    
    # 2. Gather candidate pairs with fast pre-filtering
    candidate_pairs = []
    feature_rows = []
    
    comparisons_considered = 0
    total_naive_pairs = (len(df) * (len(df) - 1)) // 2
    
    for block_key, members in blocks.items():
        m_len = len(members)
        if m_len < 2:
            continue
        for i in range(m_len):
            for j in range(i + 1, m_len):
                comparisons_considered += 1
                r1 = members[i]
                r2 = members[j]
                
                # Fast pre-filter: candidates must share either first initial, DOB, or high token similarity
                fn1, fn2 = str(r1["First_Name"]), str(r2["First_Name"])
                dob1, dob2 = str(r1["Date_Of_Birth"]), str(r2["Date_Of_Birth"])
                
                if (fn1[:1].upper() != fn2[:1].upper()) and (dob1 != dob2) and (soundex(fn1) != soundex(fn2)):
                    continue
                    
                feats = extract_pairwise_features(
                    str(r1["Full_Name"]), str(r2["Full_Name"]),
                    str(r1["Address"]), str(r2["Address"]),
                    dob1, dob2
                )
                feature_rows.append(feats)
                candidate_pairs.append((r1, r2))

    print(f"Evaluated {comparisons_considered:,} block pairs. Extracted {len(feature_rows)} high-potential candidate vectors.", flush=True)
    
    # 3. Batched ML Prediction
    dsu = DisjointSetUnion(voter_ids)
    matched_pairs = []
    
    if feature_rows:
        X_batch = np.array(feature_rows)
        probabilities = matcher.clf.predict_proba(X_batch)[:, 1]
        
        for (r1, r2), prob in zip(candidate_pairs, probabilities):
            if prob >= confidence_threshold:
                dsu.union(r1["Voter_ID"], r2["Voter_ID"])
                matched_pairs.append({
                    "Voter_ID_1": r1["Voter_ID"],
                    "Name_1": r1["Full_Name"],
                    "DOB_1": r1["Date_Of_Birth"],
                    "Address_1": r1["Address"],
                    "Voter_ID_2": r2["Voter_ID"],
                    "Name_2": r2["Full_Name"],
                    "DOB_2": r2["Date_Of_Birth"],
                    "Address_2": r2["Address"],
                    "Match_Probability": round(float(prob), 4)
                })

    # 4. Cluster mapping
    cluster_map = {}
    for vid in voter_ids:
        root = dsu.find(vid)
        if root not in cluster_map:
            cluster_map[root] = []
        cluster_map[root].append(vid)
        
    duplicate_clusters = {k: v for k, v in cluster_map.items() if len(v) > 1}
    reduction_pct = ((total_naive_pairs - comparisons_considered) / total_naive_pairs) * 100.0
    
    print("\n" + "=" * 65, flush=True)
    print("            VOTER DEDUPLICATION AUDIT SUMMARY            ", flush=True)
    print("=" * 65, flush=True)
    print(f"Total Registry Size              : {len(df):,} records", flush=True)
    print(f"Naive O(N^2) Search Space        : {total_naive_pairs:,} pairs", flush=True)
    print(f"Phonetic Blocking Search Space   : {comparisons_considered:,} pairs ({reduction_pct:.1f}% reduction)", flush=True)
    print(f"Candidate Feature Vectors Checked: {len(feature_rows):,}", flush=True)
    print(f"Confirmed Duplicate Pairs Found  : {len(matched_pairs)}", flush=True)
    print(f"Distinct Duplicate Clusters      : {len(duplicate_clusters)}", flush=True)
    print("=" * 65, flush=True)
    
    result_df = pd.DataFrame(matched_pairs)
    if output_report_csv:
        os.makedirs(os.path.dirname(output_report_csv), exist_ok=True)
        result_df.to_csv(output_report_csv, index=False)
        print(f"Saved duplicate resolution report to {output_report_csv}", flush=True)
        
    return result_df


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent
    reg_file = base_dir / "data" / "voter_registry.csv"
    mod_file = base_dir / "models" / "voter_matcher_rf.joblib"
    out_file = base_dir / "reports" / "flagged_duplicate_voters.csv"
    
    run_deduplication(str(reg_file), str(mod_file), output_report_csv=str(out_file))
