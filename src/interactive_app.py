"""
Interactive Voter Search & Resolution Console
==============================================
Enables real-time fuzzy matching and ML probability resolution for noisy voter queries.

Usage:
  python src/interactive_app.py --name "Smtih, Jonathon" --dob "1985-06-12"
  python src/interactive_app.py --interactive
"""

import argparse
import pandas as pd
from pathlib import Path

from similarity_algorithms import (
    jaro_winkler_similarity,
    token_sort_similarity,
    levenshtein_similarity,
    soundex
)
from ml_matcher import VoterMLMatcher


def search_voter(
    query_name: str,
    query_dob: str = "",
    query_addr: str = "",
    registry_csv: str = None,
    matcher_model_path: str = None,
    top_k: int = 5
):
    base_dir = Path(__file__).resolve().parent.parent
    if registry_csv is None:
        registry_csv = str(base_dir / "data" / "voter_registry.csv")
    if matcher_model_path is None:
        matcher_model_path = str(base_dir / "models" / "voter_matcher_rf.joblib")
        
    df = pd.read_csv(registry_csv)
    matcher = VoterMLMatcher(matcher_model_path)
    
    results = []
    for _, row in df.iterrows():
        reg_name = str(row["Full_Name"])
        reg_addr = str(row["Address"])
        reg_dob = str(row["Date_Of_Birth"])
        
        # Calculate algorithmic metrics
        tok_sim = token_sort_similarity(query_name, reg_name)
        jw_sim = jaro_winkler_similarity(query_name, reg_name)
        lev_sim = levenshtein_similarity(query_name, reg_name)
        
        # ML Record Linkage match probability
        _, match_prob = matcher.predict_match(
            query_name, reg_name,
            query_addr, reg_addr,
            query_dob, reg_dob
        )
        
        # Blended ranking score
        rank_score = (match_prob * 0.6) + (tok_sim * 0.4)
        
        results.append({
            "Voter_ID": row["Voter_ID"],
            "Full_Name": reg_name,
            "Date_Of_Birth": reg_dob,
            "Address": reg_addr,
            "ML_Confidence": match_prob * 100.0,
            "Token_Sort_Sim": tok_sim * 100.0,
            "Jaro_Winkler": jw_sim * 100.0,
            "Rank_Score": rank_score
        })
        
    res_df = pd.DataFrame(results).sort_values(by="Rank_Score", ascending=False).head(top_k)
    
    print("\n" + "=" * 80)
    print(f"SEARCH QUERY: '{query_name}' | DOB: '{query_dob or 'N/A'}' | Addr: '{query_addr or 'N/A'}'")
    print("=" * 80)
    print(f"{'Voter ID':<10} | {'Matched Name':<20} | {'DOB':<10} | {'ML Conf %':<9} | {'Token Sort':<10} | {'Status'}")
    print("-" * 80)
    
    for _, r in res_df.iterrows():
        status = "[HIGH MATCH]" if r["ML_Confidence"] >= 75 else ("[REVIEW]" if r["ML_Confidence"] >= 50 else "[LOW]")
        print(f"{r['Voter_ID']:<10} | {r['Full_Name']:<20} | {r['Date_Of_Birth']:<10} | {r['ML_Confidence']:7.1f}% | {r['Token_Sort_Sim']:8.1f}% | {status}")
    print("=" * 80 + "\n")
    return res_df


def interactive_mode():
    print("=" * 70)
    print("    VOTER SEARCH & MATCHING SYSTEM - INTERACTIVE CONSOLE")
    print("=" * 70)
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            name = input("Enter Voter Name (e.g. 'Smith, Jhon', 'Bill Williams'): ").strip()
            if not name or name.lower() == 'exit':
                break
            dob = input("Enter Date of Birth (YYYY-MM-DD, or press Enter to skip): ").strip()
            addr = input("Enter Address (or press Enter to skip): ").strip()
            search_voter(name, dob, addr)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voter Search and Fuzzy Matching System")
    parser.add_argument("--name", type=str, help="Voter query name")
    parser.add_argument("--dob", type=str, default="", help="Optional Date of Birth (YYYY-MM-DD)")
    parser.add_argument("--addr", type=str, default="", help="Optional Street Address")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI")
    args = parser.parse_args()
    
    if args.interactive or not args.name:
        interactive_mode()
    else:
        search_voter(args.name, args.dob, args.addr)
