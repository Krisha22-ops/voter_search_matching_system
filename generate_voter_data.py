"""
Noisy Voter Registry & Search Benchmark Generator
==================================================
Simulates a real-world municipal voter database with realistic perturbations:
1. Ground-truth registered voter roster (`data/voter_registry.csv`)
2. Noisy real-world search queries (`data/voter_search_queries.csv`)
3. Labeled training pairs (`data/voter_training_pairs.csv`) for Machine Learning Record Linkage.
"""

import os
import random
import pandas as pd
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
    "Christopher", "Daniel", "Matthew", "Anthony", "Donald", "Mark", "Paul", "Steven", "Andrew", "Kenneth",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
    "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

STREETS = [
    "Main St", "Oak Ave", "Maple Rd", "Cedar Ln", "Washington Blvd", "Pine St", "Elm St", "Highland Dr",
    "Lincoln Way", "Park Ave", "Sunset Blvd", "Broadway", "Lakeview Dr", "River Rd", "Chester St"
]

NICKNAMES = {
    "William": ["Bill", "Billy", "Liam", "Will"],
    "Robert": ["Bob", "Bobby", "Rob"],
    "Richard": ["Dick", "Rich", "Rick"],
    "James": ["Jim", "Jimmy"],
    "Michael": ["Mike", "Mikey"],
    "Elizabeth": ["Liz", "Beth", "Eliza", "Betty"],
    "Margaret": ["Peggy", "Maggie", "Marge"],
    "Jennifer": ["Jen", "Jenny"],
    "Thomas": ["Tom", "Tommy"],
    "Christopher": ["Chris"],
    "Matthew": ["Matt"],
    "Anthony": ["Tony"],
    "Steven": ["Steve", "Stephen"]
}

def inject_ocr_typo(text: str) -> str:
    """Injects realistic single-character OCR or keyboard typos."""
    if len(text) <= 3:
        return text
    action = random.choice(["swap", "delete", "replace", "insert"])
    chars = list(text)
    idx = random.randint(1, len(chars) - 2)
    
    if action == "swap":
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif action == "delete":
        chars.pop(idx)
    elif action == "replace":
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
    elif action == "insert":
        chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz"))
        
    return "".join(chars)

def generate_voter_datasets(base_dir: str):
    data_dir = Path(base_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Canonical Voter Registry (2,000 voters)
    print("Generating Canonical Voter Registry...")
    n_voters = 2000
    voter_records = []
    
    for i in range(1, n_voters + 1):
        vid = f"VTR-{i:06d}"
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        year = random.randint(1945, 2004)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dob = f"{year}-{month:02d}-{day:02d}"
        
        street_no = random.randint(100, 9999)
        street = random.choice(STREETS)
        addr = f"{street_no} {street}"
        zip_code = random.choice(["98101", "98102", "98103", "98104", "98105"])
        
        voter_records.append({
            "Voter_ID": vid,
            "First_Name": fn,
            "Last_Name": ln,
            "Full_Name": f"{fn} {ln}",
            "Date_Of_Birth": dob,
            "Address": addr,
            "City": "Seattle",
            "State": "WA",
            "Zip_Code": zip_code
        })
        
    voter_df = pd.DataFrame(voter_records)
    registry_csv = data_dir / "voter_registry.csv"
    voter_df.to_csv(registry_csv, index=False)
    print(f"Saved {len(voter_df)} voters to {registry_csv}")

    # 2. Generate Noisy Search Queries (500 queries with ground truth)
    print("Generating Noisy Search Query Benchmark...")
    search_queries = []
    
    for i in range(500):
        # 80% matches an existing voter, 20% unmatched
        is_matched = random.random() < 0.80
        if is_matched:
            target = voter_df.sample(1).iloc[0]
            true_vid = target["Voter_ID"]
            fn = target["First_Name"]
            ln = target["Last_Name"]
            addr = target["Address"]
            dob = target["Date_Of_Birth"]
            
            # Apply perturbations
            perturbation = random.choice(["typo", "nickname", "inverted", "address_var", "clean"])
            if perturbation == "typo":
                if random.random() < 0.5:
                    query_name = f"{inject_ocr_typo(fn)} {ln}"
                else:
                    query_name = f"{fn} {inject_ocr_typo(ln)}"
            elif perturbation == "nickname" and fn in NICKNAMES:
                nick = random.choice(NICKNAMES[fn])
                query_name = f"{nick} {ln}"
            elif perturbation == "inverted":
                query_name = f"{ln}, {fn}"
            else:
                query_name = f"{fn} {ln}"
                
            query_addr = addr.replace("St", "Street").replace("Ave", "Avenue").replace("Rd", "Road") if random.random() < 0.5 else addr
        else:
            true_vid = "NONE"
            fn = random.choice(["Zackary", "Desmond", "Bryce", "Brianna", "Gwen", "Kendall"])
            ln = random.choice(["Wellington", "VanDyke", "Kensington", "O'Reilly"])
            query_name = f"{fn} {ln}"
            query_addr = f"{random.randint(100, 999)} Pinecrest Way"
            dob = "1990-01-01"
            
        search_queries.append({
            "Query_ID": f"QRY_{i+1:04d}",
            "Input_Name": query_name,
            "Input_Address": query_addr,
            "Input_DOB": dob,
            "Expected_Voter_ID": true_vid
        })
        
    query_df = pd.DataFrame(search_queries)
    query_csv = data_dir / "voter_search_queries.csv"
    query_df.to_csv(query_csv, index=False)
    print(f"Saved {len(query_df)} search queries to {query_csv}")

    # 3. Generate Machine Learning Record Linkage Training Pairs (1,200 pairs)
    print("Generating Pairwise Training Data for Machine Learning Linkage...")
    pairs = []
    
    # 600 Positive match pairs (noisy duplicates)
    for _ in range(600):
        target = voter_df.sample(1).iloc[0]
        fn1, ln1 = target["First_Name"], target["Last_Name"]
        addr1, dob1 = target["Address"], target["Date_Of_Birth"]
        
        # Perturbed duplicate representation
        if fn1 in NICKNAMES and random.random() < 0.4:
            fn2 = random.choice(NICKNAMES[fn1])
        elif random.random() < 0.5:
            fn2 = inject_ocr_typo(fn1)
        else:
            fn2 = fn1
            
        ln2 = inject_ocr_typo(ln1) if random.random() < 0.4 else ln1
        addr2 = addr1.replace(" St", " Street").replace(" Ave", " Avenue") if random.random() < 0.5 else addr1
        dob2 = dob1
        
        pairs.append({
            "Name_1": f"{fn1} {ln1}",
            "Name_2": f"{fn2} {ln2}",
            "Address_1": addr1,
            "Address_2": addr2,
            "DOB_1": dob1,
            "DOB_2": dob2,
            "Is_Match": 1
        })
        
    # 600 Negative match pairs (different people, some sharing family name or address)
    for _ in range(600):
        t1, t2 = voter_df.sample(2).iloc
        hard_negative = random.random() < 0.4
        if hard_negative:
            # Sibling / same household negative: same last name, same address, different first name
            fn2 = random.choice([f for f in FIRST_NAMES if f != t1["First_Name"]])
            pairs.append({
                "Name_1": f"{t1['First_Name']} {t1['Last_Name']}",
                "Name_2": f"{fn2} {t1['Last_Name']}",
                "Address_1": t1["Address"],
                "Address_2": t1["Address"],
                "DOB_1": t1["Date_Of_Birth"],
                "DOB_2": f"19{random.randint(50, 99)}-05-15",
                "Is_Match": 0
            })
        else:
            pairs.append({
                "Name_1": f"{t1['First_Name']} {t1['Last_Name']}",
                "Name_2": f"{t2['First_Name']} {t2['Last_Name']}",
                "Address_1": t1["Address"],
                "Address_2": t2["Address"],
                "DOB_1": t1["Date_Of_Birth"],
                "DOB_2": t2["Date_Of_Birth"],
                "Is_Match": 0
            })
            
    pair_df = pd.DataFrame(pairs)
    pair_csv = data_dir / "voter_training_pairs.csv"
    pair_df.to_csv(pair_csv, index=False)
    print(f"Saved {len(pair_df)} pairwise training records to {pair_csv}")
    print("Voter data generation completed successfully!")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    generate_voter_datasets(str(current_dir))
