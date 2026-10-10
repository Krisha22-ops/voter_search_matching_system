"""
Machine Learning Record Linkage & Matcher
==========================================
Extracts multi-dimensional similarity features and trains a Random Forest classifier
to output calibrated match probabilities (0.0 to 1.0) for voter identity resolution.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from similarity_algorithms import (
    levenshtein_similarity,
    jaro_winkler_similarity,
    token_sort_similarity,
    soundex,
    jaccard_token_similarity
)


def extract_pairwise_features(
    name1: str, name2: str,
    addr1: str = "", addr2: str = "",
    dob1: str = "", dob2: str = ""
) -> List[float]:
    """
    Constructs an 8-dimensional similarity feature vector between two records.
    """
    # 1. Full name metrics
    lev_sim = levenshtein_similarity(name1, name2)
    jw_sim = jaro_winkler_similarity(name1, name2)
    tok_sort = token_sort_similarity(name1, name2)
    
    # 2. Tokenized first/last name comparison
    parts1 = name1.strip().split()
    parts2 = name2.strip().split()
    fn1 = parts1[0] if parts1 else ""
    ln1 = parts1[-1] if len(parts1) > 1 else ""
    fn2 = parts2[0] if parts2 else ""
    ln2 = parts2[-1] if len(parts2) > 1 else ""
    
    fn_jw = jaro_winkler_similarity(fn1, fn2) if fn1 and fn2 else 0.0
    ln_jw = jaro_winkler_similarity(ln1, ln2) if ln1 and ln2 else 0.0
    
    # 3. Phonetic Soundex match
    soundex_fn_match = 1.0 if (fn1 and fn2 and soundex(fn1) == soundex(fn2)) else 0.0
    soundex_ln_match = 1.0 if (ln1 and ln2 and soundex(ln1) == soundex(ln2)) else 0.0
    
    # 4. Address similarity
    addr_sim = jaccard_token_similarity(addr1, addr2) if (addr1 and addr2) else 0.5
    
    # 5. Date of Birth matching
    dob_exact = 1.0 if (dob1 and dob2 and dob1.strip() == dob2.strip()) else 0.0
    
    return [
        lev_sim,
        jw_sim,
        tok_sort,
        fn_jw,
        ln_jw,
        (soundex_fn_match + soundex_ln_match) / 2.0,
        addr_sim,
        dob_exact
    ]


class VoterMLMatcher:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            random_state=42
        )
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train_from_csv(self, pairs_csv: str, output_model_path: str = None):
        """Trains the Random Forest model on labeled pairwise data."""
        print(f"Loading training pairs from {pairs_csv}...")
        df = pd.read_csv(pairs_csv)
        
        X_features = []
        y_labels = df["Is_Match"].values
        
        for _, row in df.iterrows():
            feat = extract_pairwise_features(
                str(row["Name_1"]), str(row["Name_2"]),
                str(row["Address_1"]), str(row["Address_2"]),
                str(row["DOB_1"]), str(row["DOB_2"])
            )
            X_features.append(feat)
            
        X = np.array(X_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
        
        print(f"Training Random Forest on {len(X_train)} pairwise feature vectors...")
        self.clf.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        
        print("\n--- ML Matcher Validation Performance ---")
        print(classification_report(y_test, y_pred, target_names=["Non-Match", "Match"], digits=4))
        print(f"ROC-AUC Score: {auc:.4f}")
        
        # Feature Importances
        feature_names = ["Levenshtein", "Jaro-Winkler", "Token-Sort", "First-Name-JW", "Last-Name-JW", "Phonetic-Soundex", "Address-Jaccard", "DOB-Match"]
        importances = self.clf.feature_importances_
        print("\nFeature Importances:")
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {name.ljust(18)}: {imp*100:5.2f}%")
            
        if output_model_path:
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            joblib.dump(self.clf, output_model_path)
            print(f"Model successfully saved to {output_model_path}")
            
    def predict_match(self, name1: str, name2: str, addr1: str = "", addr2: str = "", dob1: str = "", dob2: str = "") -> Tuple[int, float]:
        """
        Returns (predicted_class, match_probability).
        """
        feats = extract_pairwise_features(name1, name2, addr1, addr2, dob1, dob2)
        X = np.array([feats])
        prob = float(self.clf.predict_proba(X)[0][1])
        pred = 1 if prob >= 0.50 else 0
        return pred, prob

    def load(self, model_path: str):
        self.clf = joblib.load(model_path)
        self.is_trained = True


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent
    pairs_file = base_dir / "data" / "voter_training_pairs.csv"
    save_file = base_dir / "models" / "voter_matcher_rf.joblib"
    
    matcher = VoterMLMatcher()
    matcher.train_from_csv(str(pairs_file), str(save_file))
