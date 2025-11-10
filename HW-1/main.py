"""
Text Similarity
Implements preprocessing, shingling, minhashing, and LSH for document similarity
"""

import re
import os
import glob
import random
import hashlib
from typing import List, Set, Dict, Tuple
from collections import defaultdict


class TextPreprocessor:
    """Preprocesses text by removing spaces, symbols, and normalizing"""
    
    def __init__(self, remove_spaces=True, remove_symbols=True, lowercase=True):
        """
        Initialize preprocessor
        
        Args:
            remove_spaces: Whether to remove all spaces
            remove_symbols: Whether to remove punctuation and symbols
            lowercase: Whether to convert to lowercase
        """
        self.remove_spaces = remove_spaces
        self.remove_symbols = remove_symbols
        self.lowercase = lowercase
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess a text string
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        processed = text
        
        # Convert to lowercase
        if self.lowercase:
            processed = processed.lower()
        
        # Remove symbols and punctuation
        if self.remove_symbols:
            # Keep only alphanumeric characters
            processed = re.sub(r'[^a-z0-9\s]', '', processed)
        
        # # Remove spaces
        if self.remove_spaces:
            processed = re.sub(r'\s+', '', processed)
        else:
            # Normalize whitespace
            processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def preprocess_file(self, filepath: str) -> str:
        """
        Preprocess text from a file
        
        Args:
            filepath: Path to text file
            
        Returns:
            Preprocessed text string
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.preprocess(text)


class Shingler:
    """Creates k-shingles from preprocessed text"""
    
    def __init__(self, k: int = 5):
        """
        Initialize shingler
        
        Args:
            k: Size of shingles (k-grams)
        """
        self.k = k
    
    def create_shingles(self, text: str) -> Set[str]:
        """
        Create k-shingles from text
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Set of k-shingles (strings of length k)
        """
        if len(text) < self.k:
            return {text}  # Return the text itself if shorter than k
        
        shingles = set()
        for i in range(len(text) - self.k + 1):
            shingle = text[i:i + self.k]
            shingles.add(shingle)
        
        return shingles
    
    def create_shingle_hashes(self, text: str) -> Set[int]:
        """
        Create hashed k-shingles (more memory efficient)
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Set of hashed shingle values
        """
        shingles = self.create_shingles(text)
        return {self._hash_shingle(shingle) for shingle in shingles}
    
    def _hash_shingle(self, shingle: str) -> int:
        """Hash a shingle to an integer"""
        return int(hashlib.md5(shingle.encode('utf-8')).hexdigest(), 16)


class SetComparator:
    """Compares sets using Jaccard similarity"""
    
    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """
        Calculate Jaccard similarity between two sets
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity (intersection / union)
        """
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def compare_all_pairs(sets: Dict[str, Set]) -> Dict[Tuple[str, str], float]:
        """
        Compare all pairs of sets
        
        Args:
            sets: Dictionary mapping document names to their sets
            
        Returns:
            Dictionary mapping (doc1, doc2) pairs to their Jaccard similarity
        """
        results = {}
        doc_names = list(sets.keys())
        
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                doc1 = doc_names[i]
                doc2 = doc_names[j]
                similarity = SetComparator.jaccard_similarity(
                    sets[doc1], sets[doc2]
                )
                pair = tuple(sorted((doc1, doc2)))
                results[pair] = similarity
        
        return results


class MinHasher:
    """Creates MinHash signatures for sets"""
    
    def __init__(self, num_hash_functions: int = 100, seed: int = 42):
        """
        Initialize MinHasher
        
        Args:
            num_hash_functions: Number of hash functions to use
            seed: Random seed for reproducibility
        """
        self.num_hash_functions = num_hash_functions
        self.hash_functions = self._generate_hash_functions(seed)
    
    def _generate_hash_functions(self, seed: int) -> List[Tuple[int, int]]:
        """
        Generate hash functions of form (ax + b) mod p
        
        Args:
            seed: Random seed
            
        Returns:
            List of (a, b) tuples for hash functions
        """
        random.seed(seed)
        # Use a large prime number
        self.p = 2147483647  # 2^31 - 1
        
        hash_functions = []
        for _ in range(self.num_hash_functions):
            a = random.randint(1, self.p - 1)
            b = random.randint(0, self.p - 1)
            hash_functions.append((a, b))
        
        return hash_functions
    
    def _hash(self, x: int, hash_func: Tuple[int, int]) -> int:
        """
        Apply hash function to a value
        
        Args:
            x: Value to hash (integer representation of shingle)
            hash_func: (a, b) tuple for hash function
            
        Returns:
            Hashed value
        """
        a, b = hash_func
        return (a * x + b) % self.p
    
    def compute_signature(self, shingle_set: Set[int]) -> List[int]:
        """
        Compute MinHash signature for a set of shingles
        
        Args:
            shingle_set: Set of hashed shingles (integers)
            
        Returns:
            MinHash signature as list of integers
        """
        if not shingle_set:
            return [self.p] * self.num_hash_functions
        
        signature = []
        
        for hash_func in self.hash_functions:
            min_hash = self.p
            for shingle in shingle_set:
                hash_value = self._hash(shingle, hash_func)
                min_hash = min(min_hash, hash_value)
            signature.append(min_hash)
        
        return signature
    
    def compute_signatures(self, shingle_sets: Dict[str, Set[int]]) -> Dict[str, List[int]]:
        """
        Compute MinHash signatures for multiple sets
        
        Args:
            shingle_sets: Dictionary mapping document names to their shingle sets
            
        Returns:
            Dictionary mapping document names to their MinHash signatures
        """
        signatures = {}
        for doc_name, shingles in shingle_sets.items():
            signatures[doc_name] = self.compute_signature(shingles)
        return signatures


class SignatureComparator:
    """Compares MinHash signatures to estimate similarity"""
    
    @staticmethod
    def estimate_similarity(sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity from MinHash signatures
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Estimated Jaccard similarity
        """
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have the same length")
        
        if not sig1:
            return 1.0
        
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)
    
    @staticmethod
    def compare_all_pairs(signatures: Dict[str, List[int]]) -> Dict[Tuple[str, str], float]:
        """
        Compare all pairs of signatures
        
        Args:
            signatures: Dictionary mapping document names to their signatures
            
        Returns:
            Dictionary mapping (doc1, doc2) pairs to estimated similarity
        """
        results = {}
        doc_names = list(signatures.keys())
        
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                doc1 = doc_names[i]
                doc2 = doc_names[j]
                similarity = SignatureComparator.estimate_similarity(
                    signatures[doc1], signatures[doc2]
                )
                pair = tuple(sorted((doc1, doc2)))
                results[pair] = similarity
        
        return results


class LSH:
    """Locality Sensitive Hashing for efficient similarity search"""
    
    def __init__(self, num_bands: int, rows_per_band: int):
        """
        Initialize LSH
        
        Args:
            num_bands: Number of bands to divide signature into
            rows_per_band: Number of hash functions per band
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.bands = defaultdict(list)  # band_hash -> list of document names
    
    def _band_hash(self, signature_band: List[int]) -> int:
        """
        Hash a band of signature to an integer
        
        Args:
            signature_band: A band (subset) of the signature
            
        Returns:
            Hash value for the band
        """
        band_str = ','.join(map(str, signature_band))
        return int(hashlib.md5(band_str.encode('utf-8')).hexdigest(), 16)
    
    def index(self, signatures: Dict[str, List[int]]):
        """
        Index signatures in LSH structure
        
        Args:
            signatures: Dictionary mapping document names to their signatures
        """
        self.bands.clear()
        
        for doc_name, signature in signatures.items():
            # Split signature into bands
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                
                if end <= len(signature):
                    band = signature[start:end]
                    band_hash = self._band_hash(band)
                    
                    # Store in bucket
                    if doc_name not in self.bands[band_hash]:
                        self.bands[band_hash].append(doc_name)
    
    def find_candidates(self, signature: List[int]) -> Set[str]:
        """
        Find candidate similar documents for a given signature
        
        Args:
            signature: MinHash signature to search for
            
        Returns:
            Set of candidate document names
        """
        candidates = set()
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            
            if end <= len(signature):
                band = signature[start:end]
                band_hash = self._band_hash(band)
                
                # Add all documents in this bucket as candidates
                candidates.update(self.bands.get(band_hash, []))
        
        return candidates
    
    def find_all_candidate_pairs(self) -> List[Tuple[str, str]]:
        """
        Find all candidate pairs from the indexed signatures
        
        Returns:
            List of (doc1, doc2) candidate pairs
        """
        candidate_pairs = set()
        
        # For each bucket, all documents in the same bucket are candidates
        for bucket_docs in self.bands.values():
            if len(bucket_docs) > 1:
                # All pairs within a bucket are candidates
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        doc1, doc2 = bucket_docs[i], bucket_docs[j]
                        # Ensure consistent ordering
                        pair = tuple(sorted([doc1, doc2]))
                        candidate_pairs.add(pair)
        
        return sorted(candidate_pairs)
    
    def compute_threshold(self) -> float:
        """
        Compute the similarity threshold for the LSH parameters
        
        Returns:
            Estimated similarity threshold (t ≈ (1/b)^(1/r))
        """
        # Simplified threshold approximation
        # t ≈ (1/b)^(1/r) where b = num_bands, r = rows_per_band
        return (1.0 / self.num_bands) ** (1.0 / self.rows_per_band)


def load_text_files(data_dir: str) -> Dict[str, str]:
    """
    Load all text files from data directory
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary mapping file names to their raw content
    """
    texts = {}
    pattern = os.path.join(data_dir, 'text*.txt')
    
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            texts[filename] = f.read()
    
    return texts


def main():
    """Main function to demonstrate the text similarity pipeline"""
    
    # Configuration: use path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    k_shingle = 5
    num_hash_functions = 100
    num_bands = 20
    rows_per_band = 5
    
    print("=" * 70)
    print("Text Similarity Project")
    print("=" * 70)
    print()
    
    # Step 1: Load text files
    print("Step 1: Loading text files...")
    texts = load_text_files(data_dir)
    print(f"Loaded {len(texts)} text files: {list(texts.keys())}")
    print()
    
    # Step 2: Preprocess texts
    print("Step 2: Preprocessing texts...")
    preprocessor = TextPreprocessor(remove_spaces=False, remove_symbols=False, lowercase=True)
    preprocessed = {}
    for filename, text in texts.items():
        preprocessed[filename] = preprocessor.preprocess(text)
        print(f"  {filename}: {len(preprocessed[filename])} characters after preprocessing")
    print()
    
    # Step 3: Create shingles
    print(f"Step 3: Creating {k_shingle}-shingles...")
    shingler = Shingler(k=k_shingle)
    shingle_sets = {}
    shingle_hash_sets = {}
    for filename, text in preprocessed.items():
        shingles = shingler.create_shingles(text)
        shingle_sets[filename] = shingles
        shingle_hash_sets[filename] = shingler.create_shingle_hashes(text)
        print(f"  {filename}: {len(shingles)} unique shingles")
    print()
    
    # Step 4: Compare sets using Jaccard similarity
    print("Step 4: Comparing sets using Jaccard similarity...")
    set_comparator = SetComparator()
    jaccard_similarities = set_comparator.compare_all_pairs(shingle_sets)
    print("Jaccard Similarities (exact):")
    for (doc1, doc2), sim in sorted(jaccard_similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc1} vs {doc2}: {sim:.4f}")
    print()
    
    # Step 5: MinHashing
    print(f"Step 5: Computing MinHash signatures ({num_hash_functions} hash functions)...")
    minhasher = MinHasher(num_hash_functions=num_hash_functions, seed=42)
    signatures = minhasher.compute_signatures(shingle_hash_sets)
    print(f"Computed signatures for {len(signatures)} documents")
    if signatures:
        print(f"Signature length: {len(list(signatures.values())[0])}")
    else:
        print("Warning: No signatures were computed. Check if shingle_hash_sets is empty.")
    print()
    
    # Step 6: Compare signatures
    print("Step 6: Comparing MinHash signatures...")
    sig_comparator = SignatureComparator()
    estimated_similarities = sig_comparator.compare_all_pairs(signatures)
    print("Estimated Similarities (from signatures):")
    for (doc1, doc2), sim in sorted(estimated_similarities.items(), key=lambda x: x[1], reverse=True):
        # Find corresponding exact Jaccard (with matching (doc1, doc2) order)
        pair = tuple(sorted((doc1, doc2)))
        exact_sim = jaccard_similarities.get(pair, 0.0)
        error = abs(sim - exact_sim)
        print(f"  {doc1} vs {doc2}: {sim:.4f} (exact: {exact_sim:.4f}, error: {error:.4f})")
    print()
    
    # Step 7: LSH
    print(f"Step 7: Building LSH index ({num_bands} bands, {rows_per_band} rows per band)...")
    lsh = LSH(num_bands=num_bands, rows_per_band=rows_per_band)
    lsh.index(signatures)
    threshold = lsh.compute_threshold()
    print(f"LSH similarity threshold: {threshold:.4f}")
    print()
    
    # Find candidate pairs
    candidate_pairs = lsh.find_all_candidate_pairs()
    print(f"Found {len(candidate_pairs)} candidate pairs from LSH:")
    for doc1, doc2 in candidate_pairs:
        pair = tuple(sorted((doc1, doc2)))
        exact_sim = jaccard_similarities.get(pair, 0.0)
        estimated_sim = estimated_similarities.get(pair, 0.0)
        print(f"  {doc1} vs {doc2}: Jaccard={exact_sim:.4f}, Estimated={estimated_sim:.4f}")
    print()
    
    # Compare with all pairs (to see what LSH filtered)
    print("Comparison: LSH candidates vs all pairs")
    all_pairs = set(jaccard_similarities.keys())
    lsh_candidate_set = set(candidate_pairs)
    filtered_out = all_pairs - lsh_candidate_set
    
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  LSH candidates: {len(lsh_candidate_set)}")
    print(f"  Filtered out: {len(filtered_out)}")
    
    if filtered_out:
        print("\n  Filtered out pairs (low similarity):")
        for doc1, doc2 in sorted(filtered_out):
            pair = tuple(sorted((doc1, doc2)))
            exact_sim = jaccard_similarities.get(pair, 0.0)
            print(f"    {doc1} vs {doc2}: {exact_sim:.4f}")
    
    print()
    print("=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
