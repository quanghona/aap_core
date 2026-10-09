import re
from typing import Any, ClassVar, List, Literal

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from rbloom import Bloom
from rensa import CMinHash, RMinHash  # type: ignore


class BaseAlgoConfig(BaseModel):
    """Base configuration for deduplication algorithms."""

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom", "suffix_array"
    ] = Field(..., description="Name of the algorithm")


class MinHashAlgoConfig(BaseAlgoConfig):
    """Configuration for C-MinHash based deduplication.

    Uses the **rensa** library for C-MinHash deduplication.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom", "suffix_array"
    ] = "minhash"
    threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0)",
    )
    num_perm: int = Field(
        default=128,
        gt=0,
        description="Number of permutations for MinHash",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class SuffixAlgoConfig(BaseAlgoConfig):
    """Configuration for Suffix Array based deduplication.

    Uses a suffix array with LCP (Longest Common Prefix) array for exact
    substring deduplication. Inspired by Google's deduplicate-text-datasets
    and Chenghao Mou's text-dedup.

    This is a deterministic algorithm with 100% precision for exact substring matches.
    Unlike probabilistic methods (MinHash, SimHash), it guarantees no false positives.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom", "suffix_array"
    ] = "suffix_array"
    min_length: int = Field(
        default=50,
        gt=0,
        description="Minimum substring length (in tokens) to consider as duplicate.",
    )
    max_length: int = Field(
        default=1000,
        gt=0,
        description="Maximum substring length (in tokens) to consider.",
    )


class MinHashLSHAlgoConfig(BaseAlgoConfig):
    """Configuration for MinHash LSH based deduplication.

    Uses the **rensa** library for R-MinHash with Locality-Sensitive Hashing.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom", "suffix_array"
    ] = "minhash_lsh"
    threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0)",
    )
    num_perm: int = Field(
        default=128,
        gt=0,
        description="Number of permutations for MinHash",
    )
    num_bands: int = Field(
        default=16,
        gt=0,
        description="Number of bands for LSH index. Higher = fewer false positives, more false negatives.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class BloomAlgoConfig(BaseAlgoConfig):
    """Configuration for Bloom Filter based deduplication.

    Uses the **rbloom** library for Bloom Filter based exact deduplication.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom", "suffix_array"
    ] = "bloomfilter"
    expected_items: int = Field(
        default=10,
        gt=0,
        description="Expected number of items to add to the bloom filter",
    )
    false_positive_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Desired false positive rate (0.0 to 1.0)",
    )


class SimHashAlgoConfig(BaseAlgoConfig):
    """Configuration for SimHash based deduplication.

    Uses a custom numpy-based SimHash implementation for near-duplicate detection.
    SimHash produces a fingerprint for each document/sentence, and near-duplicates
    have similar fingerprints (low Hamming distance).
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom", "suffix_array"
    ] = "simhash"
    hash_bits: Literal[64, 128] = Field(
        default=64,
        gt=0,
        description="Number of bits in the SimHash fingerprint",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0). "
        "Sentences with similarity >= threshold are considered duplicates.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class LSHBloomAlgoConfig(BaseAlgoConfig):
    """Configuration for LSH-Bloom Filter based deduplication.

    Uses MinHash with Locality-Sensitive Hashing (LSH) where each band's hash table
    is replaced by a Bloom filter for space-efficient near-duplicate detection.
    Inspired by the LSHBloom algorithm from https://arxiv.org/abs/2411.04257
    and the datasketch implementation.

    Each band's Bloom filter stores ``sum(hashvalues) % MersennePrime`` instead of
    all individual hash values, drastically reducing space usage for large-scale
    deduplication.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom", "suffix_array"
    ] = "lsh_bloom"
    threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0)",
    )
    num_perm: int = Field(
        default=128,
        gt=0,
        description="Number of permutations for MinHash",
    )
    num_bands: int = Field(
        default=16,
        gt=0,
        description="Number of bands for LSH index. Higher = fewer false positives, more false negatives.",
    )
    expected_items: int = Field(
        default=10,
        gt=0,
        description="Expected number of items to add to each Bloom filter",
    )
    false_positive_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Desired false positive rate for each Bloom filter (0.0 to 1.0)",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class SimHash(BaseModel):
    """A custom SimHash implementation using numpy.

    SimHash produces a fingerprint for each document/sentence, and near-duplicates
    have similar fingerprints (low Hamming distance). This implementation does not
    require any external dependencies beyond numpy.

    Algorithm:
    1. For each token, compute a hash to get a bit vector
    2. Use the hash to create a weighted contribution
    3. Sum all weighted contributions to get a signature vector
    4. Take the sign of each element to get the final fingerprint
    """

    hash_bits: Literal[64, 128] = Field(
        default=64, gt=0, description="Number of bits in the fingerprint"
    )

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def _token_hash(token: str, hash_bits: int) -> np.ndarray:
        """Compute a hash bit vector for a token.

        Uses the random number generator to create a random bit vector
        based on the token's hash value.
        """
        # Use Python's built-in hash combined with position for stability
        h = hash(token)
        bits = np.zeros(hash_bits, dtype=np.int8)
        for i in range(hash_bits):
            # Use different bits of the hash for each position
            bits[i] = 1 if (h >> (i % 64)) & 1 else -1
        return bits

    def compute_hash(self, text: str) -> np.ndarray:
        """Compute the SimHash fingerprint for a text.

        Args:
            text: The input text to hash.

        Returns:
            A numpy array of int8 values (-1 or 1) representing the fingerprint.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return np.zeros(self.hash_bits, dtype=np.int8)

        # Initialize signature vector
        signature = np.zeros(self.hash_bits, dtype=np.float64)

        for token in tokens:
            # Get hash bits for this token
            token_hash = self._token_hash(token, self.hash_bits)
            # Add weighted contribution to signature
            signature += token_hash.astype(np.float64)

        # Convert to fingerprint: +1 if positive, -1 if negative
        fingerprint = np.sign(signature).astype(np.int8)
        # Handle zeros (treat as +1)
        fingerprint[fingerprint == 0] = 1
        return fingerprint

    @staticmethod
    def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
        """Compute the Hamming distance between two SimHash fingerprints.

        Args:
            hash1: First fingerprint array.
            hash2: Second fingerprint array.

        Returns:
            The Hamming distance (number of differing bits).
        """
        return int(np.sum(hash1 != hash2))

    @staticmethod
    def similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Compute the similarity between two SimHash fingerprints.

        Similarity is computed as 1 - (Hamming distance / hash_bits).

        Args:
            hash1: First fingerprint array.
            hash2: Second fingerprint array.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        distance = SimHash.hamming_distance(hash1, hash2)
        hash_bits = len(hash1)
        return 1.0 - (distance / hash_bits)


class LSHBloom(BaseModel):
    """LSH-Bloom Filter implementation for space-efficient near-duplicate detection.

    This implementation combines MinHash with Locality-Sensitive Hashing (LSH)
    where each band's hash table is replaced by a Bloom filter. This approach
    drastically reduces space usage compared to traditional LSH while maintaining
    good near-duplicate detection capabilities.

    Inspired by the LSHBloom algorithm from https://arxiv.org/abs/2411.04257
    and the datasketch implementation.

    Each band's Bloom filter stores ``sum(hashvalues) % MersennePrime`` instead of
    all individual hash values, making it suitable for large-scale deduplication
    scenarios with millions or billions of documents.

    Attributes:
        num_bands: Number of bands in the LSH index.
        band_size: Number of hash values per band (num_perm / num_bands).
        bloom_filters: List of Bloom filters, one per band.
        num_perm: Number of permutations for MinHash.
        seed: Random seed for reproducibility.
        threshold: Similarity threshold for duplicate detection.
    """

    _MERSENNE_PRIME: ClassVar[int] = (1 << 61) - 1  # Mersenne prime 2^61 - 1

    threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0)",
    )
    num_perm: int = Field(
        default=128, gt=0, description="Number of permutations for MinHash"
    )
    num_bands: int = Field(
        default=16, gt=0, description="Number of bands for LSH index"
    )
    expected_items: int = Field(
        default=10,
        gt=0,
        description="Expected number of items to add to each Bloom filter",
    )
    false_positive_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Desired false positive rate for each Bloom filter (0.0 to 1.0)",
    )

    _band_size: int = PrivateAttr(default=0)
    _bloom_filters: List[Bloom] = PrivateAttr(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("num_bands")
    @classmethod
    def validate_num_bands(cls, v: int, info: Any) -> int:
        """Validate that num_bands is less than num_perm."""
        num_perm = info.data.get("num_perm", 128)
        if v > num_perm:
            raise ValueError("num_bands cannot be greater than num_perm")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived attributes and Bloom filters after model initialization."""
        self._band_size = self.num_perm // self.num_bands
        self._bloom_filters = [
            Bloom(
                expected_items=self.expected_items,
                false_positive_rate=self.false_positive_rate,
            )
            for _ in range(self.num_bands)
        ]

    def _compute_band_hash(self, hashvalues: List[int]) -> int:
        """Compute a single hash value for a band using sum of hashvalues.

        This is the key space optimization: instead of storing all hash values
        in the Bloom filter, we store a single hash of their sum.

        Args:
            hashvalues: List of hash values for a single band.

        Returns:
            A single hash value to store in the Bloom filter.
        """
        return sum(hashvalues) % self._MERSENNE_PRIME

    def insert(self, minhash_key: str, mh: CMinHash | RMinHash) -> None:
        """Insert a MinHash into the LSH-Bloom index.

        Args:
            minhash_key: A unique key for the item (e.g., sentence index).
                        This is kept for API compatibility but not used in storage.
            mh: A MinHash object (CMinHash or RMinHash) with computed hash values.
        """
        # Get the hash values from the MinHash
        hash_values = self._get_minhash_values(mh)

        for band_idx in range(self.num_bands):
            start = band_idx * self._band_size
            end = start + self._band_size
            band_values = hash_values[start:end]
            band_hash = self._compute_band_hash(band_values)
            # Store just the band hash, matching datasketch implementation
            self._bloom_filters[band_idx].add(band_hash)

    def query(self, mh: CMinHash | RMinHash) -> bool:
        """Query whether a MinHash has a potential match in the index.

        Args:
            mh: A MinHash object (CMinHash or RMinHash) with computed hash values.

        Returns:
            True if there's a potential match (possible duplicate), False otherwise.
        """
        hash_values = self._get_minhash_values(mh)

        for band_idx in range(self.num_bands):
            start = band_idx * self._band_size
            end = start + self._band_size
            band_values = hash_values[start:end]
            band_hash = self._compute_band_hash(band_values)

            # Check if band hash exists in the Bloom filter
            # Use 'in' operator since rbloom.Bloom uses __contains__
            if band_hash not in self._bloom_filters[band_idx]:
                return False

        # All bands matched - potential duplicate
        return True

    def clear(self) -> None:
        """Clear all Bloom filters in the index."""
        for bloom_filter in self._bloom_filters:
            bloom_filter.clear()

    @staticmethod
    def _get_minhash_values(mh: CMinHash | RMinHash) -> List[int]:
        """Extract hash values from a MinHash object.

        This method extracts the hash values from either CMinHash or RMinHash
        objects by using their internal hash representation.

        Args:
            mh: A MinHash object (CMinHash or RMinHash).

        Returns:
            List of integer hash values.
        """
        # For rensa's CMinHash and RMinHash, use the digest() method
        if hasattr(mh, "digest") and callable(mh.digest):
            return list(mh.digest())  # type: ignore
        elif hasattr(mh, "h"):
            # CMinHash has 'h' attribute with hash values
            val = mh.h
            if hasattr(val, "tolist"):
                return val.tolist()
            return list(val)
        elif hasattr(mh, "hash_values"):
            # Some MinHash implementations use 'hash_values'
            val = mh.hash_values
            if hasattr(val, "tolist"):
                return val.tolist()
            return list(val)
        else:
            raise ValueError(
                "Unable to extract hash values from MinHash object. "
                "Expected CMinHash or RMinHash from rensa library."
            )


class SuffixArray(BaseModel):
    """Suffix Array with LCP array for exact substring deduplication.

    This implementation uses the SA-IS (Suffix-array Inducing-Sorting) algorithm
    for O(n) suffix array construction, which is optimal for large texts.
    Inspired by Google's deduplicate-text-datasets
    (https://github.com/google-research/deduplicate-text-datasets) and
    Chenghao Mou's text-dedup (https://github.com/ChenghaoMou/text-dedup/).

    The algorithm works as follows:
    1. Convert text to Unicode code points (handles full Unicode range)
    2. Build suffix array using SA-IS in O(n) time
    3. Build LCP array in O(n) time using Kasai's algorithm
    4. Scan LCP array to find repeated substrings >= min_length
    5. Mark duplicate regions and remove them

    Unlike probabilistic methods (MinHash, SimHash), this is deterministic
    with 100% precision for exact substring matches.

    Attributes:
        min_length: Minimum substring length (in characters) to consider as duplicate.
        max_length: Maximum substring length to consider (prevents memory issues).
    """

    min_length: int = Field(
        default=50,
        gt=0,
        description="Minimum substring length (in characters) to consider as duplicate.",
    )
    max_length: int = Field(
        default=1000,
        gt=0,
        description="Maximum substring length (in characters) to consider.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def _build_suffix_array_sais(text: List[int]) -> List[int]:
        """Build suffix array using SA-IS (Suffix-array Inducing-Sorting) algorithm.

        Time complexity: O(n) - linear time construction.
        This is optimal and significantly faster than prefix doubling for large texts.

        The algorithm works by:
        1. Classifying each character position as S-type (smaller) or L-type (larger)
        2. Building a list of "secondary suffixes" that need to be sorted
        3. Inducing sorting to build the final suffix array

        Args:
            text: List of integer code points representing the text.

        Returns:
            Suffix array where sa[i] is the starting position of the i-th
            lexicographically smallest suffix.
        """
        n = len(text)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Add sentinel (value -1) to mark end of string
        # This ensures all suffixes are unique and properly terminated
        text_with_sentinel = text + [-1]
        n += 1

        # Classify each position as S-type (0) or L-type (1)
        # A position i is S-type if text[i] < text[i+1] (or i is the last position)
        # A position i is L-type if text[i] > text[i+1]
        # For equal characters, inherit from the next position
        type_array = [0] * n
        type_array[n - 1] = 1  # Last position is always L-type

        for i in range(n - 2, -1, -1):
            if text_with_sentinel[i] < text_with_sentinel[i + 1]:
                type_array[i] = 0  # S-type
            elif text_with_sentinel[i] > text_with_sentinel[i + 1]:
                type_array[i] = 1  # L-type
            else:
                type_array[i] = type_array[i + 1]

        # Find all LMS (Leftmost Suffix-Matching) positions
        # These are positions where an S-type character is preceded by an L-type character
        lms_positions = []
        for i in range(1, n):
            if type_array[i] == 0 and type_array[i - 1] == 1:
                lms_positions.append(i)

        if not lms_positions:
            # Edge case: all positions are L-type (shouldn't happen with sentinel)
            return list(range(n))

        # Assign names to LMS substrings
        # Two LMS substrings have the same name if they are identical
        lms_names = [-1] * n
        current_name = 0
        lms_names[lms_positions[0]] = 0

        for idx in range(1, len(lms_positions)):
            pos = lms_positions[idx]
            prev_pos = lms_positions[idx - 1]

            # Compare substrings starting at pos and prev_pos
            # Both are LMS positions, so we compare up to the next LMS position
            is_same = True
            max_len = min(
                lms_positions[idx + 1] - pos if idx + 1 < len(lms_positions) else n,
                lms_positions[idx] - prev_pos if idx > 0 else n,
            )

            for j in range(max_len):
                if text_with_sentinel[pos + j] != text_with_sentinel[prev_pos + j]:
                    is_same = False
                    break
                if type_array[pos + j] != type_array[prev_pos + j]:
                    is_same = False
                    break

            if is_same:
                lms_names[pos] = current_name
            else:
                current_name += 1
                lms_names[pos] = current_name

        # Create the LMS substring array for recursive SA-IS
        # Map LMS positions to their names
        num_unique_names = current_name + 1
        lms_substring = [0] * len(lms_positions)
        for i, pos in enumerate(lms_positions):
            lms_substring[i] = lms_names[pos]

        # If all LMS substrings have unique names, we can build SA directly
        # Otherwise, recursively apply SA-IS
        if num_unique_names < len(lms_positions):
            # Recursive step: build SA for the LMS substring
            lms_substring_sa = SuffixArray._build_suffix_array_sais(lms_substring)

            # Create the sorted LMS positions based on the recursive SA
            # lms_substring_sa contains indices into lms_substring
            # which correspond to indices into lms_positions
            # Add bounds check to handle edge cases
            sorted_lms_positions = [
                lms_positions[i]
                for i in lms_substring_sa
                if 0 <= i < len(lms_positions)
            ]
        else:
            # All LMS substrings are unique, sort by name
            sorted_lms_positions = [
                lms_positions[i]
                for i in sorted(
                    range(len(lms_positions)), key=lambda x: lms_names[lms_positions[x]]
                )
            ]

        # Induce sorting: place LMS positions in correct order
        # Then place all other suffixes in correct positions
        sa = [-1] * n

        # Place LMS suffixes in their correct positions
        # First, find the end positions for each character
        bucket_end = [0] * (max(text_with_sentinel) + 2)
        for i in range(n):
            bucket_end[text_with_sentinel[i] + 1] += 1

        # Compute cumulative counts (end positions)
        for i in range(1, len(bucket_end)):
            bucket_end[i] += bucket_end[i - 1]
        bucket_end = [x - 1 for x in bucket_end]

        # Place LMS suffixes from right to left (L-type)
        for pos in reversed(sorted_lms_positions):
            c = text_with_sentinel[pos]
            if bucket_end[c] >= 0:
                sa[bucket_end[c]] = pos
                bucket_end[c] -= 1

        # Update bucket_end for L-type inducing (left to right)
        bucket_start = [0] * len(bucket_end)
        bucket_start[0] = 0
        for i in range(1, len(bucket_end)):
            bucket_start[i] = bucket_end[i - 1] + 1

        # Place LMS suffixes from left to right (S-type)
        for pos in sorted_lms_positions:
            c = text_with_sentinel[pos]
            if bucket_start[c] < n:
                sa[bucket_start[c]] = pos
                bucket_start[c] += 1

        # Induce sorting for non-LMS suffixes
        # L-type inducing (right to left)
        for i in range(n - 1, 0, -1):
            pos = sa[i]
            if pos > 0 and type_array[pos - 1] == 1:
                c = text_with_sentinel[pos - 1]
                if bucket_start[c] < n:
                    sa[bucket_start[c]] = pos - 1
                    bucket_start[c] += 1

        # Update bucket_start for S-type inducing (left to right)
        bucket_end = [0] * len(bucket_end)
        for i in range(n):
            bucket_end[text_with_sentinel[i] + 1] += 1
        for i in range(1, len(bucket_end)):
            bucket_end[i] += bucket_end[i - 1]
        bucket_end = [x - 1 for x in bucket_end]

        # S-type inducing (left to right)
        for i in range(n):
            pos = sa[i]
            if pos > 0 and type_array[pos - 1] == 0:
                c = text_with_sentinel[pos - 1]
                if bucket_end[c] >= 0:
                    sa[bucket_end[c]] = pos - 1
                    bucket_end[c] -= 1

        # Remove sentinel position from suffix array
        sa = [pos for pos in sa if pos != n - 1]

        return sa

    @staticmethod
    def _build_lcp_array(tokens: List[int], sa: List[int]) -> List[int]:
        """Build LCP array using Kasai's algorithm.

        Time complexity: O(n)

        Args:
            tokens: List of integer token IDs.
            sa: Suffix array.

        Returns:
            LCP array where lcp[i] is the length of the longest common prefix
            between suffix sa[i-1] and sa[i]. lcp[0] is always 0.
        """
        n = len(tokens)
        if n == 0:
            return []

        # Compute inverse suffix array (rank array)
        rank = [0] * n
        for i in range(n):
            rank[sa[i]] = i

        lcp = [0] * n
        h = 0
        for i in range(n):
            if rank[i] > 0:
                j = sa[rank[i] - 1]  # Previous suffix in sorted order
                # Extend common prefix
                while i + h < n and j + h < n and tokens[i + h] == tokens[j + h]:
                    h += 1
                lcp[rank[i]] = h
                if h > 0:
                    h -= 1
            # lcp[0] remains 0

        return lcp

    def find_duplicates(self, sentences: List[str]) -> bytearray:
        """Find duplicate character positions in the text.

        This is a convenience method that handles the full pipeline:
        1. Joins sentences with space separator
        2. Converts text to Unicode code points with case folding
        3. Builds the suffix array using SA-IS in O(n) time
        4. Builds LCP array in O(n) time using Kasai's algorithm
        5. Scans the LCP array for duplicate regions >= min_length
        6. Returns a bytearray where position i is 1 if duplicate, 0 otherwise

        The method uses Unicode-aware case folding (via str.casefold()) for
        case-insensitive matching, which correctly handles characters from
        all Unicode scripts including Latin, Greek, Cyrillic, Arabic, CJK,
        emoji, and other scripts.

        Uses bytearray instead of set for:
        - Memory efficiency: 1 byte per position vs. ~28 bytes per int in set
        - Faster membership: direct index access vs. hash lookup
        - Better cache locality for sequential scans

        Args:
            sentences: List of sentence strings to deduplicate.

        Returns:
            Bytearray where duplicate_positions[i] == 1 means position i is a duplicate.
        """
        if not sentences:
            return bytearray()

        # Join sentences with space separator (matches caller's " ".join(sentences))
        text = " ".join(sentences)
        if not text:
            return bytearray()

        # Use Unicode-aware case folding for case-insensitive matching
        # casefold() is more aggressive than lower() and handles Unicode correctly
        # e.g., German ß → ss, Turkish İ/i, Greek σ/ς, etc.
        folded_text = text.casefold()

        # Convert characters to Unicode code points for the suffix array
        # This handles all Unicode characters including emoji, CJK, etc.
        tokens = [ord(c) for c in folded_text]
        n = len(tokens)
        if n == 0:
            return bytearray()

        sa = self._build_suffix_array_sais(tokens)
        lcp = self._build_lcp_array(tokens, sa)

        # Use bytearray for memory-efficient duplicate tracking
        # Each position is 0 (not duplicate) or 1 (duplicate)
        duplicate_positions = bytearray(n)

        # Scan LCP array to find duplicate regions
        # A duplicate exists when LCP[i] >= min_length
        for i in range(1, n):
            if lcp[i] >= self.min_length:
                # Found a duplicate region between sa[i-1] and sa[i]
                # Mark all positions in both occurrences
                # Cap at max_length to avoid marking too much
                mark_len = min(lcp[i], self.max_length)

                for j in range(mark_len):
                    pos1 = sa[i - 1] + j
                    pos2 = sa[i] + j
                    if pos1 < n:
                        duplicate_positions[pos1] = 1
                    if pos2 < n:
                        duplicate_positions[pos2] = 1

        return duplicate_positions
