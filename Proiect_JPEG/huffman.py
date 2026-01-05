import heapq
import numpy as np
from collections import Counter, namedtuple

ZIGZAG_ORDER = np.array([
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
])

def zigzag_flatten(block):
    return block.ravel()[ZIGZAG_ORDER]

def inverse_zigzag(flat_block):
    block_flat = np.zeros(64, dtype=flat_block.dtype)
    block_flat[ZIGZAG_ORDER] = flat_block
    return block_flat.reshape(8, 8)


class HuffmanNode:
    def __init__(self, freq, value, left=None, right=None):
        self.freq = freq
        self.value = value
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoder:
    def __init__(self):
        self.tree_root = None
        self.coduri = {} 
        self.coduri_reverse = {}

    def _build_tree(self, data):

        counts = Counter(data)
        heap = [HuffmanNode(freq, val, None, None) for val, freq in counts.items()]
        heapq.heapify(heap)

        if len(heap) == 0:
            return None
            
        if len(heap) == 1:
            #In caz ca exista doar un simbol (ex: imagine complet neagra)
            node = heap[0]
            return HuffmanNode(node.freq, node.value, None, None)

        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = HuffmanNode(node1.freq + node2.freq, None, node1, node2)
            heapq.heappush(heap, merged)
        
        return heap[0]

    def _generate_codes(self, node, prefix=""):
        """Parcurgem arborele si daca e un nod de legatura (Care mereu nu are valoare)
            continuam pana dam de o frunza ca sa construim dictionarele"""


        if node is None:
            return

        if node.value is not None:
            self.codebook[node.value] = prefix or "0"
            self.reverse_mapping[prefix or "0"] = node.value
        else:
            self._generate_codes(node.left, prefix + "0")
            self._generate_codes(node.right, prefix + "1")

    def compress(self, data_array):

        if len(data_array) == 0:
            return "", {}

        self.codebook = {}
        self.reverse_mapping = {}
        
        self.tree_root = self._build_tree(data_array)
        self._generate_codes(self.tree_root)
        
        bitstream = "".join([self.codebook[val] for val in data_array])
        
        return bitstream, self.codebook

    def decompress(self, bitstream, dictionar):
        rev_map = {v: k for k, v in dictionar.items()}
        
        decoded_data = []
        current_code = ""
        
        for bit in bitstream:
            current_code += bit
            if current_code in rev_map:
                decoded_data.append(rev_map[current_code])
                current_code = ""
                
        return np.array(decoded_data)