"""Conservative deterministic near-duplicate filtering over normalized word shingles."""
from collections import defaultdict
import hashlib
import numpy as np


def signature(normalized):
    words=normalized.split()
    if len(words)<20:return None
    shingles={' '.join(words[i:i+5]) for i in range(len(words)-4)}
    values=np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little') for s in shingles),dtype='<u8')
    bits=np.unpackbits(values.view(np.uint8).reshape(-1,8),axis=1,bitorder='little')
    voted=(bits.sum(axis=0)*2>=len(values)).astype(np.uint8)
    return int.from_bytes(np.packbits(voted,bitorder='little').tobytes(),'little')


class NearDedup:
    """Four 16-bit bands find all 64-bit signatures within Hamming distance 3."""
    def __init__(self):
        self.bands=[defaultdict(list) for _ in range(4)]
        self.signatures=[]

    def duplicate(self,normalized):
        sig=signature(normalized)
        if sig is None:return False
        candidates=set()
        for i,band in enumerate(self.bands):
            candidates.update(band[(sig>>(i*16))&65535])
        if any((sig^self.signatures[idx]).bit_count()<=3 for idx in candidates):return True
        idx=len(self.signatures);self.signatures.append(sig)
        for i,band in enumerate(self.bands):band[(sig>>(i*16))&65535].append(idx)
        return False
