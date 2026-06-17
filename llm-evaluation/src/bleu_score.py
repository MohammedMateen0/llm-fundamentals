from collections import Counter

reference='''
the cat sitss on the mat'''.split()

candidate='''the cat sits on mat'''.split()

ref_counts=Counter(
    reference
)

cand_counts=Counter(
    candidate
)

overlap=sum(
    min(
        cand_counts[word],
        ref_counts[word]
    )
    for word in cand_counts
)
bleu=overlap/len(candidate)

print(
    f"BLEU-1: {bleu:.4f}"
)