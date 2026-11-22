"""Summarize paired, already-judged polarities for one trait under one prompt.

A neutral/absent/unclear trait is 0, never inferred as negative. Call separately
for each trait and task; do not count the three correlated labels or prompts as
independent images. This does not turn automatic text matching into a blind review.
"""
from collections import Counter


def summarize(rows):
    by_class = {key: [] for key in ('bird', 'nonbird')}
    for row in rows:
        if any(row[model] not in (-1,0,1) for model in ('base','adapter')):
            raise ValueError('expected signed direction or zero')
        by_class[row['class']].append(row)
    if not all(by_class.values()):
        raise ValueError('both classes required')
    result = {}
    for model in ('base', 'adapter'):
        rates = []
        result[model] = {}
        for cls, subset in by_class.items():
            counts = Counter(r[model] for r in subset)
            rates.append(counts[1 if cls=='bird' else -1]/len(subset))
            result[model][cls] = {'positive':counts[1], 'negative':counts[-1], 'unassigned_or_neutral':counts[0], 'n':len(subset)}
        result[model]['balanced_endpoint_accuracy'] = sum(rates)/2
    result['paired'] = {cls:{key+'_gain':sum(int(r['adapter']==val)-int(r['base']==val) for r in subset) for key,val in [('positive',1),('negative',-1)]} for cls,subset in by_class.items()}
    return result
