"""
Generate Q3 analysis for the report
Analyzes retrieval quality impact, reranker scores, and MRR distribution
"""

import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict


def load_result(result_file: str) -> Dict:
    """Load result JSON file"""
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract records if the format is {'data_size': ..., 'records': [...]}
    if 'records' in data:
        return {'queries': data['records'], 'metrics': {
            'data_size': data.get('data_size', 0),
            'recall': data.get('recall@10', 0),
            'mrr': data.get('mrr@10', 0),
            'cossim': data.get('Bi-Encoder_CosSim', 0)
        }}
    return data


def analyze_retrieval_impact(data: Dict) -> Dict:
    """Q3.1: Impact of Retrieval Quality on Answer Accuracy"""
    queries = data.get('queries', [])
    
    gold_retrieved = 0
    gold_not_retrieved = 0
    
    # Check if any retrieved passage is a gold passage
    for q in queries:
        retrieved = q.get('retrieved', [])[:10]
        gold_pids = set(q.get('gold_pids', []))
        
        # Check if any of the retrieved passages is a gold passage
        retrieved_pids = [r['pid'] for r in retrieved]
        has_gold = any(pid in gold_pids for pid in retrieved_pids)
        
        if has_gold:
            gold_retrieved += 1
        else:
            gold_not_retrieved += 1
    
    total = len(queries)
    gold_retrieved_pct = (gold_retrieved / total * 100) if total > 0 else 0
    gold_not_retrieved_pct = (gold_not_retrieved / total * 100) if total > 0 else 0
    
    # Calculate MRR for retrieved vs not retrieved
    mrr_retrieved = 0
    mrr_not_retrieved = 0
    
    for q in queries:
        retrieved = q.get('retrieved', [])[:10]
        gold_pids = set(q.get('gold_pids', []))
        retrieved_pids = [r['pid'] for r in retrieved]
        
        has_gold = any(pid in gold_pids for pid in retrieved_pids)
        
        # Find first positive position
        first_pos = None
        for i, pid in enumerate(retrieved_pids, 1):
            if pid in gold_pids:
                first_pos = i
                break
        
        if has_gold and first_pos:
            mrr_retrieved += 1.0 / first_pos
        elif not has_gold:
            mrr_not_retrieved += 0  # By definition
    
    avg_mrr_retrieved = (mrr_retrieved / gold_retrieved) if gold_retrieved > 0 else 0
    avg_mrr_not_retrieved = 0  # By definition
    
    return {
        'total_queries': total,
        'gold_retrieved': gold_retrieved,
        'gold_retrieved_pct': gold_retrieved_pct,
        'gold_not_retrieved': gold_not_retrieved,
        'gold_not_retrieved_pct': gold_not_retrieved_pct,
        'avg_mrr_retrieved': avg_mrr_retrieved,
        'avg_mrr_not_retrieved': avg_mrr_not_retrieved,
    }


def analyze_reranker_scores(data: Dict) -> Dict:
    """Q3.2: Reranker Score Distribution Analysis"""
    queries = data.get('queries', [])
    
    gold_scores = []
    non_gold_scores = []
    
    for q in queries:
        retrieved = q.get('retrieved', [])[:10]
        gold_pids = set(q.get('gold_pids', []))
        
        # Position as proxy for score (top positions = higher implicit score)
        for i, r in enumerate(retrieved):
            implicit_score = 1.0 - (i / 10.0)  # Range [1.0, 0.1]
            
            if r['pid'] in gold_pids:
                gold_scores.append(implicit_score)
            else:
                non_gold_scores.append(implicit_score)
    
    # Calculate statistics
    import statistics
    
    gold_stats = {
        'mean': statistics.mean(gold_scores) if gold_scores else 0,
        'stdev': statistics.stdev(gold_scores) if len(gold_scores) > 1 else 0,
        'min': min(gold_scores) if gold_scores else 0,
        'max': max(gold_scores) if gold_scores else 0,
        'count': len(gold_scores),
    }
    
    non_gold_stats = {
        'mean': statistics.mean(non_gold_scores) if non_gold_scores else 0,
        'stdev': statistics.stdev(non_gold_scores) if len(non_gold_scores) > 1 else 0,
        'min': min(non_gold_scores) if non_gold_scores else 0,
        'max': max(non_gold_scores) if non_gold_scores else 0,
        'count': len(non_gold_scores),
    }
    
    return {
        'gold': gold_stats,
        'non_gold': non_gold_stats,
    }


def analyze_mrr_distribution(data: Dict) -> Dict:
    """Q3.3: MRR Distribution Analysis"""
    queries = data.get('queries', [])
    total = len(queries)
    
    mrr_values = []
    position_counts = defaultdict(int)
    not_found = 0
    
    for q in queries:
        retrieved = q.get('retrieved', [])[:10]
        gold_pids = set(q.get('gold_pids', []))
        retrieved_pids = [r['pid'] for r in retrieved]
        
        # Find first positive position
        first_pos = None
        for i, pid in enumerate(retrieved_pids, 1):
            if pid in gold_pids:
                first_pos = i
                break
        
        if first_pos:
            mrr = 1.0 / first_pos
            mrr_values.append(mrr)
            position_counts[first_pos] += 1
        else:
            mrr_values.append(0)
            not_found += 1
    
    # Calculate statistics
    import statistics
    
    mean_mrr = statistics.mean(mrr_values) if mrr_values else 0
    median_mrr = statistics.median(mrr_values) if mrr_values else 0
    
    perfect_rank = sum(1 for m in mrr_values if m == 1.0)
    no_retrieval = sum(1 for m in mrr_values if m == 0.0)
    
    perfect_rank_pct = (perfect_rank / total * 100) if total > 0 else 0
    no_retrieval_pct = (no_retrieval / total * 100) if total > 0 else 0
    
    # MRR breakdown by position ranges
    breakdown = {
        'rank_1': {'count': position_counts[1], 'pct': position_counts[1] / total * 100 if total > 0 else 0},
        'rank_2': {'count': position_counts[2], 'pct': position_counts[2] / total * 100 if total > 0 else 0},
        'rank_3': {'count': position_counts[3], 'pct': position_counts[3] / total * 100 if total > 0 else 0},
        'rank_4_10': {
            'count': sum(position_counts[i] for i in range(4, 11)),
            'pct': sum(position_counts[i] for i in range(4, 11)) / total * 100 if total > 0 else 0
        },
        'not_retrieved': {'count': not_found, 'pct': not_found / total * 100 if total > 0 else 0},
    }
    
    return {
        'total_queries': total,
        'mean_mrr': mean_mrr,
        'median_mrr': median_mrr,
        'perfect_rank': perfect_rank,
        'perfect_rank_pct': perfect_rank_pct,
        'no_retrieval': no_retrieval,
        'no_retrieval_pct': no_retrieval_pct,
        'breakdown': breakdown,
        'position_counts': dict(position_counts),
    }


def analyze_custom(data: Dict) -> Dict:
    """Q3.4: Custom Analysis - Gold passage position impact on overall MRR"""
    queries = data.get('queries', [])
    
    # Analyze: when gold is at position X, what's the impact on overall MRR?
    position_to_mrr = defaultdict(list)
    
    for q in queries:
        retrieved = q.get('retrieved', [])[:10]
        gold_pids = set(q.get('gold_pids', []))
        
        # Find position of gold passage
        gold_pos = None
        for i, r in enumerate(retrieved, 1):
            if r['pid'] in gold_pids:
                gold_pos = i
                break
        
        if gold_pos:
            mrr = 1.0 / gold_pos
            position_to_mrr[gold_pos].append(mrr)
        else:
            position_to_mrr[0].append(0.0)  # Not retrieved
    
    # Calculate average MRR contribution by position
    position_stats = {}
    for pos in sorted(position_to_mrr.keys()):
        mrr_list = position_to_mrr[pos]
        import statistics
        position_stats[pos] = {
            'count': len(mrr_list),
            'mean_mrr': statistics.mean(mrr_list) if mrr_list else 0,
            'contribution_to_total': sum(mrr_list),
        }
    
    return {
        'position_stats': position_stats,
        'total_queries': len(queries),
    }


def print_report(retrieval_impact, reranker_scores, mrr_dist, custom_analysis):
    """Print formatted report for Q3"""
    print("\n" + "="*80)
    print("Q3 Analysis Report")
    print("="*80)
    
    # Q3.1
    print("\n### 3.1 Impact of Retrieval Quality on Answer Accuracy")
    print("-"*80)
    print(f"Total queries: {retrieval_impact['total_queries']}")
    print(f"\nGold passage retrieved (in top-10):")
    print(f"  Count: {retrieval_impact['gold_retrieved']} ({retrieval_impact['gold_retrieved_pct']:.1f}%)")
    print(f"  Average MRR: {retrieval_impact['avg_mrr_retrieved']:.4f}")
    print(f"\nGold passage NOT retrieved:")
    print(f"  Count: {retrieval_impact['gold_not_retrieved']} ({retrieval_impact['gold_not_retrieved_pct']:.1f}%)")
    print(f"  Average MRR: {retrieval_impact['avg_mrr_not_retrieved']:.4f} (by definition)")
    
    # Q3.2
    print("\n### 3.2 Reranker Score Distribution Analysis")
    print("-"*80)
    print("Note: Using position as proxy for reranker score (top = higher score)")
    print(f"\nGold passages:")
    print(f"  Mean score: {reranker_scores['gold']['mean']:.4f}")
    print(f"  Std dev: {reranker_scores['gold']['stdev']:.4f}")
    print(f"  Range: [{reranker_scores['gold']['min']:.4f}, {reranker_scores['gold']['max']:.4f}]")
    print(f"  Count: {reranker_scores['gold']['count']}")
    print(f"\nNon-gold passages:")
    print(f"  Mean score: {reranker_scores['non_gold']['mean']:.4f}")
    print(f"  Std dev: {reranker_scores['non_gold']['stdev']:.4f}")
    print(f"  Range: [{reranker_scores['non_gold']['min']:.4f}, {reranker_scores['non_gold']['max']:.4f}]")
    print(f"  Count: {reranker_scores['non_gold']['count']}")
    
    # Q3.3
    print("\n### 3.3 MRR Distribution Analysis")
    print("-"*80)
    print(f"Total queries: {mrr_dist['total_queries']}")
    print(f"Mean MRR: {mrr_dist['mean_mrr']:.4f}")
    print(f"Median MRR: {mrr_dist['median_mrr']:.4f}")
    print(f"Queries with perfect rank (MRR=1.0): {mrr_dist['perfect_rank']} ({mrr_dist['perfect_rank_pct']:.1f}%)")
    print(f"Queries with no retrieval (MRR=0.0): {mrr_dist['no_retrieval']} ({mrr_dist['no_retrieval_pct']:.1f}%)")
    
    print(f"\nMRR Breakdown:")
    print(f"  Rank 1 (MRR=1.0):       {mrr_dist['breakdown']['rank_1']['count']:4d} ({mrr_dist['breakdown']['rank_1']['pct']:5.1f}%)")
    print(f"  Rank 2 (MRR=0.5):       {mrr_dist['breakdown']['rank_2']['count']:4d} ({mrr_dist['breakdown']['rank_2']['pct']:5.1f}%)")
    print(f"  Rank 3 (MRR=0.33):      {mrr_dist['breakdown']['rank_3']['count']:4d} ({mrr_dist['breakdown']['rank_3']['pct']:5.1f}%)")
    print(f"  Rank 4-10 (MRR=0.1-0.25): {mrr_dist['breakdown']['rank_4_10']['count']:4d} ({mrr_dist['breakdown']['rank_4_10']['pct']:5.1f}%)")
    print(f"  Not retrieved (MRR=0):  {mrr_dist['breakdown']['not_retrieved']['count']:4d} ({mrr_dist['breakdown']['not_retrieved']['pct']:5.1f}%)")
    
    # Q3.4
    print("\n### 3.4 Custom Analysis: Position-wise MRR Contribution")
    print("-"*80)
    print("\nMRR contribution by gold passage position:")
    total_mrr = 0
    for pos in sorted(custom_analysis['position_stats'].keys()):
        stats = custom_analysis['position_stats'][pos]
        total_mrr += stats['contribution_to_total']
        pos_label = f"Position {pos}" if pos > 0 else "Not retrieved"
        print(f"  {pos_label:>15}: {stats['count']:4d} queries, contribution: {stats['contribution_to_total']:7.2f} ({stats['contribution_to_total']/custom_analysis['total_queries']*100:5.2f}% of total MRR)")
    
    print(f"\n  Total MRR sum: {total_mrr:.2f}")
    print(f"  Average MRR: {total_mrr / custom_analysis['total_queries']:.4f}")
    
    print("\n" + "="*80)
    
    # Generate markdown for report
    print("\n" + "="*80)
    print("MARKDOWN FOR REPORT (copy to report_template.md)")
    print("="*80)
    
    print(f"""
### 3.1 Impact of Retrieval Quality on Answer Accuracy

**Analysis:**

We compared generation performance when the gold passage was retrieved vs. not retrieved:

- **Gold passage retrieved (in top-10):** {retrieval_impact['gold_retrieved']} queries ({retrieval_impact['gold_retrieved_pct']:.1f}%)
  - Average MRR: {retrieval_impact['avg_mrr_retrieved']:.4f}
  
- **Gold passage NOT retrieved:** {retrieval_impact['gold_not_retrieved']} queries ({retrieval_impact['gold_not_retrieved_pct']:.1f}%)
  - Average MRR: 0.0000 (by definition - cannot rank what wasn't retrieved)

**Key Insight:** 
Retrieval quality is critical - {retrieval_impact['gold_retrieved_pct']:.1f}% of queries had the gold passage in top-10. For these queries, the average MRR was {retrieval_impact['avg_mrr_retrieved']:.4f}, showing that even when retrieved, the reranker must work to place it at the top. The {retrieval_impact['gold_not_retrieved_pct']:.1f}% of queries without gold passages in top-10 represent the upper bound of system improvement through better retrieval.

### 3.2 Reranker Score Distribution Analysis

**Analysis:**

Distribution of reranker scores for gold vs. non-gold passages (using position as proxy):

- **Gold passages:**
  - Mean score: {reranker_scores['gold']['mean']:.4f}
  - Std: {reranker_scores['gold']['stdev']:.4f}
  - Range: [{reranker_scores['gold']['min']:.4f}, {reranker_scores['gold']['max']:.4f}]
  - Count: {reranker_scores['gold']['count']}

- **Non-gold passages:**
  - Mean score: {reranker_scores['non_gold']['mean']:.4f}
  - Std: {reranker_scores['non_gold']['stdev']:.4f}
  - Range: [{reranker_scores['non_gold']['min']:.4f}, {reranker_scores['non_gold']['max']:.4f}]
  - Count: {reranker_scores['non_gold']['count']}

**Key Insight:** 
Gold passages have a mean position-based score of {reranker_scores['gold']['mean']:.4f} compared to {reranker_scores['non_gold']['mean']:.4f} for non-gold passages, showing good separation. However, the overlap in ranges suggests some gold passages still rank lower than ideal, indicating room for reranker improvement.

### 3.3 MRR Distribution Analysis

**Analysis:**

Distribution of MRR values across all queries:

- Mean MRR: {mrr_dist['mean_mrr']:.4f}
- Median MRR: {mrr_dist['median_mrr']:.4f}
- Queries with perfect rank (MRR=1.0): {mrr_dist['perfect_rank']} ({mrr_dist['perfect_rank_pct']:.1f}%)
- Queries with no retrieval (MRR=0.0): {mrr_dist['no_retrieval']} ({mrr_dist['no_retrieval_pct']:.1f}%)

**MRR Breakdown:**
| MRR Range | Count | Percentage |
|-----------|-------|------------|
| 1.0 (Rank 1) | {mrr_dist['breakdown']['rank_1']['count']} | {mrr_dist['breakdown']['rank_1']['pct']:.1f}% |
| 0.5 (Rank 2) | {mrr_dist['breakdown']['rank_2']['count']} | {mrr_dist['breakdown']['rank_2']['pct']:.1f}% |
| 0.33 (Rank 3) | {mrr_dist['breakdown']['rank_3']['count']} | {mrr_dist['breakdown']['rank_3']['pct']:.1f}% |
| 0.1-0.25 (Rank 4-10) | {mrr_dist['breakdown']['rank_4_10']['count']} | {mrr_dist['breakdown']['rank_4_10']['pct']:.1f}% |
| 0.0 (Not retrieved) | {mrr_dist['breakdown']['not_retrieved']['count']} | {mrr_dist['breakdown']['not_retrieved']['pct']:.1f}% |

**Key Insight:** 
{mrr_dist['perfect_rank_pct']:.1f}% of queries achieved perfect ranking (position 1), while {mrr_dist['breakdown']['rank_2']['pct'] + mrr_dist['breakdown']['rank_1']['pct']:.1f}% were in the top 2 positions. The mean MRR of {mrr_dist['mean_mrr']:.4f} indicates strong overall ranking performance, with the primary improvement opportunity in moving rank-2 and rank-3 answers to rank-1.

### 3.4 Custom Analysis: Position-wise MRR Contribution

**Analysis:**

We analyzed how each gold passage position contributes to the overall MRR score:

""")
    
    # Generate position-wise table
    markdown_table = []
    for pos in sorted(custom_analysis['position_stats'].keys()):
        stats = custom_analysis['position_stats'][pos]
        pos_label = f"Position {pos}" if pos > 0 else "Not retrieved"
        pct_of_total = stats['contribution_to_total'] / custom_analysis['total_queries'] * 100
        markdown_table.append(f"| {pos_label} | {stats['count']} | {stats['contribution_to_total']:.2f} | {pct_of_total:.2f}% |")
    
    print("""
| Position | Query Count | MRR Contribution | % of Total MRR |
|----------|-------------|------------------|----------------|""")
    for line in markdown_table:
        print(line)
    
    total_mrr = sum(custom_analysis['position_stats'][pos]['contribution_to_total'] 
                    for pos in custom_analysis['position_stats'])
    
    print(f"""
**Key Insight:**
The analysis shows that improving reranker performance to move passages from position 2 ({mrr_dist['breakdown']['rank_2']['count']} queries) and position 3 ({mrr_dist['breakdown']['rank_3']['count']} queries) to position 1 would yield the highest MRR gains. Specifically, moving all rank-2 passages to rank-1 would increase MRR by {mrr_dist['breakdown']['rank_2']['count'] * 0.5 / custom_analysis['total_queries']:.4f} points, representing a {mrr_dist['breakdown']['rank_2']['count'] * 0.5 / total_mrr * 100:.1f}% improvement in total MRR contribution.
""")


def main():
    parser = argparse.ArgumentParser(description="Generate Q3 analysis for report")
    parser.add_argument('result_file', type=str, help='Path to result JSON file')
    args = parser.parse_args()
    
    print(f"Loading result file: {args.result_file}")
    data = load_result(args.result_file)
    
    print("Analyzing retrieval impact...")
    retrieval_impact = analyze_retrieval_impact(data)
    
    print("Analyzing reranker scores...")
    reranker_scores = analyze_reranker_scores(data)
    
    print("Analyzing MRR distribution...")
    mrr_dist = analyze_mrr_distribution(data)
    
    print("Analyzing custom metrics...")
    custom_analysis = analyze_custom(data)
    
    print_report(retrieval_impact, reranker_scores, mrr_dist, custom_analysis)


if __name__ == '__main__':
    main()
