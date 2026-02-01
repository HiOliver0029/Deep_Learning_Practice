#!/bin/bash

# RL Inference Complete Pipeline
# This script runs the full RL inference pipeline including:
# 1. RL inference with dynamic Top_M selection
# 2. Evaluation of results
# 3. Comparison with baseline fixed Top_M results

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
}

# Check if required files exist
check_requirements() {
    print_header "Checking Requirements"
    
    local all_ok=true
    
    # Check RL model
    if [ -d "results/models/rl_agent" ]; then
        print_success "RL model found: results/models/rl_agent"
    else
        print_error "RL model not found: results/models/rl_agent"
        all_ok=false
    fi
    
    # Check retriever results
    if [ -f "results/result_bm25_hard.json" ]; then
        print_success "Retriever results found: results/result_bm25_hard.json"
    else
        print_error "Retriever results not found: results/result_bm25_hard.json"
        all_ok=false
    fi
    
    # Check inference script
    if [ -f "code/inference_with_rl_dynamic.py" ]; then
        print_success "RL inference script found"
    else
        print_error "RL inference script not found: code/inference_with_rl_dynamic.py"
        all_ok=false
    fi
    
    # Check evaluation script
    if [ -f "code/evaluate_results.py" ]; then
        print_success "Evaluation script found"
    else
        print_error "Evaluation script not found: code/evaluate_results.py"
        all_ok=false
    fi
    
    if [ "$all_ok" = false ]; then
        print_error "Missing required files. Please check the setup."
        exit 1
    fi
    
    print_success "All requirements met!"
    echo ""
}

# Run RL inference
run_rl_inference() {
    print_header "Step 1: Running RL Inference"
    
    print_info "This will use the trained RL agent to predict optimal Top_M for each query"
    print_info "and run LLM inference with dynamic passage selection."
    print_warning "Expected time: 2-3 hours (3342 queries)"
    echo ""
    
    read -p "Continue with RL inference? (y/n): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "RL inference skipped by user"
        return 1
    fi
    
    print_info "Starting RL inference..."
    echo ""
    
    python code/inference_with_rl_dynamic.py \
        --rl_model results/models/rl_agent \
        --retriever_results results/result_bm25_hard.json \
        --output results/result_rl_dynamic.json
    
    if [ $? -eq 0 ]; then
        print_success "RL inference completed successfully!"
        print_info "Results saved to: results/result_rl_dynamic.json"
        print_info "RL decisions saved to: results/result_rl_dynamic_rl_decisions.json"
    else
        print_error "RL inference failed!"
        exit 1
    fi
    
    echo ""
    return 0
}

# Evaluate RL results
evaluate_rl_results() {
    print_header "Step 2: Analyzing RL Results"
    
    if [ ! -f "results/result_rl_dynamic.json" ]; then
        print_error "RL results not found. Please run RL inference first."
        return 1
    fi
    
    print_info "Analyzing RL decision statistics..."
    echo ""
    
    # Fix result format if needed
    if [ -f "fix_rl_results_format.py" ]; then
        print_info "Fixing result format for evaluation compatibility..."
        python fix_rl_results_format.py \
            --rl_results results/result_rl_dynamic.json \
            --retriever_results results/result_bm25_hard.json \
            --output results/result_rl_dynamic_fixed.json
        
        if [ $? -eq 0 ]; then
            print_success "Format fixed!"
        fi
    fi
    
    # Display RL statistics
    if [ -f "results/result_rl_dynamic.json" ]; then
        print_info "=== RL Top_M Decision Statistics ==="
        python -c "
import json
data = json.load(open('results/result_rl_dynamic.json', encoding='utf-8'))
meta = data.get('metadata', {})
stats = meta.get('top_m_statistics', {})
print(f\"  Mean Top_M: {stats.get('mean', 0):.2f}\")
print(f\"  Median Top_M: {stats.get('median', 0):.0f}\")
print(f\"  Std Dev: {stats.get('std', 0):.2f}\")
print(f\"\\n  Distribution:\")
dist = stats.get('distribution', {})
for k in sorted(int(x) for x in dist.keys()):
    count = dist[str(k)]
    pct = 100 * count / meta.get('total_queries', 1)
    print(f\"    Top_M={k}: {count:4d} queries ({pct:5.1f}%)\")
"
    fi
    
    print_info "\nFor full evaluation (Recall, MRR, CosSim), run on the server:"
    print_info "  python evaluate_rl_simple.py --result_file results/result_rl_dynamic_fixed.json"
    
    echo ""
    return 0
}

# Compare with baselines
compare_with_baselines() {
    print_header "Step 3: Comparing with Baselines"
    
    if [ ! -f "results/result_rl_dynamic.json" ]; then
        print_error "RL results not found. Please run RL inference first."
        return 1
    fi
    
    print_info "Checking available baseline results..."
    
    # Check which baselines exist
    local baselines=""
    local baseline_files=""
    
    if [ -f "results/result_top3.json" ]; then
        baselines="$baselines Top_M=3"
        baseline_files="$baseline_files --baseline_top3 results/result_top3.json"
    fi
    
    if [ -f "results/result_top5.json" ]; then
        baselines="$baselines Top_M=5"
        baseline_files="$baseline_files --baseline_top5 results/result_top5.json"
    fi
    
    if [ -f "results/result_top7.json" ]; then
        baselines="$baselines Top_M=7"
        baseline_files="$baseline_files --baseline_top7 results/result_top7.json"
    fi
    
    if [ -f "results/result_bm25_hard.json" ]; then
        baselines="$baselines Top_M=10"
        baseline_files="$baseline_files --baseline_top10 results/result_bm25_hard.json"
    fi
    
    if [ -z "$baselines" ]; then
        print_warning "No baseline results found."
        print_info "Available baselines: (none)"
        print_info "You can generate baselines by running inference with fixed Top_M values."
        echo ""
        read -p "Skip comparison? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            return 0
        else
            return 1
        fi
    fi
    
    print_info "Available baselines:$baselines"
    echo ""
    
    read -p "Run comparison with available baselines? (y/n): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Comparison skipped by user"
        return 0
    fi
    
    print_info "Running comparison..."
    echo ""
    
    python compare_rl_with_baselines.py \
        --rl_results results/result_rl_dynamic.json \
        $baseline_files \
        --output results/rl_comparison_report.json
    
    if [ $? -eq 0 ]; then
        print_success "Comparison completed!"
        print_info "Report saved to: results/rl_comparison_report.json"
        
        # Display comparison results
        if [ -f "results/rl_comparison_report.json" ]; then
            echo ""
            print_info "=== Comparison Summary ==="
            cat results/rl_comparison_report.json | python -m json.tool
        fi
    else
        print_error "Comparison failed!"
        return 1
    fi
    
    echo ""
    return 0
}

# Generate summary
generate_summary() {
    print_header "Pipeline Summary"
    
    print_info "Generated files:"
    
    if [ -f "results/result_rl_dynamic.json" ]; then
        local size=$(du -h "results/result_rl_dynamic.json" | cut -f1)
        print_success "✓ RL inference results: results/result_rl_dynamic.json ($size)"
    fi
    
    if [ -f "results/result_rl_dynamic_rl_decisions.json" ]; then
        local size=$(du -h "results/result_rl_dynamic_rl_decisions.json" | cut -f1)
        print_success "✓ RL decisions: results/result_rl_dynamic_rl_decisions.json ($size)"
    fi
    
    if [ -f "results/eval_rl_dynamic.json" ]; then
        print_success "✓ Evaluation metrics: results/eval_rl_dynamic.json"
    fi
    
    if [ -f "results/rl_comparison_report.json" ]; then
        print_success "✓ Comparison report: results/rl_comparison_report.json"
    fi
    
    echo ""
    print_info "Next steps:"
    echo "  1. Review the evaluation metrics in results/eval_rl_dynamic.json"
    echo "  2. Check the RL decision distribution in results/result_rl_dynamic_rl_decisions.json"
    echo "  3. Analyze the comparison report in results/rl_comparison_report.json"
    echo "  4. Update report_template.md Bonus.3 section with experimental results"
    echo ""
}

# Main execution
main() {
    print_header "RL Inference Complete Pipeline"
    
    print_info "This script will execute the full RL inference pipeline:"
    echo "  1. Run RL inference with dynamic Top_M selection (2-3 hours)"
    echo "  2. Evaluate the results (Recall@10, MRR@10, CosSim)"
    echo "  3. Compare with fixed Top_M baselines"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Run RL inference
    if run_rl_inference; then
        # Evaluate results
        evaluate_rl_results
        
        # Compare with baselines
        compare_with_baselines
        
        # Generate summary
        generate_summary
        
        print_success "Pipeline completed successfully!"
    else
        print_warning "Pipeline execution was interrupted or skipped."
        
        # Check if results exist from previous runs
        if [ -f "results/result_rl_dynamic.json" ]; then
            echo ""
            print_info "Found existing RL results from previous run."
            read -p "Would you like to continue with evaluation and comparison? (y/n): " -n 1 -r
            echo ""
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                evaluate_rl_results
                compare_with_baselines
                generate_summary
            fi
        fi
    fi
    
    echo ""
    print_info "For more information, see RL_INFERENCE_GUIDE.md"
}

# Run main function
main
