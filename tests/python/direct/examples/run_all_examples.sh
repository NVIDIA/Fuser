#!/bin/bash
# Run all nvFuser tutorial examples
# Usage: ./run_all_examples.sh [--skip-tma]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_TMA=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-tma)
            SKIP_TMA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-tma]"
            echo "  --skip-tma    Skip TMA examples (requires Hopper GPU)"
            exit 0
            ;;
    esac
done

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

run_example() {
    local file=$1
    local description=$2
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running: ${file}${NC}"
    echo -e "${BLUE}${description}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if python "$file"; then
        echo -e "${GREEN}✓ ${file} completed successfully${NC}\n"
        return 0
    else
        echo -e "${RED}✗ ${file} failed${NC}\n"
        return 1
    fi
}

FAILED_TESTS=()
PASSED_TESTS=()

# Basic Examples
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}BASIC EXAMPLES${NC}"
echo -e "${YELLOW}========================================${NC}\n"

run_example "01-memcpy.py" "Basic memory copy" && PASSED_TESTS+=("01-memcpy.py") || FAILED_TESTS+=("01-memcpy.py")
run_example "02-memcpy_scheduled.py" "Scheduled memory copy" && PASSED_TESTS+=("02-memcpy_scheduled.py") || FAILED_TESTS+=("02-memcpy_scheduled.py")
run_example "03-reduction.py" "Reduction operations" && PASSED_TESTS+=("03-reduction.py") || FAILED_TESTS+=("03-reduction.py")
run_example "04-reduction_rfactor.py" "Rfactor reductions" && PASSED_TESTS+=("04-reduction_rfactor.py") || FAILED_TESTS+=("04-reduction_rfactor.py")
run_example "05-reshape.py" "Reshape operations" && PASSED_TESTS+=("05-reshape.py") || FAILED_TESTS+=("05-reshape.py")
run_example "06-id_model_reshape_analysis.py" "IdModel analysis" && PASSED_TESTS+=("06-id_model_reshape_analysis.py") || FAILED_TESTS+=("06-id_model_reshape_analysis.py")

# TMA Examples (Hopper GPU required)
if [ "$SKIP_TMA" = false ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}TMA EXAMPLES (Hopper GPU Required)${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
    
    run_example "07-basic_tma_example1.py" "Single 1D TMA" && PASSED_TESTS+=("07-basic_tma_example1.py") || FAILED_TESTS+=("07-basic_tma_example1.py")
    run_example "08-basic_tma_example2.py" "Multiple 1D TMA (for loop)" && PASSED_TESTS+=("08-basic_tma_example2.py") || FAILED_TESTS+=("08-basic_tma_example2.py")
    run_example "09-basic_tma_example3.py" "Thread-parallelized 1D TMA" && PASSED_TESTS+=("09-basic_tma_example3.py") || FAILED_TESTS+=("09-basic_tma_example3.py")
    run_example "10-basic_tma_example4.py" "TMA store" && PASSED_TESTS+=("10-basic_tma_example4.py") || FAILED_TESTS+=("10-basic_tma_example4.py")
    run_example "11-basic_tma_example5.py" "2D TMA tiling" && PASSED_TESTS+=("11-basic_tma_example5.py") || FAILED_TESTS+=("11-basic_tma_example5.py")
    run_example "12-basic_tma_example6.py" "2D TMA store" && PASSED_TESTS+=("12-basic_tma_example6.py") || FAILED_TESTS+=("12-basic_tma_example6.py")
    run_example "13-vectorize_store_pointwise_tma.py" "Vectorized pointwise" && PASSED_TESTS+=("13-vectorize_store_pointwise_tma.py") || FAILED_TESTS+=("13-vectorize_store_pointwise_tma.py")
    run_example "14-pointwise_broadcast_tma.py" "Broadcasting with TMA" && PASSED_TESTS+=("14-pointwise_broadcast_tma.py") || FAILED_TESTS+=("14-pointwise_broadcast_tma.py")
    run_example "15-tma_bank_conflict_free_transpose.py" "Bank-conflict-free transpose" && PASSED_TESTS+=("15-tma_bank_conflict_free_transpose.py") || FAILED_TESTS+=("15-tma_bank_conflict_free_transpose.py")
else
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}TMA EXAMPLES (SKIPPED)${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
    echo -e "${YELLOW}TMA examples skipped (use without --skip-tma to run)${NC}\n"
fi

# Auto Scheduler Examples
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}AUTO SCHEDULER EXAMPLES${NC}"
echo -e "${YELLOW}========================================${NC}\n"

run_example "16-pointwise_auto_scheduler.py" "Pointwise auto scheduler" && PASSED_TESTS+=("16-pointwise_auto_scheduler.py") || FAILED_TESTS+=("16-pointwise_auto_scheduler.py")
run_example "17-reduction_auto_scheduler.py" "Reduction auto scheduler" && PASSED_TESTS+=("17-reduction_auto_scheduler.py") || FAILED_TESTS+=("17-reduction_auto_scheduler.py")
run_example "18-inner_persistent_auto_scheduler.py" "Inner persistent auto scheduler" && PASSED_TESTS+=("18-inner_persistent_auto_scheduler.py") || FAILED_TESTS+=("18-inner_persistent_auto_scheduler.py")

# Summary
echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}SUMMARY${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "${GREEN}Passed: ${#PASSED_TESTS[@]}${NC}"
echo -e "${RED}Failed: ${#FAILED_TESTS[@]}${NC}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}✗ $test${NC}"
    done
    exit 1
else
    echo -e "\n${GREEN}All tests passed! 🎉${NC}"
    exit 0
fi

