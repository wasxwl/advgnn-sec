#!/bin/bash
set -e

echo "=== Running Full Pipeline ==="

echo ""
echo "Step 1: Training baseline detector..."
python experiments/baseline.py --device auto --epochs 200

echo ""
echo "Step 2: Evaluating adversarial attacks..."
python experiments/attack_eval.py --device auto

echo ""
echo "Step 3: Evaluating defenses..."
python experiments/defense_eval.py --device auto --epochs 200

echo ""
echo "Step 4: Running ablation studies..."
python experiments/ablation.py --device auto --epochs 200

echo ""
echo "Step 5: Running comprehensive evaluation (5 seeds)..."
python experiments/comprehensive_eval.py --device auto

echo ""
echo "Step 6: Running adaptive arms race simulation..."
python experiments/adaptive_eval.py --device auto

echo ""
echo "Step 7: Running MGTAB real-world dataset evaluation..."
python experiments/mgtab_eval.py --device auto

echo ""
echo "=== Pipeline Complete ==="
echo "Results are in the results/ directory"
