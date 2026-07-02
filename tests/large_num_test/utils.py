import matplotlib.pyplot as plt
from inference import *

def format_operand(num, is_first=False):
    if num < 0:
        return f"{num}" if is_first else f"({num})"
    return f"{num}"

def evaluate(formula, true_ans, model, tokenizer, device):
    try:
        pred_str = singe_inference(formula, model, tokenizer, device=device)

        if '=' in pred_str:
            clean_pred = pred_str.split('=')[-1].replace('[EOS]', '').replace('[PAD]', '').strip()

            if clean_pred == '[NAN]':
                return (true_ans == '[NAN]', clean_pred)

            if int(clean_pred) == true_ans:
                return (True, clean_pred)
            else:
                return (False, clean_pred)
    except Exception as e:
        return (False, e)


def test(model, tokenizer, digit_ranges, sample_generator, task_name, num_samples=100, device='cuda'):
    print(f"Running Test: [{task_name}]")
    print("=" * 60)

    categories = []
    accuracies = []

    for label, (low, high) in digit_ranges.items():
        correct_count = 0
        error_cases = []

        for _ in range(num_samples):
            formula, true_ans = sample_generator(low, high)
            is_correct, pred_val = evaluate(formula, true_ans, model, tokenizer, device)

            if is_correct:
                correct_count += 1
            else:
                error_cases.append((formula, true_ans, pred_val))

        acc = (correct_count / num_samples) * 100
        categories.append(label)
        accuracies.append(acc)

        print(f"{label} Accuracy: {acc:.2f}%")

        if error_cases:
            print("Error cases:")
            for idx, (form, real, pred) in enumerate(error_cases[:3]):
                print(f"  Case {idx + 1}:  Input: {form}  |  Answer: {real}  |  Output: {pred}")
        print("-" * 50)
    return categories, accuracies

def plot_all_results(results_dict, save_path="./large_num_tests.png"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()
    line_color = '#e056fd'

    for idx, (task_name, (categories, accuracies)) in enumerate(results_dict.items()):
        ax = axes[idx]
        ax.plot(categories, accuracies, marker='o', linewidth=2.5, color=line_color, label='Accuracy')

        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 3, f"{acc:.1f}%", ha='center', va='bottom', fontsize=9, weight='bold')
        ax.set_title(f"{task_name} Test", fontsize=12, weight='bold', pad=10)
        ax.set_ylim(-10, 115)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.axhline(y=80, color='r', linestyle=':', alpha=0.6, label='Baseline (80%)')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)

        if idx in [0, 3]:
            ax.set_ylabel("Accuracy (%)", fontsize=11)

        ax.legend(loc='lower left', fontsize=9)

    plt.suptitle("Large Number Test", fontsize=18, weight='bold',
                 y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Figure Saved: {save_path}")
    plt.show()
    plt.close()