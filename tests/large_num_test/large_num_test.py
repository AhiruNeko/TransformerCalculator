from utils import *
import random

RANGES = {
    "1 Digit (0-9)": (0, 9),
    "2 Digits (10-99)": (10, 99),
    "3 Digits (100-999)": (100, 999),
    "4 Digits (1000-9999)": (1000, 9999),
    "5 Digits (10000-99999)": (10000, 99999)
}


def generate_add_sample(low, high):
    a = random.randint(low, high) * random.choice([1, -1])
    b = random.randint(low, high) * random.choice([1, -1])
    return f"{format_operand(a, is_first=True)}+{format_operand(b, is_first=False)}", a + b


def generate_sub_sample(low, high):
    a = random.randint(low, high) * random.choice([1, -1])
    b = random.randint(low, high) * random.choice([1, -1])
    return f"{format_operand(a, is_first=True)}-{format_operand(b, is_first=False)}", a - b


def generate_mul_sample(low, high):
    a = random.randint(low, high) * random.choice([1, -1])
    b = random.randint(low, high) * random.choice([1, -1])
    return f"{format_operand(a, is_first=True)}*{format_operand(b, is_first=False)}", a * b


def generate_div_sample(low, high):
    a = random.randint(low, high) * random.choice([1, -1])
    b = random.randint(low, high) * random.choice([1, -1]) if random.random() > 0.1 else 0

    if b == 0:
        return f"{format_operand(a, is_first=True)}/{format_operand(b, is_first=False)}", "[NAN]"

    return f"{format_operand(a, is_first=True)}/{format_operand(b, is_first=False)}", int(a / b)


def generate_pow_sample(low, high):
    a = random.randint(low, high) * random.choice([1, -1])
    p = random.randint(1, 2)
    return f"{format_operand(a, is_first=True)}^{p}", a ** p


def generate_sqrt_sample(low, high):
    a = random.randint(low, high) * random.choice([1, -1])

    if a < 0:
        return f"sqrt({a})", "[NAN]"
    return f"sqrt({a})", int(a ** 0.5)

def add_test(model, tokenizer, num_samples=100, device='cuda'):
    return test(model, tokenizer, RANGES, generate_add_sample, "Addition", num_samples, device)

def sub_test(model, tokenizer, num_samples=100, device='cuda'):
    return test(model, tokenizer, RANGES, generate_sub_sample, "Subtraction", num_samples, device)

def mul_test(model, tokenizer, num_samples=100, device='cuda'):
    return test(model, tokenizer, RANGES, generate_mul_sample, "Multiplication", num_samples, device)

def div_test(model, tokenizer, num_samples=100, device='cuda'):
    return test(model, tokenizer, RANGES, generate_div_sample, "Division", num_samples, device)

def pow_test(model, tokenizer, num_samples=100, device='cuda'):
    return test(model, tokenizer, RANGES, generate_pow_sample, "Exponentiation (^1 or ^2)", num_samples, device)

def sqrt_test(model, tokenizer, num_samples=100, device='cuda'):
    return test(model, tokenizer, RANGES, generate_sqrt_sample, "Square Root", num_samples, device)

if __name__ == '__main__':
    tokenizer = Tokenizer()
    model = load_model('../../best_model.pth')
    num_samples = 100
    all_results = {}
    all_results["Addition"] = add_test(model, tokenizer, num_samples)
    all_results["Subtraction"] = sub_test(model, tokenizer, num_samples)
    all_results["Multiplication"] = mul_test(model, tokenizer, num_samples)
    all_results["Division"] = div_test(model, tokenizer, num_samples)
    all_results["Exponentiation"] = pow_test(model, tokenizer, num_samples)
    all_results["Square Root"] = sqrt_test(model, tokenizer, num_samples)
    plot_all_results(all_results, save_path="./large_num_tests.png")