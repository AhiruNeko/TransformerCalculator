from Tokenizer import CHARS
from tqdm import tqdm
import json

from dataset.FormulaTree import FormulaTree


class Dataset:
    def __init__(
            self,
            size=100,
            step_range=(1, 5),
            num_range=(-99, 99),
            power_range=(1, 2),
            operator_probabilities=None,
            operator_max_use=None,
            bracket_probability=0.2,
            skip_steps=0.1
    ):
        self.size = size
        self.chars = CHARS
        self.step_range = step_range
        self.num_range = num_range
        self.power_range = power_range
        self.operator_probabilities = operator_probabilities
        self.operator_max_use = operator_max_use
        self.bracket_probability = bracket_probability
        self.skip_steps = skip_steps
        self.data = self.generate()

    def save(self, path=None):
        if path is None:
            path = f'./{self.__class__.__name__}.json'
        with open(path, 'w') as f:
            json.dump(self.data, f)
            print(f'Dataset saved as: {path}')

    def __call__(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    def generate(self):
        data = []
        for _ in tqdm(range(self.size), desc="Generating Dataset"):
            formula = FormulaTree(
                self.step_range,
                self.num_range,
                self.power_range,
                self.operator_probabilities,
                self.operator_max_use,
                self.bracket_probability,
                self.skip_steps
            )
            data.append(formula())
        return data


if __name__ == '__main__':
    dataset = Dataset(
        300000,
        operator_max_use={'^': 1, 'sqrt': 1}
    )
    dataset.save()
