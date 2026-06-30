import copy
import math
import random

NUMS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
OPERATORS = ['+', '-', '*', '/', 'sqrt', '^']
PRIORITY = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '^': 3,
    'sqrt': 4
}

class Node:
    def __init__(self, char):
        self.char = char
        self.left = None
        self.right = None

class FormulaTree:
    def __init__(
            self,
            step_range=(1, 5),
            num_range=(-99, 99),
            power_range=(1, 2),
            operator_probabilities=None,
            operator_max_use=None,
            bracket_probability=0,
            skip_steps=0
    ):
        if operator_probabilities is None:
            operator_probabilities = [1 / len(OPERATORS) for _ in range(len(OPERATORS))]
        assert len(operator_probabilities) == len(OPERATORS)
        self.origin_operator_probabilities = operator_probabilities
        self.operator_probabilities = copy.deepcopy(self.origin_operator_probabilities)
        self.bracket_probability = bracket_probability
        self.root = None
        self.steps = 0
        self.min_step, self.max_step = step_range
        self.min_num, self.max_num = num_range
        self.min_pow, self.max_pow = power_range
        self.skip_steps = skip_steps
        self.operator_max_use = operator_max_use
        self.available_operators = copy.deepcopy(OPERATORS)
        self.operator_count = {i: 0 for i in self.available_operators}

    def _update_operator_count(self, operator):
        if operator in self.available_operators:
            self.operator_count[operator] += 1
            if self.operator_max_use is not None and self.operator_count[operator] >= self.operator_max_use.get(operator, self.steps):
                idx = self.available_operators.index(operator)
                self.available_operators.remove(operator)
                self.operator_probabilities.pop(idx)

    def _insert(self, node, char):
        if node is None:
            self.steps += 1
            new_node = Node(char)
            if char == '^':
                new_node.right = Node(str(random.randint(self.min_pow, self.max_pow)))
            return new_node
        if node.char == 'sqrt':
            node.right = self._insert(node.right, char)
        elif node.char == '^':
            node.left = self._insert(node.left, char)
        else:
            random_num = random.randint(0, 1)
            if random_num == 0:
                node.left = self._insert(node.left, char)
            else:
                node.right = self._insert(node.right, char)
        return node

    def insert(self, char):
        self.root = self._insert(self.root, char)

    def _insert_nums(self, node):
        if node is None:
            return Node(str(random.randint(self.min_num, self.max_num)))
        if node.char != 'sqrt':
            node.left = self._insert_nums(node.left)
        if node.char != '^':
            node.right = self._insert_nums(node.right)
        return node

    def insert_nums(self):
        self.root = self._insert_nums(self.root)

    def random_insert(self):
        length = random.randint(self.min_step, self.max_step)
        for _ in range(length):
            operator = random.choices(self.available_operators, weights=self.operator_probabilities, k=1)[0]
            self.insert(operator)
            self._update_operator_count(operator)
        self.insert_nums()

    def _has_bracket_inside(self, node):
        if node is None:
            return False

        if self._has_bracket_inside(node.left) or self._has_bracket_inside(node.right):
            return True

        char = node.char
        left_char = node.left.char if node.left is not None else None
        right_char = node.right.char if node.right is not None else None

        if char in OPERATORS:

            if left_char in OPERATORS and PRIORITY[char] > PRIORITY[left_char]:
                return True
            if right_char in OPERATORS:
                if PRIORITY[char] >= PRIORITY[right_char]:
                    return True

            if left_char in OPERATORS and self._has_bracket_inside(node.right):
                return True

        if char == 'sqrt':
            return True

        return False

    def _in_order(self, node, s=''):
        if node is None:
            return s

        char = node.char

        right_has_any_brac = self._has_bracket_inside(node.right)

        left_char = node.left.char if node.left is not None else None
        right_char = node.right.char if node.right is not None else None

        left_brackets = False
        right_brackets = False

        if char in OPERATORS:

            if left_char in OPERATORS and PRIORITY[char] > PRIORITY[left_char]:
                left_brackets = True
            if right_char in OPERATORS:
                if PRIORITY[char] >= PRIORITY[right_char]:
                    right_brackets = True

            if (right_brackets or right_has_any_brac) and left_char in OPERATORS:
                left_brackets = True

        if char == 'sqrt':
            right_brackets = True

        s += '(' if left_brackets else ''
        s = self._in_order(node.left, s)
        s += ')' if left_brackets else ''
        if '-' in char and len(char) > 1:
            if s == '' or s[-1] == '(':
                s += char
            else:
                s += f'({char})'
        else:
            s += char

        s += '(' if right_brackets else ''
        s = self._in_order(node.right, s)
        s += ')' if right_brackets else ''

        return s

    def in_order(self):
        return self._in_order(self.root)

    @staticmethod
    def _calculate(operator, *operands):
        assert 0 < len(operands) <= 2
        match operator:
            case '+':
                return int(sum(operands))
            case '-':
                return int(operands[0] - operands[1])
            case '*':
                return int(operands[0] * operands[1])
            case '/':
                if operands[1] == 0:
                    return '[NAN]'
                return int(operands[0] / operands[1])
            case '^':
                return int(operands[0] ** operands[1])
            case 'sqrt':
                if operands[1] < 0:
                    return '[NAN]'
                return int(math.sqrt(operands[1]))
        return None

    def _get_steps_trees(self, node, steps_trees=None):
        if steps_trees is None:
            steps_trees = []
        if node is None or node.char not in OPERATORS:
            return steps_trees

        steps_trees = self._get_steps_trees(node.left, steps_trees)
        if len(steps_trees) > 0 and steps_trees[-1].root.char == '[NAN]':
            return steps_trees
        steps_trees = self._get_steps_trees(node.right, steps_trees)
        if len(steps_trees) > 0 and steps_trees[-1].root.char == '[NAN]':
            return steps_trees

        left_num = int(node.left.char) if node.left is not None else None
        right_num = int(node.right.char) if node.right is not None else None
        node.char = str(self._calculate(node.char, left_num, right_num))
        self.steps -= 1
        if node.char == '[NAN]':
            self.root = Node('[NAN]')
            self.steps = 0
            steps_trees.append(copy.deepcopy(self))
            return steps_trees
        node.left, node.right = None, None
        skip_step = random.choices([True, False], weights=[self.skip_steps, 1-self.skip_steps], k=1)[0]
        if not skip_step or self.steps <= 1:
            steps_trees.append(copy.deepcopy(self))

        return steps_trees

    def get_steps_trees(self):
        return self._get_steps_trees(self.root)

    def get_steps(self):
        s = ''
        steps = self.get_steps_trees()
        for trees in steps[:-1]:
            s += trees.in_order() + '='
        s += steps[-1].in_order()
        return s

    def __call__(self):
        self.available_operators = copy.deepcopy(OPERATORS)
        self.operator_probabilities = copy.deepcopy(self.origin_operator_probabilities)
        self.root = None
        self.steps = 0
        self.random_insert()
        return f'{self.in_order()}={self.get_steps()}'


if __name__ == '__main__':
    tree = FormulaTree(
        operator_max_use={'^': 1, 'sqrt': 1}
    )
    print(tree())



