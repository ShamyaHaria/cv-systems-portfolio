# Shamya Haria - CS 5330 Project 5
# Runs all tasks in sequence
# 04/05/2026

import sys
import os
import subprocess


def run_script(name, script):
    print(f'\n{"="*60}')
    print(f'Running: {name}')
    print('='*60)
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f'WARNING: {script} exited with code {result.returncode}')


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('outputs', exist_ok=True)

    scripts = [
        ('Task 1: Train MNIST CNN', 'task1_train.py'),
        ('Task 1b: Evaluate', 'task1_evaluate.py'),
        ('Task 1c: Custom Digits', 'task1_custom_digits.py'),
        ('Task 2: Examine Network', 'task2_examine.py'),
        ('Task 3: Greek Letters', 'task3_greek.py'),
        ('Task 4: Transformer', 'task4_transformer.py'),
        ('Task 5: Experiment', 'task5_experiment.py'),
        ('Extension 1: Gabor', 'extension1_gabor.py'),
        ('Extension 2: ResNet', 'extension2_resnet_analysis.py'),
    ]

    for name, script in scripts:
        run_script(name, script)

    print('\n' + '='*60)
    print('All done! Check outputs/ for results.')
    print('\nFor live webcam recognition: python extension3_live_recognition.py')


if __name__ == '__main__':
    main()
