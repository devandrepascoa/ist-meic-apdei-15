import subprocess
import time

i = 0


def run_command(command, test_name):
    global i
    start_time = time.time()
    with open(f"stdout {test_name}.txt", "w") as outfile:
        try:
            # Execute the command and redirect output to a file
            subprocess.run(command, shell=True, check=True, stdout=outfile, stderr=outfile)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")
            sys.exit(1)
    end_time = time.time()

    elapsed_time = end_time - start_time
    with open("plots/Q2/execution_times.txt", "a") as file:
        file.write(f"{test_name} (Run {i}) - Elapsed time: {elapsed_time:.2f} seconds\n")
    i += 1


def run_mlp_test(test_name, epochs=20, learning_rate=0.1, hidden_size=200, layers=2, dropout=0.0, batch_size=16,
                 l2_decay=0, activation='relu', optimizer='sgd'):
    command = f"python.exe ./hw1-q2.py mlp -epochs={epochs} -learning_rate={learning_rate} -hidden_size={hidden_size} -layers={layers} -dropout={dropout} -batch_size={batch_size} -l2_decay={l2_decay} -activation={activation} -optimizer={optimizer}"
    run_command(command, test_name)


def run_logistic_regression_test(test_name, epochs=20, batch_size=16, learning_rate=0.01, l2_decay=0,
                                 hidden_size=100, layers=1, dropout=0.3, activation='relu', optimizer='sgd'):
    command = f"python.exe ./hw1-q2.py logistic_regression -epochs={epochs} -learning_rate={learning_rate} -hidden_size={hidden_size} -layers={layers} -dropout={dropout} -batch_size={batch_size} -l2_decay={l2_decay} -activation={activation} -optimizer={optimizer}"
    run_command(command, test_name)


# Logistic Regression exercise
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    run_logistic_regression_test(f"1. Logistic Regression - LR {lr}", learning_rate=lr)

# Exercise 1: Batch sizes
for batch_size in [16, 1024]:
    run_mlp_test(f"2.a) MLP - Batch Size {batch_size}", batch_size=batch_size)

# Exercise 2: Learning rates
for lr in [1, 0.1, 0.01, 0.001]:
    run_mlp_test(f"2.b) MLP - Learning Rate {lr}", learning_rate=lr)

configs = [
    {"l2_decay": "0", "dropout": "0.0"},
    {"l2_decay": "0.0001", "dropout": "0.0"},
    {"l2_decay": "0", "dropout": "0.2"}
]

for config in configs:
    run_mlp_test(f"2.c) MLP - Dropout {config['dropout']} L2 Decay {config['l2_decay']}", epochs=150,
                 batch_size=256, dropout=config["dropout"], l2_decay=config["l2_decay"])
