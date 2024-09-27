from multiprocessing import Pool
import optuna
from optimization import optimize_file

optuna.logging.set_verbosity(optuna.logging.WARNING)

if __name__ == '__main__':
    with Pool() as p:
        res = p.map(optimize_file, ["../Project 002/aapl_5m_train.csv"])
        for result in res:
            print(result)

