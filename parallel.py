"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool
from functools import partial
from typing import Tuple


def eval_function_with_genome_id(eval_function, genome_id, genome, config)-> Tuple[str, float]:
    score = eval_function(genome, config)
    return genome_id, score

class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.eval_function = partial(eval_function_with_genome_id, eval_function)
    

    def __del__(self):
        self.pool.terminate()
        self.pool.join()

    def evaluate(self, genomes, config):
        genome_dict = {genome_id:genome for genome_id, genome in genomes}
        args = [(genome_id, genome, config) for genome_id, genome in genomes]
        results = self.pool.starmap(self.eval_function, args)

        for genome_id, fitness in results:
            genome_dict[genome_id].fitness = fitness

