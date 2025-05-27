from neat.reporting import BaseReporter


class NewBestReport(BaseReporter):
    def __init__(self, simulation_eval):
        super().__init__()
        self.best_fitness = -float("inf")
        self.fitness_target = 1000
        self.simulation_eval = simulation_eval

    def post_evaluate(self, config, population, species, best_genome):
        if self.best_fitness < best_genome.fitness:
            self.best_fitness = best_genome.fitness
            if best_genome.fitness > self.fitness_target:
                self.fitness_target = best_genome.fitness + 200
                self.simulation_eval(best_genome, config, config.visualizer)
