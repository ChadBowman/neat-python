import unittest
import os
import neat
import random
import csv
import uuid


class TestStatistics(unittest.TestCase):

    def load_pop(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        return neat.Population(config)

    def test_save_genome_fitness_without_cv(self):
        def eval_genomes(genomes, config):
            for gnome_id, genome in genomes:
                genome.fitness = random.random() / 2
        pop = self.load_pop()
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.run(eval_genomes, 10)

        filename = str(uuid.uuid1())

        try:
            stats.save_genome_fitness(filename=filename)
            assert os.stat(filename).st_size != 0

            with open(filename, 'r') as f:
                for row in csv.reader(f, delimiter=' '):
                    assert len(row) == 2
        finally:
            os.remove(filename)

    def test_save_genome_fitness_with_cv(self):
        def eval_genomes(genomes, config):
            for gnome_id, genome in genomes:
                genome.fitness = random.random() / 2
                genome.cv_fitness = random.random() / 2
        pop = self.load_pop()
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.run(eval_genomes, 10)

        filename = str(uuid.uuid1())

        try:
            stats.save_genome_fitness(filename=filename)
            assert os.stat(filename).st_size != 0

            with open(filename, 'r') as f:
                for row in csv.reader(f, delimiter=' '):
                    assert len(row) == 4
        finally:
            os.remove(filename)
