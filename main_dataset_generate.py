from dataset_generators.base_preparator import BasePreparator
from dataset_generators.individual_datasets import IndividualDatasetPrep
from dataset_generators.PCA_generators import pcaGenerator
from dataset_generators.correlation_generators import CorrelationGenerator
from dataset_generators.balancing import Balancer

#dataset_generator = BasePreparator()
#dataset_generator.base_dataset_pipeline()

#dataset_generator = Balancer()
#dataset_generator.sampler()

#dataset_generator = IndividualDatasetPrep()
#dataset_generator.pipeline()

#dataset_generator = pcaGenerator()
#dataset_generator.pipeline_binary()

dataset_generator = CorrelationGenerator()
dataset_generator.correlation_pipeline()


