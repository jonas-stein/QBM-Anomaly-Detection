from neal import SimulatedAnnealingSampler

import numpy as np
import dimod as di
from src.utils import *
import math

from uqo.client.config import Config
from uqo import Problem

from pathlib import Path, PurePath
import dwave_networkx
import minorminer
from matplotlib import pyplot as plt
from dwave.cloud import Client
from src.secrets.dwave_token import TOKEN

from src.model import MODEL
from itertools import repeat

from tqdm import tqdm

# semi-restricted (?) Quantum Boltzmann machine with one hidden and one visible layer
# (connections only between layers, and between visible nodes if restricted == False, no connections between hidden nodes)
# lateral connections between hidden neurons in one layer should easily be possible given enough qubits -> TODO?


class QBM(MODEL):
    def __init__(self, data, epochs=2, n_hidden_nodes=4, seed=77, weight_csv=None, solver="SA",
                 sample_count=20, anneal_steps=20, beta_eff=1, quantile=0.95, trained=False, restricted=True,
                 param_string="", savepoint=""):

        super().__init__(n_hidden_nodes, seed, epochs, trained, quantile)
        self.embedding = None

        self.encoded_data, bits_input_vector, num_features = self.binary_encode_data(
            data, use_folding=True)
        self.dim_input = bits_input_vector * num_features

        self.restricted = restricted
        if not trained:
            self.weights_visible_to_hidden, self.weights_visible_to_visible, self.biases_visible, self.biases_hidden = self.init_weights()
            self.paramstring = param_string
            self.weight_objects = [
                self.weights_visible_to_hidden, self.biases_visible, self.biases_hidden]
            if not restricted:
                self.weight_objects.append(self.weights_visible_to_visible)
        else:
            # TODO: adjust
            pass
            #self.Q_hh, self.weights_visible_to_hidden = self.init_weights_from_saved_model(read_from_csv(weight_csv, seed))
        if solver == "SA":
            self.sampler = SimulatedAnnealingSampler()
            # number of simulated annealing steps to create one sample
            self.anneal_steps = anneal_steps
        self.solver_string = solver
        # number of samples from Annealing (i.e. number of anneals) TODO: find good default value
        self.sample_count = sample_count
        # 1/(k_b * T) "effective Temperature" TODO: find good value / way to calculate (Müller and Adachi have 2 -> ?)
        self.beta_eff = beta_eff

        if savepoint != "":
            self.load_savepoint(savepoint)

    # done
    def init_weights(self):
        '''Initializes a (restricted) Boltzmann machine.
           Creates two weight-matrices weights_visible_to_hidden, and weights_visible_to_visible if not restricted.
            Also creates two bias-vectors biases_visible and biases_hidden
        '''
        # weights from visible to hidden layer
        weights_visible_to_hidden = np.random.uniform(
            -1, 1, (self.dim_input, self.n_hidden_nodes))

        # bias hidden layer
        biases_hidden = np.random.uniform(-1, 1, self.n_hidden_nodes)

        # bias visible layer
        # initialization according to tipps from Hinton (2012): A Practical guide to training Restricted Boltzmann Machines
        visible_bias_value = - \
            math.log((1/self.dim_input)/(1 - (1/self.dim_input)))
        biases_visible = np.array(
            [visible_bias_value for _ in range(self.dim_input)])
        # might work better?, but no time to test: biases_visible = np.random.uniform(-1, 1, self.dim_input)

        if self.restricted:
            # no connections between visible nodes = connections with weights valued zero that do no change
            weights_visible_to_visible = np.zeros(
                (self.dim_input, self.dim_input))
        else:
            # weights between visible nodes (upper triangular matrix, with diagonal still zeros)
            weights_visible_to_visible = np.triu(
                np.random.uniform(-1, 1, (self.dim_input, self.dim_input)), k=1)

        return weights_visible_to_hidden, weights_visible_to_visible, biases_visible, biases_hidden

    # done

    def create_qubo_matrix_from(self, input_vector: np.ndarray = None):

        # clamped
        # 3 visible, 4 hidden, clamped, upper right triangular matrix
        #
        #          (hb1 + v1h1w*v1 + v2h1w*v2.......) -------
        #             (hb2  + v1h2w*v1 + v2h2w*v2.......)-----
        #                hb3 + v1h3w*v1 + v2h3w*v2.......)-----
        #                   hb4
        if type(input_vector) is np.ndarray:
            input_vector = np.array([input_vector])
            # if visible nodes are clamped, biases and weights between them are ignored
            # build diagonal matrix containing hidden biases
            qubo_matrix = (np.diag(self.biases_hidden) +
                           np.diag(np.matmul(input_vector, self.weights_visible_to_hidden).flatten())) / self.beta_eff

        # unclamped
        # 3 visible, 4 hidden, unclamped, upper right triangular matrix
        #
        # Restricted:
        # vb1 -  -  v1h1w v1h2w .....
        #    vb2 -  v2h1w v2h2w .....
        #       vb3 v3h1w v3h2w .....
        #           hb1     -    -    -
        #                  hb2   -    -
        #                       hb3   -
        #                            hb4
        #
        # Not restricted:
        # vb1 v1v2w  v1v3w  v1h1w v1h2w .....
        #      vb2   v2v3w  v2h1w v2h2w .....
        #            vb3    v3h1w v3h2w .....
        #                    hb1    -    -    -
        #                          hb2   -    -
        #                               hb3   -
        #                                    hb4

        else:
            # turn weights_visible_to_hidden into a matrix of qubo_matrix's shape
            # in the restricted case, self.weights_visible_to_visible is simply a matrix filled with zeros
            upper_part = np.concatenate(
                (self.weights_visible_to_visible, self.weights_visible_to_hidden), axis=1)
            upper_part.resize(
                (self.dim_input+self.n_hidden_nodes, upper_part.shape[1]), refcheck=False)

            qubo_matrix = (np.diag(np.concatenate(
                (self.biases_visible, self.biases_hidden))) + upper_part) / self.beta_eff

        return qubo_matrix

    # done
    def get_samples(self, input_vector=None):
        # TODO all solvers (except SA) respond with only one sample
        #     Except DAU who seems to map equal samples upon each other

        qubo_as_bqm = di.BQM(
            self.create_qubo_matrix_from(input_vector), "BINARY")
        # TODO: try whether this works better with qubo or ising?
        if self.solver_string == "SA":
            samples = list(self.sampler.sample(
                qubo_as_bqm, num_reads=self.sample_count, num_sweeps=self.anneal_steps).samples())
        else:
            qubo_dict = qubo_as_bqm.to_qubo()[0]
            config = Config(configpath='src/secrets/config.json')
            if self.solver_string == "QBSolv":
                # qbsolv for testing uqp connection without decreasing our quota
                answer = Problem.Qubo(config, qubo_dict).with_platform(
                    "qbsolv").solve(self.sample_count)
            elif self.solver_string == "Advantage_system4.1" or self.solver_string == "DW_2000Q_6":
                # needs dwave quota
                problem = Problem.Qubo(config, qubo_dict).with_platform(
                    "dwave").with_solver(self.solver_string)
                # calculate embedding
                if input_vector is None:
                    problem.embedding = self.find_embedding_with_client(
                        qubo_as_bqm, self.solver_string, False) if self.embedding is None else self.embedding
                try:
                    answer = problem.solve(self.sample_count)
                except:
                    answer = problem.solve(self.sample_count)
            elif self.solver_string == "FujitsuDAU":
                # there are many parameters, that can be checked in the UQO examples
                parameters = {
                    "optimization_method": "annealing",
                    "auto_tuning": 2
                }
                # "CPU" for testing (no quota used), "DAU" with quota
                # TODO
                # 0 valued qubits will be kicked out of the qubit
                problem = Problem.Qubo(config, qubo_dict).with_platform("fujitsu").with_solver("DAU").with_params(
                    **parameters)
                try:
                    answer = problem.solve(self.sample_count)
                except:
                    answer = problem.solve(self.sample_count)
            else:
                raise Exception(
                    'No valid solver specified. Valid solvers are "SA", "QBSolv", "DW_2000Q_6", "Advantage_system4.1", "FujitsuDAU"')
            samples = self.split_samplelist_according_to_occurences(
                answer.solutions, answer.num_occurrences)
        return samples

    def get_energy(self, input_vector=None):
        qubo_as_bqm = di.BQM(
            self.create_qubo_matrix_from(input_vector), "BINARY")
        energy = None
        # TODO: try whether this works better with qubo or ising?
        if self.solver_string == "SA":
            sampleset = self.sampler.sample(
                qubo_as_bqm, num_reads=self.sample_count, num_sweeps=self.anneal_steps)
            energy = sampleset.first.energy
        else:
            qubo_dict = qubo_as_bqm.to_qubo()[0]
            config = Config(configpath='src/secrets/config.json')
            if self.solver_string == "QBSolv":
                # qbsolv for testing uqp connection without decreasing our quota
                answer = Problem.Qubo(config, qubo_dict).with_platform(
                    "qbsolv").solve(self.sample_count)
            elif self.solver_string == "Advantage_system4.1" or self.solver_string == "DW_2000Q_6":
                # needs dwave quota
                problem = Problem.Qubo(config, qubo_dict).with_platform(
                    "dwave").with_solver(self.solver_string)
                # calculate embedding
                if input_vector is None:
                    problem.embedding = self.find_embedding_with_client(
                        qubo_as_bqm, self.solver_string, False) if self.embedding is None else self.embedding
                answer = problem.solve(self.sample_count)
            elif self.solver_string == "FujitsuDAU":
                # there are many parameters, that can be checked in the UQO examples
                parameters = {
                    "optimization_method": "annealing",
                    "auto_tuning": 2
                }
                # "CPU" for testing (no quota used), "DAU" with quota
                # TODO
                # 0 valued qubits will be kicked out of the qubit
                answer = Problem.Qubo(config, qubo_dict).with_platform("fujitsu").with_solver("DAU").with_params(
                    **parameters).solve(self.sample_count)
            else:
                raise Exception(
                    'No valid solver specified. Valid solvers are "SA", "QBSolv", "DW_2000Q_6", "Advantage_system4.1", "FujitsuDAU"')
            energy = min(answer.energies)
        return energy

    def free_energy(self, input_vector):
        '''Function to compute the free energy'''

        # calculate hidden term
        hidden_term = self.get_energy(input_vector)

        # calculate visible_term
        # visible bias
        visible_term = np.matmul(
            input_vector, self.biases_visible.T) / self.beta_eff
        if not self.restricted:
            # add visible weights
            visible_term = visible_term + \
                (np.matmul(np.matmul(input_vector,
                 self.weights_visible_to_visible), input_vector) / self.beta_eff)

        return hidden_term + visible_term

    # takes samples from clamped phase, i.e. values for hidden neurons, and calculates average, i.e. probabilities of hidden neurons being 1
    def get_average_hidden_samples(self, samples):
        # row = sample, column=neuron, biases
        np_samples = np.vstack(
            tuple([np.array(list(sample.values())) for sample in samples]))
        avgs = np.average(np_samples, axis=0)
        return avgs

    # only used in logistic regression
    def get_hidden_features(self, data):
        features = np.zeros(
            (len(data), self.n_hidden_nodes))
        for i in range(len(data)):
            datapoint = data[i]
            # get samples from QBM
            samples_clamped = self.get_samples(datapoint)
            # average over sample values
            features[i] = self.get_average_hidden_samples(samples_clamped)
        return features

    # done
    def get_average_configuration(self, samples: list, input_vector=None):
        ''' Takes samples from Annealer and averages for each neuron and connection
        '''

        # unclamped if input_vector == None
        unclamped = not (type(input_vector) is np.ndarray)

        # biases (row = sample, column = neuron)
        np_samples = np.vstack(
            tuple([np.array(list(sample.values())) for sample in samples]))
        avgs_biases = np.average(np_samples, axis=0)
        avgs_biases_hidden = avgs_biases[self.dim_input:] if unclamped else avgs_biases
        avgs_biases_visible = avgs_biases[:
                                          self.dim_input] if unclamped else input_vector

        # weights
        avgs_weights_visible_to_hidden = np.zeros(
            self.weights_visible_to_hidden.shape)
        if not self.restricted:
            avgs_weights_visible_to_visible = np.zeros(
                self.weights_visible_to_visible.shape)
        for v in range(self.dim_input):
            # visible to hidden connections
            for h in range(self.n_hidden_nodes):
                x, y = (np_samples[:, v], self.dim_input +
                        h) if unclamped else (input_vector[v], h)
                avgs_weights_visible_to_hidden[v, h] = np.average(
                    x*np_samples[:, y])
            # visible to visible connections
            if not self.restricted:
                for v2 in range(v, self.dim_input):
                    x, y = (np_samples[:, v], np_samples[:, v2]) if unclamped else (
                        input_vector[v], input_vector[v2])
                    avgs_weights_visible_to_visible[v, v2] = np.average(x*y)

        if self.restricted:
            return avgs_biases_hidden, avgs_biases_visible, avgs_weights_visible_to_hidden, None
        else:
            return avgs_biases_hidden, avgs_biases_visible, avgs_weights_visible_to_hidden, avgs_weights_visible_to_visible

    def train_for_one_iteration(self, batch, learning_rate):

        errors_biases_hidden = 0
        errors_biases_visible = 0
        errors_weights_visible_to_hidden = 0
        if not self.restricted:
            errors_weights_visible_to_visible = 0

        for input in batch:
            samples_clamped = self.get_samples(input)
            # avgs_weights_visible_to_visible_clamped only has a value if not restricted
            avgs_bias_hidden_clamped, avgs_bias_visible_clamped, avgs_weights_visible_to_hidden_clamped, avgs_weights_visible_to_visible_clamped = self.get_average_configuration(
                samples_clamped, input)

            samples_unclamped = self.get_samples()
            # avgs_weights_visible_to_visible_unclamped only has a value if not restricted
            avgs_bias_hidden_unclamped, avgs_bias_visible_unclamped, avgs_weights_visible_to_hidden_unclamped, avgs_weights_visible_to_visible_unclamped = self.get_average_configuration(
                samples_unclamped)

            errors_biases_hidden += (avgs_bias_hidden_clamped -
                                     avgs_bias_hidden_unclamped)
            errors_biases_visible += (avgs_bias_visible_clamped -
                                      avgs_bias_visible_unclamped)
            errors_weights_visible_to_hidden += (
                avgs_weights_visible_to_hidden_clamped - avgs_weights_visible_to_hidden_unclamped)
            if not self.restricted:
                errors_weights_visible_to_visible += (
                    avgs_weights_visible_to_visible_clamped - avgs_weights_visible_to_visible_unclamped)

        errors_biases_hidden /= batch.shape[0]
        errors_biases_visible /= batch.shape[0]
        errors_weights_visible_to_hidden /= batch.shape[0]
        if not self.restricted:
            errors_weights_visible_to_visible /= batch.shape[0]

        self.weights_visible_to_hidden -= learning_rate * errors_weights_visible_to_hidden
        self.biases_hidden -= learning_rate * errors_biases_hidden
        self.biases_visible -= learning_rate * errors_biases_visible
        if not self.restricted:
            self.weights_visible_to_visible -= learning_rate * \
                errors_weights_visible_to_visible

        return errors_biases_visible

    def calculate_outlier_threshold(self, quantile=0.95):
        self.quantile = quantile
        energies = np.apply_along_axis(
            self.free_energy, axis=1, arr=self.encoded_data)
        self.outlier_threshold = np.quantile(energies, self.quantile)

    # done ?
    def train_model(self, batch_size=8, learning_rate=0.005):
        data = self.encoded_data
        batch_num = data.shape[0] // batch_size
        diff = data.shape[0] % batch_size
        data = data[:-diff]

        batches = np.vsplit(data, batch_num)
        last_batch = data[data.shape[0] - diff:]
        batches.append(last_batch)

        self.error_container = Errcol(
            self.epochs, self.dim_input * len(batches))

        for epoch in range(1, self.epochs+1):
            print(f'Epoch {epoch}')
            batch_errors = None
            batchnum = 1
            for batch in tqdm(batches):
                try:
                    errors = self.train_for_one_iteration(batch, learning_rate)
                    if type(batch_errors) is np.ndarray:
                        batch_errors = np.hstack((batch_errors, errors))
                    else:
                        batch_errors = errors
                    self.save_weights(
                        f'e{epoch}_b{batchnum}_{self.paramstring}')
                    batchnum += 1
                except Exception as e:
                    self.save_weights(
                        f'e{epoch}_b{batchnum}_{self.paramstring}')
                    raise e
            self.error_container.add_error(batch_errors)
        self.error_container.plot("qbm_errors" + self.paramstring)
        self.save_weights(title="final_weights_qbm" + self.paramstring)
        # make list of values of the error dicts

        self.calculate_outlier_threshold(self.quantile)

    def find_embedding_with_client(self, bqm, solver, save):
        client = Client(token=TOKEN, solver=solver)
        # get graph of specified solver
        graph = client.get_solver().edges
        # find embedding
        embedding = minorminer.find_embedding(bqm.quadratic, graph)

        # save embedding as pdf if parameter save is set to True
        if save:
            if solver == "DW_2000Q_6":
                dwave_networkx.draw_chimera_embedding(dwave_networkx.chimera_graph(16), emb=embedding, node_size=3,
                                                      width=.3)
            elif solver == "Advantage_system4.1":
                dwave_networkx.draw_pegasus_embedding(dwave_networkx.pegasus_graph(16), emb=embedding, node_size=3,
                                                      width=.3)
            path = PurePath()
            path = Path(path / 'embeddings')
            path.mkdir(mode=0o770, exist_ok=True)
            plt.savefig("embeddings/embedding.pdf")

        self.embedding = embedding
        return embedding

    def split_samplelist_according_to_occurences(self, samples, occurences):
        """ Repeat the samples according to their occurence. """
        split_samples = []

        for i in range(0, len(samples)):
            split_samples.extend(repeat(samples[i], occurences[i]))

        return split_samples

    def save_weights(self, title):
        path = PurePath()
        path = Path(path / 'saved_weights')
        path.mkdir(mode=0o770, exist_ok=True)

        if self.restricted:
            np.savez_compressed(path / f'{title}', w_vh=self.weights_visible_to_hidden,
                                b_v=self.biases_visible, b_h=self.biases_hidden)
        else:
            np.savez_compressed(path / f'{title}', w_vh=self.weights_visible_to_hidden,
                                b_v=self.biases_visible, b_h=self.biases_hidden, lat=self.weights_visible_to_visible)

    def load_savepoint(self, file):
        print(f'Loading {file}')
        data = np.load(file)
        self.weights_visible_to_hidden = data['w_vh']
        self.biases_visible = data['b_v']
        self.biases_hidden = data['b_h']
        if not self.restricted:
            self.weights_visible_to_visible = data['lat']
