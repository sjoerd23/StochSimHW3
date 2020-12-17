import numpy as np
import numba
import sys
import copy
import tsp


@numba.njit
def init_pop(initial_distance, pop_size, t0_low, t0_high, markov_length, markov_low, markov_high):
	"""Initiates population

	Args:


	Returns:
		pop : np array [[t0, markov_multiplier, tot_distance], ...]
			population of individuals with adjustable t0 and markov_multiplier
	"""
	pop = np.arange(pop_size*3, dtype=np.float64).reshape(pop_size, 3)
	for i in range(pop_size):
		pop[i] = [np.random.uniform(t0_low, t0_high),
		markov_length * np.random.uniform(markov_low, markov_high),
		initial_distance]

	return pop


@numba.njit
def mutate(pop):
	"""Mutates individuals in population

	Args:


	Returns:
		pop : np array [[t0, markov_multiplier, tot_distance], ...]
			population of individuals with adjustable t0 and markov_multiplier
	"""
	for i in range(len(pop)):
		for j in range(2):
			pop[i][j] += np.random.normal(0, 1)
			if pop[i][j] < 0.1:
				pop[i][j] = 0.1

	return pop


@numba.njit
def recombine(pop, pop_size, offspring_multiplier=2):
	"""Recombines individuals in population

	Args:


	Returns:
		new_pop : np array [[t0, markov_multiplier, tot_distance], ...]
			population of individuals with adjustable t0 and markov_multiplier
	"""
	new_pop = np.arange(pop_size*offspring_multiplier*3, dtype=np.float64).reshape(pop_size*offspring_multiplier, 3)
	indexes = [index for index in range(len(pop))]

	for i in range(pop_size * offspring_multiplier):

		# select 5 random candidate parents
		n_candidates = 5
		parents_index = np.random.choice(pop_size, n_candidates)
		parents = np.arange(5*3, dtype=np.float64).reshape(n_candidates, 3)
		for j in range(n_candidates):
			parents[j] = pop[parents_index[j]]

		# select best 2
		p1, p2 = sort_pop(parents)[:2]

		if p1[0] < p2[0]:
			new_pop[i] = [np.random.uniform(1.5*p1[0] - 0.5*p2[0], 0.5*p1[0] + 0.5*p2[0]),
				np.random.uniform(0.5*p1[1] + 0.5*p2[1], 0.5*p1[1] + 0.5*p2[1]), p1[2]]
		elif p2[0] < p1[0]:
			new_pop[i] = [np.random.uniform(1.5*p2[0] - 0.5*p1[0], 0.5*p2[0] + 0.5*p1[0]),
				np.random.uniform(0.5*p2[1] + 0.5*p1[1], 0.5*p2[1] + 0.5*p1[1]), p1[2]]
		else:
			new_pop[i] = p1.copy()

		for j in range(2):
			if new_pop[i][j] < 0.1:
				new_pop[i][j] = 0.1

	return new_pop


@numba.njit
def run_genetic(nodes, pop_size, n_generations, n_runs, offspring_multiplier, t0_low, t0_high,
	markov_length, markov_low, markov_high):
	"""Genetic algorithm for parameter estimation of t0 and markov_multiplier

	Args:


	Returns:
		pop : np array [[t0, markov_multiplier, tot_distance], ...]
			population of individuals with adjustable t0 and markov_multiplier
	"""
	initial_distance = tsp.tot_distance(nodes)

	# intialise population
	pop = init_pop(initial_distance, pop_size, t0_low, t0_high, markov_length, markov_low, markov_high)

	for i in range(n_generations):
		print("Calculating generation", i, "of", n_generations)
		pop = recombine(pop, pop_size, offspring_multiplier)
		pop = mutate(pop)

		# evaluate
		for i in range(len(pop)):
			temp = []
			for _ in range(n_runs):
			    sa, dl = tsp.simulated_annealing(nodes.copy(), pop[i][1], pop[i][0])
			    temp.append(tsp.tot_distance(sa))

			# pop[i][2] = np.mean(np.array([tsp.tot_distance(tsp.simulated_annealing(nodes.copy(), pop[i][1], pop[i][0])) for _ in range(n_runs)]))
			pop[i][2] = np.mean(np.array(temp))

		pop = survivor_selection(pop, pop_size)

	# sort pop based on their tot_distance
	pop = sort_pop(pop)

	return pop


@numba.njit
def survivor_selection(pop, pop_size):
	"""Survivor selection

	Args:


	Returns:
		pop : np array [[t0, markov_multiplier, tot_distance], ...]
			population of individuals with adjustable t0 and markov_multiplier
	"""
	# select pop_size best pop
	pop = sort_pop(pop)
	pop = pop[:pop_size]

	return pop


@numba.njit
def sort_pop(pop):
	"""Sort population based on total distance, first element has lowest total distance

	Args:
		pop : np array [[t0, markov_multiplier, tot_distance], ...]
			population of individuals with adjustable t0 and markov_multiplier

	Returns:
		pop : np array [[t0, markov_multiplier, tot_distance], ...]
			population of individuals with adjustable t0 and markov_multiplier
	"""
	return pop[np.argsort(pop[:,2])]
