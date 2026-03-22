import random
import math

def classical_probability(favorable, total):
	return favorable / total

def conditional_probability(p_A_and_B, p_B):
	return p_A_and_B / p_B

def bayes_theorem(p_B_given_A, p_A, p_B):
	return (p_B_given_A * p_A) / p_B

def main():
	print("\n1. Classical Probability")
	prob_head = classical_probability(1, 2)
	print("Probability of Head:", prob_head)

	print("\n2. Conditional Probability")
	p_A_and_B = 0.2
	p_B = 0.5
	print("Conditional Probability P(A|B):", conditional_probability(p_A_and_B, p_B))

	print("\n3. Bayes Theorem")
	p_B_given_A = 0.8
	p_A = 0.3
	p_B = 0.5
	print("Bayes Theorem Result:", bayes_theorem(p_B_given_A, p_A, p_B))

	print("\n4. Permutations and Combinations")
	n = 5
	r = 2
	permutation = math.perm(n, r)
	combination = math.comb(n, r)
	print("Permutation (5P2):", permutation)
	print("Combination (5C2):", combination)

	print("\n5. Experimental Probability (Dice Roll Simulation)")
	trials = 10000
	count_six = 0
	for _ in range(trials):
		if random.randint(1, 6) == 6:
			count_six += 1
	experimental_prob = count_six / trials
	print("Probability of getting 6 (Experimental):", experimental_prob)
	print("\n")

if __name__ == "__main__":
	main()
