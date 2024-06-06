import random
import matplotlib.pyplot as plt

# calculate dominance by accuracy and carbon
def dominates(item_a, item_b):
    acc_a, cf_a = item_a
    acc_b, cf_b = item_b
    # a dominates b
    if acc_a >= acc_b and cf_a <= cf_b:     
         return 1
    # b dominates a
    elif acc_a <= acc_b and cf_a >= cf_b:
         return -1
    return 0

def crowd_sort(population, parents_size):
    crowd_sorted_population = []
    dist = [0] * len(population)

    for obj in range(2):    # we only have two objectives(accuracy and carbon) need further change if we have more objectives
        population.sort(key=lambda p: p[obj])
        max_val = population[-1][obj]
        min_val = population[0][obj]
        dist[0] = dist[-1] = float('inf')

        for i in range(1, len(population) - 1):
            cur_dist = dist[i]
            # if cur_dist == float('inf'):
            #     continue
            cur_dist += (population[i + 1][obj] - population[i - 1][obj]) / (max_val - min_val)
            dist[i] = cur_dist
    # put the biggest parent_size distance into return list
    paired = zip(population, dist)
    sorted_pairs = sorted(paired, key=lambda pair: pair[1], reverse=True)
    crowd_sorted_population = [individual for individual, dist in sorted_pairs[:parents_size]]
    return crowd_sorted_population

def fast_non_dominated_sort(population, parents_size):
    S = []  # S[p] = set of solutions; the solution p dominates.
    N = []  # N[p] = domination count; the number of solutions which dominate p.
    sorted_population = []  # sorted population (only for size parents_size)

    # assign ranks to each solution
    for p, p_idx in zip(population, range(len(population))):
        # initialize
        S.append(set())
        N.append(0)
        for q, q_idx in zip(population, range(len(population))):
            if p_idx == q_idx: continue
            judge = dominates(p, q)
            if judge == 1:
                # p dominates q
                S[p_idx].add(q_idx)
            elif judge == -1:
                # q dominates p
                N[p_idx] += 1

    # put in the sorted solution and consider crowded distance for final rank in parents_size
    while parents_size >= N.count(min(N)):
        parents_size -= N.count(min(N))
        process_population = [(index, population[index]) for index, rank in enumerate(N) if rank == min(N)]
        for index, subset in process_population:
            sorted_population.append(subset)
            N[index] = float('inf')
            #  remove the dominance from S
            for i in S[index]:
                N[i] -= 1

    if parents_size > 0:
        last_rank_population = [population[index] for index, rank in enumerate(N) if rank == min(N)]
        sorted_population.extend(crowd_sort(last_rank_population, parents_size))
    return sorted_population





# test NSGA-II
population = [(random.randint(70, 100), random.randint(70, 100)) for _ in range(200)]
# print(population)
new_population = fast_non_dominated_sort(population, 50)

x1, y1 = zip(*population)
x2, y2 = zip(*new_population)

plt.scatter(x1, y1, color='blue', label='origin population')
for point in population:
    if point in new_population:
        plt.scatter(point[0], point[1], color='orange')

plt.xlabel('Accuracy')
plt.ylabel('Carbon')

plt.legend()
plt.show()


			
